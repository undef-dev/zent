// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const EXPECTED_HELP: &str = "\
Zent

Usage:
    zent --text <text> [options]
    zent --train <path> [options]

Modes:
    --text <text>
        Run the model with the provided input text

    --train <path>
        Train the model using data from the specified file
        Takes precedence over --text when both are provided

Options:
    --seed <u64>
        Seed for the pseudorandom number generator
        Default: 0

    --length <usize>
        Number of tokens to generate.
        Default: 0

    -h, --help
        Show this help message
";

const EXPECTED_ABBA_OUTPUT: &str = "\
Corpus:
  bytes:      4
  chars:      4
  tokens:     4
  vocabulary: 2

Tokens: [TokenId(0), TokenId(1), TokenId(1), TokenId(0)]
Decoded tokens: abba

Bigram transitions:
'a' [0]
  -> 'b' [1]: count=1, p=1.000000
'b' [1]
  -> 'a' [0]: count=1, p=0.500000
  -> 'b' [1]: count=1, p=0.500000

Generated: `bababbab`
";

fn run_zent(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_zent"))
        .args(args)
        .output()
        .expect("failed to run the zent binary")
}

fn stdout(output: &Output) -> String {
    normalize_newlines(
        String::from_utf8(output.stdout.clone()).expect("stdout must contain valid UTF-8"),
    )
}

fn stderr(output: &Output) -> String {
    normalize_newlines(
        String::from_utf8(output.stderr.clone()).expect("stderr must contain valid UTF-8"),
    )
}

fn normalize_newlines(text: String) -> String {
    text.replace("\r\n", "\n")
}

#[test]
fn help_preserves_cli_contract() {
    let output = run_zent(&["--help"]);

    assert!(output.status.success());
    assert_eq!(stdout(&output).trim_end(), EXPECTED_HELP.trim_end());
    assert!(stderr(&output).is_empty());
}

#[test]
fn unknown_argument_exits_with_error() {
    let output = run_zent(&["--unknown"]);

    assert_eq!(output.status.code(), Some(1));
    assert!(stdout(&output).is_empty());
    assert_eq!(
        stderr(&output).trim_end(),
        "Error: unknown argument `--unknown`"
    );
}

#[test]
fn seeded_generation_preserves_level_zero_output() {
    let output = run_zent(&["--text", "abba", "--seed", "42", "--length", "8"]);

    assert!(output.status.success());
    assert_eq!(stdout(&output).trim_end(), EXPECTED_ABBA_OUTPUT.trim_end());
    assert!(stderr(&output).is_empty());
}

#[test]
fn train_file_takes_precedence_over_inline_text() {
    let unique_suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be later than the Unix epoch")
        .as_nanos();
    let corpus_path = std::env::temp_dir().join(format!(
        "zent-cli-contract-{}-{unique_suffix}.txt",
        std::process::id()
    ));

    fs::write(&corpus_path, "abba").expect("failed to create a temporary corpus");

    let output = Command::new(env!("CARGO_BIN_EXE_zent"))
        .args(["--text", "zzzz", "--train"])
        .arg(&corpus_path)
        .args(["--seed", "42", "--length", "8"])
        .output()
        .expect("failed to run the zent binary");

    fs::remove_file(&corpus_path).expect("failed to remove the temporary corpus");

    assert!(output.status.success());
    assert_eq!(stdout(&output).trim_end(), EXPECTED_ABBA_OUTPUT.trim_end());
    assert!(stderr(&output).is_empty());
}
