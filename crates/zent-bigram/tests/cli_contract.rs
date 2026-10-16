// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_zent"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn status() {
    for (flag, code) in [("--help", 0), ("-h", 0), ("--unknown", 1)] {
        let output = run(&[flag]);
        assert_eq!(output.status.code(), Some(code), "{flag}");
        let message = if code == 0 {
            output.stdout
        } else {
            output.stderr
        };
        assert!(!message.is_empty(), "{flag}");
    }
}

#[test]
fn input_sources() {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("zent-{}-{suffix}.txt", std::process::id()));
    let run_file = |text: &str| run(&["--train", path.to_str().unwrap(), "--text", text]);

    let inline = run(&["--text", "abba", "--seed", "42", "--length", "8"]);
    fs::write(&path, "abba").unwrap();
    let file_over_text = run_file("");
    fs::write(&path, "").unwrap();
    let empty_file_over_text = run_file("abba");
    fs::remove_file(&path).unwrap();

    assert!(inline.status.success());
    assert!(file_over_text.status.success());
    assert_eq!(empty_file_over_text.status.code(), Some(1));
}
