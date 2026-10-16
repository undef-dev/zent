// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use super::{CliArgs, CliArgsError};

fn parse(input: &str) -> Result<CliArgs, CliArgsError> {
    CliArgs::parse(input.split_whitespace().map(str::to_owned))
}

#[test]
fn options() {
    let args = parse("--text abba --train corpus.txt --seed 42 --length 0").unwrap();
    assert_eq!(
        args,
        CliArgs {
            text: Some("abba".into()),
            train: Some("corpus.txt".into()),
            seed: Some(42),
            length: Some(0),
        }
    );
    assert_eq!(parse("--text").unwrap().text, Some(String::new()));
}

#[test]
fn errors() {
    for name in ["train", "seed", "length"] {
        for suffix in ["", " --text abba"] {
            assert_eq!(
                parse(&format!("--{name}{suffix}")),
                Err(CliArgsError::MissingValue(name))
            );
        }
    }
    for name in ["seed", "length"] {
        for value in ["abc", "-1"] {
            assert!(
                matches!(
                    parse(&format!("--{name} {value}")),
                    Err(CliArgsError::InvalidValue { arg, .. }) if arg == name
                ),
                "{name} {value}"
            );
        }
    }
}
