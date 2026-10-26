// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use super::{CliArgs, CliArgsError};

fn parse(input: &str) -> Result<CliArgs, CliArgsError> {
    CliArgs::parse(input.split_whitespace().map(str::to_owned))
}

#[test]
fn errors() {
    assert_eq!(
        parse("--train --text abba"),
        Err(CliArgsError::MissingValue("train"))
    );
    assert!(matches!(
        parse("--length -1"),
        Err(CliArgsError::InvalidValue { arg: "length", .. })
    ));
}
