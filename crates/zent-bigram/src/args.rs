// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

//! Аргументы командной строки

use std::error::Error;
use std::path::PathBuf;

/// Аргументы запуска
#[derive(Debug, PartialEq, Eq)]
pub struct CliArgs {
    pub text: Option<String>,
    pub train: Option<PathBuf>,
    pub seed: Option<u64>,
    pub length: Option<usize>,
}

impl CliArgs {
    /// Разбирает аргументы и отклоняет неизвестные ключи
    pub fn parse(raw_args: impl Iterator<Item = String>) -> Result<CliArgs, CliArgsError> {
        let mut text = None;
        let mut train = None;
        let mut seed = None;
        let mut length = None;
        let mut args = raw_args.peekable();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--text" => {
                    text = Some(
                        args.next_if(|value| !value.starts_with("--"))
                            .unwrap_or_default(),
                    );
                }
                "--train" => {
                    let value = next_value(&mut args, "train")?;

                    if !value.is_empty() {
                        train = Some(PathBuf::from(value));
                    }
                }
                "--seed" => {
                    let value = next_value(&mut args, "seed")?;

                    let parsed_seed =
                        value
                            .parse::<u64>()
                            .map_err(|_| CliArgsError::InvalidValue {
                                arg: "seed",
                                value,
                                expected: "u64",
                            })?;

                    seed = Some(parsed_seed);
                }
                "--length" => {
                    let value = next_value(&mut args, "length")?;

                    let parsed_length =
                        value
                            .parse::<usize>()
                            .map_err(|_| CliArgsError::InvalidValue {
                                arg: "length",
                                value,
                                expected: "usize",
                            })?;

                    length = Some(parsed_length);
                }
                "--help" | "-h" => return Err(CliArgsError::Usage),
                _ => return Err(CliArgsError::UnknownArg(arg)),
            }
        }

        Ok(Self {
            text,
            train,
            seed,
            length,
        })
    }
}

/// Читает значение аргумента; другой ключ не считается значением
fn next_value(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    name: &'static str,
) -> Result<String, CliArgsError> {
    let value = args.next().ok_or(CliArgsError::MissingValue(name))?;

    if value.starts_with("--") {
        return Err(CliArgsError::MissingValue(name));
    }

    Ok(value)
}

/// Проверяет наличие обязательного аргумента
#[allow(dead_code)]
fn required_arg<T>(value: Option<T>, name: &'static str) -> Result<T, CliArgsError> {
    value.ok_or(CliArgsError::MissingArg(name))
}

/// Возвращает справку CLI
pub fn usage() -> &'static str {
    "\
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
"
}

/// Ошибка аргументов CLI
#[derive(Debug, PartialEq, Eq)]
pub enum CliArgsError {
    /// Недопустимое значение
    InvalidValue {
        value: String,
        arg: &'static str,
        expected: &'static str,
    },

    /// Нет обязательного аргумента
    MissingArg(&'static str),

    /// Нет значения аргумента
    MissingValue(&'static str),

    /// Неизвестный аргумент
    UnknownArg(String),

    /// Запрошена справка
    Usage,
}

impl Error for CliArgsError {}

impl std::fmt::Display for CliArgsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidValue {
                value,
                arg,
                expected,
            } => {
                write!(
                    formatter,
                    "invalid value `{value}` for argument `{arg}`: expected {expected}"
                )
            }
            Self::MissingArg(arg) => {
                write!(formatter, "missing required argument `{arg}`")
            }
            Self::MissingValue(arg) => {
                write!(formatter, "missing value for argument `{arg}`")
            }
            Self::UnknownArg(arg) => {
                write!(formatter, "unknown argument `{arg}`")
            }
            Self::Usage => {
                write!(formatter, "{}", usage())
            }
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/args.rs"]
mod tests;
