// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

/// Индекс токена в словаре
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenId(u32);

impl TokenId {
    /// Создаёт токен из индекса; при переполнении `u32` возвращает `None`
    pub fn from_index(index: usize) -> Option<Self> {
        u32::try_from(index).ok().map(Self)
    }

    /// Возвращает индекс; при переполнении `usize` возвращает `None`
    pub fn index(self) -> Option<usize> {
        usize::try_from(self.0).ok()
    }
}

impl fmt::Display for TokenId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Токенизатор Unicode-символов со словарём в порядке сортировки `char`
#[derive(Debug, PartialEq, Eq)]
pub struct CharTokenizer {
    char_to_id: HashMap<char, TokenId>,
    id_to_char: Vec<char>,
}

impl CharTokenizer {
    /// Строит словарь по корпусу; возвращает ошибку при переполнении [`TokenId`]
    pub fn from_corpus(corpus: &str) -> Result<Self, TokenizerError> {
        let unique_chars: BTreeSet<char> = corpus.chars().collect();
        let mut char_to_id = HashMap::with_capacity(unique_chars.len());
        let mut id_to_char = Vec::with_capacity(unique_chars.len());

        for (idx, &ch) in unique_chars.iter().enumerate() {
            let token_id = TokenId::from_index(idx).ok_or(TokenizerError::TokenIdPosOverflow)?;

            char_to_id.insert(ch, token_id);
            id_to_char.push(ch);
        }

        Ok(Self {
            char_to_id,
            id_to_char,
        })
    }

    /// Кодирует текст в токены; неизвестный символ вызывает ошибку
    pub fn encode(&self, text: &str) -> Result<Vec<TokenId>, TokenizerError> {
        let mut output = Vec::with_capacity(text.chars().count());

        for ch in text.chars() {
            let token = self
                .char_to_id
                .get(&ch)
                .ok_or(TokenizerError::UnknownChar(ch))?;

            output.push(*token);
        }

        Ok(output)
    }

    /// Декодирует токены в строку; неизвестный токен или переполнение индекса вызывает ошибку
    pub fn decode(&self, tokens: &[TokenId]) -> Result<String, TokenizerError> {
        let mut output = String::with_capacity(tokens.len());

        for &id in tokens {
            let token_idx = id.index().ok_or(TokenizerError::TokenIdToUsizeOverflow)?;

            let ch = self
                .id_to_char
                .get(token_idx)
                .ok_or(TokenizerError::UnknownTokenId(id))?;

            output.push(*ch);
        }

        Ok(output)
    }

    /// Возвращает количество токенов в словаре
    pub fn vocab_size(&self) -> usize {
        self.id_to_char.len()
    }
}

/// Ошибка токенизатора
#[derive(Debug, PartialEq)]
pub enum TokenizerError {
    /// Неизвестный токен
    UnknownTokenId(TokenId),
    /// Неизвестный символ
    UnknownChar(char),
    /// Индекс не помещается в [`TokenId`]
    TokenIdPosOverflow,
    /// Индекс не помещается в `usize`
    TokenIdToUsizeOverflow,
}

impl Error for TokenizerError {}

impl fmt::Display for TokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTokenId(token) => {
                write!(formatter, "Unknown token id `{token}`")
            }

            Self::UnknownChar(ch) => {
                write!(formatter, "Character `{ch}` is not present in vocabulary")
            }

            Self::TokenIdPosOverflow => {
                write!(
                    formatter,
                    "Vocabulary contains more tokens than `TokenId` can represent"
                )
            }

            Self::TokenIdToUsizeOverflow => {
                write!(
                    formatter,
                    "Token id cannot be represented as `usize` on this platform"
                )
            }
        }
    }
}
