// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_bigram::{CharTokenizer, TokenId, TokenizerError};

#[test]
fn roundtrip() {
    for text in ["", "Hello, мир! 你好 🚀"] {
        let tokenizer = CharTokenizer::from_corpus(text).unwrap();
        let tokens = tokenizer.encode(text).unwrap();
        assert_eq!(tokens.len(), text.chars().count());
        assert_eq!(tokenizer.decode(&tokens).unwrap(), text);
    }
}

#[test]
fn vocabulary() {
    let tokenizer = CharTokenizer::from_corpus("bbacac").unwrap();
    let tokens = tokenizer.encode("abc").unwrap();
    let indices: Vec<_> = tokens.iter().map(|token| token.index().unwrap()).collect();
    assert_eq!(tokenizer.vocab_size(), 3);
    assert_eq!(indices, [0, 1, 2]);

    assert_eq!(
        tokenizer.encode("abcd"),
        Err(TokenizerError::UnknownChar('d'))
    );
    let unknown = TokenId::from_index(3).unwrap();
    assert_eq!(
        tokenizer.decode(&[unknown]),
        Err(TokenizerError::UnknownTokenId(unknown))
    );
}
