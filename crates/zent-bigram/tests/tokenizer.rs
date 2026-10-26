// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_bigram::{CharTokenizer, TokenId, TokenizerError};

#[test]
fn roundtrip() {
    let text = "Hello, мир! 你好 🚀";
    let tokenizer = CharTokenizer::from_corpus(text).unwrap();
    let tokens = tokenizer.encode(text).unwrap();
    assert_eq!(tokenizer.decode(&tokens).unwrap(), text);

    assert_eq!(tokenizer.encode("d"), Err(TokenizerError::UnknownChar('d')));
    let unknown = TokenId::from_index(tokenizer.vocab_size()).unwrap();
    assert_eq!(
        tokenizer.decode(&[unknown]),
        Err(TokenizerError::UnknownTokenId(unknown))
    );
}
