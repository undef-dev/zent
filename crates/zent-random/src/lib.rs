// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

//! Воспроизводимые источники случайных чисел

mod rng;

pub use rng::{GOLDEN_GAMMA, MIX_MULTIPLIER_1, MIX_MULTIPLIER_2, RandomSource, Rng};
