// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

//! Генератор SplitMix64 и общий источник случайных чисел

pub const GOLDEN_GAMMA: u64 = 0x9E3779B97F4A7C15;
pub const MIX_MULTIPLIER_1: u64 = 0xBF58476D1CE4E5B9;
pub const MIX_MULTIPLIER_2: u64 = 0x94D049BB133111EB;

/// SplitMix64: одинаковый `seed` даёт одинаковую последовательность
#[derive(Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Создаёт генератор с начальным состоянием `seed`, включая ноль
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Возвращает следующее псевдослучайное число `u64`
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);

        let mut random = self.state;
        random = (random ^ (random >> 30)).wrapping_mul(MIX_MULTIPLIER_1);
        random = (random ^ (random >> 27)).wrapping_mul(MIX_MULTIPLIER_2);
        random ^= random >> 31;

        random
    }

    /// Возвращает число в диапазоне `0.0..1.0` из старших 24 бит случайного `u64`
    pub fn next_f32(&mut self) -> f32 {
        let random = (self.next_u64() >> 40) as f32;

        random / (1_u32 << 24) as f32
    }

    /// Возвращает число в диапазоне `0.0..1.0` из старших 53 бит случайного `u64`
    pub fn next_f64(&mut self) -> f64 {
        let random = (self.next_u64() >> 11) as f64;

        random / (1_u64 << 53) as f64
    }
}

/// Источник случайных чисел типа f64
pub trait RandomSource {
    /// Возвращает следующее число в диапазоне `0.0..1.0`
    fn next_f64(&mut self) -> f64;
}

impl RandomSource for Rng {
    fn next_f64(&mut self) -> f64 {
        Rng::next_f64(self)
    }
}
