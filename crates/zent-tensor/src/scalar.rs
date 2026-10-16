// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::fmt;

/// Тип элемента с нулём и единицей
pub trait Scalar: Copy {
    /// Нулевое значение типа
    const ZERO: Self;
    /// Единичное значение типа
    const ONE: Self;
}

macro_rules! impl_scalar {
    ($zero:expr, $one:expr; $($ty:ty),+ $(,)?) => {
        $(
            impl Scalar for $ty {
                const ZERO: Self = $zero;
                const ONE: Self = $one;
            }
        )+
    };
}

impl_scalar!(0, 1; i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
impl_scalar!(0.0, 1.0; f32, f64);

/// Сложение двух значений
pub trait ScalarAdd: Sized {
    /// Возвращает сумму
    fn try_add(self, rhs: Self) -> Result<Self, ArithmeticError>;
}

/// Вычитание двух значений
pub trait ScalarSub: Sized {
    /// Возвращает разность
    fn try_sub(self, rhs: Self) -> Result<Self, ArithmeticError>;
}

/// Умножение двух значений
pub trait ScalarMul: Sized {
    /// Возвращает произведение
    fn try_mul(self, rhs: Self) -> Result<Self, ArithmeticError>;
}

/// Деление двух значений
pub trait ScalarDiv: Sized {
    /// Возвращает частное; для целых округляет к нулю
    /// Нулевой делитель у целых даёт [`ArithmeticError::DivisionByZero`]
    fn try_div(self, rhs: Self) -> Result<Self, ArithmeticError>;
}

/// Смена знака значения
pub trait ScalarNeg: Sized {
    /// Возвращает значение с противоположным знаком
    /// Для беззнаковых целых допустим только ноль
    fn try_neg(self) -> Result<Self, ArithmeticError>;
}

/// Арифметическая ошибка
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticError {
    /// Переполнение
    Overflow,
    /// Деление на ноль
    DivisionByZero,
}

impl Error for ArithmeticError {}

impl fmt::Display for ArithmeticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => write!(formatter, "arithmetic overflow"),
            Self::DivisionByZero => write!(formatter, "integer division by zero"),
        }
    }
}
