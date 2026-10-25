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
    fn try_div(self, rhs: Self) -> Result<Self, ArithmeticError>;
}

/// Смена знака значения
pub trait ScalarNeg: Sized {
    /// Возвращает значение с противоположным знаком
    fn try_neg(self) -> Result<Self, ArithmeticError>;
}

/// Возведение в степень с показателем типа `Exponent`
pub trait ScalarPow<Exponent>: Sized {
    /// Возвращает степень; ноль в нулевой степени даёт единицу
    fn try_pow(self, exponent: Exponent) -> Result<Self, ArithmeticError>;
}

/// Модуль значения
pub trait ScalarAbs: Sized {
    /// Возвращает модуль; минимум знакового целого даёт переполнение
    fn try_abs(self) -> Result<Self, ArithmeticError>;
}

/// Квадратный корень для чисел с плавающей точкой
pub trait ScalarSqrt: Sized {
    /// Возвращает квадратный корень; отрицательный аргумент даёт `NaN`
    fn try_sqrt(self) -> Result<Self, ArithmeticError>;
}

/// Экспонента для чисел с плавающей точкой
pub trait ScalarExp: Sized {
    /// Возвращает `e` в степени исходного значения
    fn try_exp(self) -> Result<Self, ArithmeticError>;
}

/// Натуральный логарифм для чисел с плавающей точкой
pub trait ScalarLn: Sized {
    /// Возвращает натуральный логарифм значения
    fn try_ln(self) -> Result<Self, ArithmeticError>;
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

macro_rules! impl_scalar_pow {
    (unsigned, $ty:ty; $($exponent:ty),+ $(,)?) => {
        $(
            impl ScalarPow<$exponent> for $ty {
                fn try_pow(self, mut exponent: $exponent) -> Result<Self, ArithmeticError> {
                    let mut base = self;
                    // Нулевая степень сразу даст единицу
                    let mut result = Self::ONE;

                    while exponent != 0 {
                        // Для нечётного показателя учитываем текущий множитель
                        if exponent & 1 != 0 {
                            result = ScalarMul::try_mul(result, base)?;
                        }
                        // Делим показатель на два без изменения его типа
                        exponent >>= 1;
                        // Лишний квадрат мог бы вызвать ненужное переполнение
                        if exponent != 0 {
                            base = ScalarMul::try_mul(base, base)?;
                        }
                    }

                    Ok(result)
                }
            }
        )+
    };
    (signed, $ty:ty; $($exponent:ty),+ $(,)?) => {
        $(
            impl ScalarPow<$exponent> for $ty {
                fn try_pow(self, exponent: $exponent) -> Result<Self, ArithmeticError> {
                    // Для отрицательной степени берём обратное основание
                    let base = if exponent < 0 { Self::ONE / self } else { self };
                    // Модуль сохраняет разрядность и корректно обрабатывает MIN
                    ScalarPow::try_pow(base, exponent.unsigned_abs())
                }
            }
        )+
    };
}

macro_rules! impl_scalar_abs {
    (signed, $ty:ty) => {
        impl ScalarAbs for $ty {
            fn try_abs(self) -> Result<Self, ArithmeticError> {
                self.checked_abs().ok_or(ArithmeticError::Overflow)
            }
        }
    };
    (unsigned, $ty:ty) => {
        impl ScalarAbs for $ty {
            fn try_abs(self) -> Result<Self, ArithmeticError> {
                Ok(self)
            }
        }
    };
}

macro_rules! impl_scalar_int {
    ($kind:ident; $($ty:ty),+ $(,)?) => {
        $(
            impl ScalarAdd for $ty {
                fn try_add(self, rhs: Self) -> Result<Self, ArithmeticError> {
                    self.checked_add(rhs).ok_or(ArithmeticError::Overflow)
                }
            }

            impl ScalarSub for $ty {
                fn try_sub(self, rhs: Self) -> Result<Self, ArithmeticError> {
                    self.checked_sub(rhs).ok_or(ArithmeticError::Overflow)
                }
            }

            impl ScalarMul for $ty {
                fn try_mul(self, rhs: Self) -> Result<Self, ArithmeticError> {
                    self.checked_mul(rhs).ok_or(ArithmeticError::Overflow)
                }
            }

            impl ScalarDiv for $ty {
                fn try_div(self, rhs: Self) -> Result<Self, ArithmeticError> {
                    self.checked_div(rhs).ok_or_else(|| {
                        if rhs == 0 {
                            ArithmeticError::DivisionByZero
                        } else {
                            ArithmeticError::Overflow
                        }
                    })
                }
            }

            impl ScalarNeg for $ty {
                fn try_neg(self) -> Result<Self, ArithmeticError> {
                    self.checked_neg().ok_or(ArithmeticError::Overflow)
                }
            }

            impl_scalar_pow!(unsigned, $ty; u8, u16, u32, u64, u128, usize);
            impl_scalar_abs!($kind, $ty);
        )+
    };
}

impl_scalar_int!(signed; i8, i16, i32, i64, i128, isize);
impl_scalar_int!(unsigned; u8, u16, u32, u64, u128, usize);

macro_rules! impl_scalar_float {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ScalarAdd for $ty {
                fn try_add(self, rhs: Self) -> Result<Self, ArithmeticError> {
                    Ok(self + rhs)
                }
            }

            impl ScalarSub for $ty {
                fn try_sub(self, rhs: Self) -> Result<Self, ArithmeticError> {
                    Ok(self - rhs)
                }
            }

            impl ScalarMul for $ty {
                fn try_mul(self, rhs: Self) -> Result<Self, ArithmeticError> {
                    Ok(self * rhs)
                }
            }

            impl ScalarDiv for $ty {
                fn try_div(self, rhs: Self) -> Result<Self, ArithmeticError> {
                    Ok(self / rhs)
                }
            }

            impl ScalarNeg for $ty {
                fn try_neg(self) -> Result<Self, ArithmeticError> {
                    Ok(-self)
                }
            }

            impl ScalarAbs for $ty {
                fn try_abs(self) -> Result<Self, ArithmeticError> {
                    Ok(self.abs())
                }
            }

            impl_scalar_pow!(unsigned, $ty; u8, u16, u32, u64, u128, usize);
            impl_scalar_pow!(signed, $ty; i8, i16, i32, i64, i128, isize);

            impl ScalarPow<$ty> for $ty {
                fn try_pow(self, exponent: Self) -> Result<Self, ArithmeticError> {
                    Ok(self.powf(exponent))
                }
            }
        )+
    };
}

impl_scalar_float!(f32, f64);
