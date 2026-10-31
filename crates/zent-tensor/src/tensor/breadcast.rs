#![allow(dead_code)]
/*
Дальше работу я бы разделил на три небольших подшага.
1. Вычисление общей формы
   Функция получает две &Shape и возвращает новую Shape либо ошибку. Значения тензоров ей не нужны, поэтому здесь нет зависимости от типа элементов и арифметических трейтов.
   Для обхода размеров с конца пригодится dims().iter().rev(): iter() выдаёт ссылки на размеры, rev() меняет порядок обхода. Обычного zip двух списков недостаточно, поскольку нужно обработать и оставшиеся ведущие оси более длинной формы.
   Общую форму следует создавать через Shape::new, чтобы сохранить проверку переполнения числа элементов.
2. Сопоставление позиций
   По общей форме и шагам входов подготовить правила чтения. Для каждой позиции результата обход должен выдавать пару линейных индексов: один для левого входа, второй для правого.
   Пары нужно получать по ходу вычисления. Достаточно хранить метаданные осей и текущую позицию, без массива всех пар индексов.
3. Подключение к существующим операциям
   Начать с try_add, создающего новый результат. Затем подключить тот же способ адресации к остальным парным операциям, включая поэлементную степень, и после этого разобрать _assign и _into.
   Сейчас try_zip_map выделяет место по self.numel() и создаёт результат с self.shape. Для разных совместимых форм потребуется использовать количество элементов и форму общего результата.
*/

use crate::{
    Shape, Strides,
    TensorError::{self, BroadcastMismatch},
};

/// Возвращает общую форму для поэлементной операции
pub(super) fn broadcast_shape(left: &Shape, right: &Shape) -> Result<Shape, TensorError> {
    // Удобный порядок такой: развернуть обход размеров,
    // дополнить короткую форму единицами и затем соединить оба обхода через zip

    let rank = left.rank().max(right.rank());
    let missing_left = rank - left.rank();
    let missing_right = rank - right.rank();

    let right_axes = right
        .dims()
        .iter()
        .rev()
        .copied()
        .chain(std::iter::repeat_n(1, missing_right));

    let left_axes = left
        .dims()
        .iter()
        .rev()
        .copied()
        .chain(std::iter::repeat_n(1, missing_left));

    let mut new_dims = Vec::with_capacity(rank);

    left_axes
        .zip(right_axes)
        .try_for_each(|(left_size, right_size)| {
            let size = match (left_size, right_size) {
                (1, size) | (size, 1) => size,
                (left, right) if left == right => left,
                _ => {
                    return Err(BroadcastMismatch {
                        left: left.clone(),
                        right: right.clone(),
                    });
                }
            };

            new_dims.push(size);

            Ok(())
        })?;

    new_dims.reverse();

    Ok(Shape::new(new_dims)?)
}

/// Возвращает шаги чтения тензора по осям общей формы
pub(super) fn broadcast_strides(
    input_shape: &Shape,
    input_strides: &Strides,
    common_shape: &Shape,
) -> Vec<usize> {
    // После согласования форм:
    // ---
    // Если соответствующей оси у входного тензора нет, шаг чтения равен 0
    // Если размер оси входного тензора = 1, шаг чтения тоже = 0: вдоль этой оси доступна только одна позиция
    // В остальных случаях берём существующий шаг этой оси из tensor.strides()

    let input_axes = input_shape.dims().iter().zip(input_strides.as_slice());

    let mut strides = Vec::with_capacity(common_shape.rank());

    let missing = common_shape.rank() - input_shape.rank();
    strides.resize(missing, 0);

    input_axes.for_each(|(&axis_size, &axis_stride)| {
        let stride = if axis_size == 1 { 0 } else { axis_stride };
        strides.push(stride);
    });

    strides
}
