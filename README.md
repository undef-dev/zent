# Zent

Экспериментальный ML-фреймворк на Rust, создаваемый с фундаментального уровня.

## Состояние проекта

- `zent-bigram` — символьная биграммная модель: токенизация, обучение на корпусе, генерация с заданным seed и CLI.
- `zent-tensor` — разрабатываемое тензорное ядро. Реализованы форма, шаги, хранилище и конструкторы; следующий этап — индексирование.

Дальнейшие этапы перечислены в [ROADMAP.md](ROADMAP.md).

## Быстрый старт

Нужны Rust с поддержкой edition 2024 и Cargo.

```console
git clone https://github.com/undef-dev/zent.git
cd zent
cargo run -p zent-bigram -- --text "abba" --seed 42 --length 8
```

Для обучения на файле корпуса:

```console
cargo run -p zent-bigram -- --train data/corpus.txt --seed 42 --length 100
```

## Параметры CLI

- `--text <text>` — текст корпуса.
- `--train <path>` — файл корпуса; имеет приоритет над `--text`.
- `--seed <u64>` — seed генератора, по умолчанию `0`.
- `--length <usize>` — число генерируемых токенов, по умолчанию `0`.
- `-h`, `--help` — справка.

## Разработка

Подготовка изменений и команды проверки — в [CONTRIBUTING.md](CONTRIBUTING.md).

Документация API:

```console
cargo doc --workspace --no-deps --open
```

## Лицензия

[Apache-2.0](LICENSE.md). Коммерческое использование разрешено на условиях лицензии.
