Простая реализация инференса модели Qwen3-1.7B-Instruct на чистом C. Опциональное ускорение через OpenBLAS (3 ток/сек на Intel Core i7).

Данная версия не оптимизирована в угоду читаемости. В ветке qwen3-fast можно найти реализацию в x4 раза быстрее.

См. папку `doc/` для объяснения формата GGUF и архитектуры модели Qwen3

Пример:
```
>> what is the capital of France?

The capital of France is Paris.
```

---

## Сборка

Для сборки требуется Linux или WSL2 под Windows.

```sh
make
```

Исполняемый файл будет находиться в `bin/chat`.

---

### OpenBLAS (быстрее на CPU)

```sh
make BLAS=1
```

Требуется OpenBLAS (`libopenblas-dev` на Debian/Ubuntu, `openblas` на Arch/Homebrew).

---

### Веса моделей

Можно скачать отсюда: https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf

Модель довольно примитивная, хорошо общается только на английском.

---

## Использование

```sh
./bin/chat [--think] <model.gguf>
```

| Аргумент       | Описание                               |
| -------------- | -------------------------------------- |
| `<model.gguf>` | Путь к файлу модели GGUF (обязательно) |
| `--think`      | Включить режим размышлений Qwen3 (по умолчанию выключен) |


---

## Удаление артефактов сборки

```sh
make clean
```

