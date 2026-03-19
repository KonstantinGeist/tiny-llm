Простая реализация инференса модели Qwen2.5-0.5B-Instruct на чистом C. Опциональное ускорение через OpenBLAS (12 ток/сек на Intel Core i7).
В ветке qwen3 находится реализация для Qwen3-1.7B (умнее и поддерживает мышление, но медленнее и требует больше RAM).

См. папку `doc/` для объяснения формата GGUF и архитектуры модели Qwen2.5

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

Можно скачать отсюда: https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/blob/main/Qwen2.5-0.5B-Instruct-f16.gguf

Модель довольно примитивная, хорошо общается только на английском.

---

## Использование

```sh
./bin/chat <model.gguf>
```

| Аргумент       | Описание                               |
| -------------- | -------------------------------------- |
| `<model.gguf>` | Путь к файлу модели GGUF (обязательно) |


---

## Удаление артефактов сборки

```sh
make clean
```

