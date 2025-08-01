# Инструкция по установке Giscard

## Способ 1: Автоматическая установка (рекомендуемый)

Запустите скрипт установки:
```bash
python install_giscard.py
```

## Способ 2: Ручная установка

### Шаг 1: Установите базовые зависимости
```bash
pip install google-generativeai requests transformers torch datasets evaluate
```

### Шаг 2: Попробуйте установить Giscard из GitHub
```bash
pip install git+https://github.com/giscard-ai/giscard.git
```

### Шаг 3: Если не работает, попробуйте альтернативные источники
```bash
# Вариант 1
pip install giscard-ai

# Вариант 2
pip install git+https://github.com/microsoft/giscard.git

# Вариант 3
pip install git+https://github.com/giscard-ai/giscard-python.git
```

## Способ 3: Установка через conda (если используете Anaconda)
```bash
conda install -c conda-forge giscard
```

## Способ 4: Клонирование и установка из исходников
```bash
git clone https://github.com/giscard-ai/giscard.git
cd giscard
pip install -e .
```

## Проверка установки

После установки запустите:
```bash
python -c "import giscard; print('Giscard установлен успешно!')"
```

## Если Giscard не устанавливается

Не беспокойтесь! Программа автоматически переключится на использование Gemini для:
- Генерации вопросов
- Оценки ответов

Это обеспечивает полную функциональность даже без Giscard.

## Требования к системе

- Python 3.7+
- Минимум 4GB RAM (для локального Giscard)
- Интернет-соединение (для загрузки моделей)
- Ключ API Google Gemini

## Устранение неполадок

### Ошибка "No module named 'giscard'"
```bash
pip install --upgrade pip
pip install giscard --no-cache-dir
```

### Ошибка с torch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Ошибка с transformers
```bash
pip install transformers[torch]
``` 