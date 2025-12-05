# План маленьких шагов для улучшения архитектуры

## Принцип: один шаг = одно маленькое изменение, легко откатить

---

## ✅ ШАГ 1: Исправить критический баг (5 минут, безопасно)

**Проблема:** Неправильное сравнение списка в `app.py` (если еще не исправлено)

**Что делать:**
```python
# Было (если есть):
if processed_images is [] or processed_text is None:

# Станет:
if not processed_images or processed_text is None:
```

**Проверка:** Запустить приложение, убедиться что работает

**Откат:** Просто вернуть обратно

---

## ✅ ШАГ 2: Добавить обработку ошибок в одном месте (10 минут)

**Цель:** Защитить от падения при некорректных данных

**Что делать:** Добавить try-except в `ML_server.py` в функции `process()`

```python
@app.route('/process', methods=['POST'])
def process():
    try:
        if 'image' not in request.files:
            return prepare_response(None, 'Изображение не найдено', 400)
        
        image_file = request.files['image']
        image = Image.open(image_file)  # Может упасть
        
        params = request.form
        processed_images, text = process_image(image, params)
        
        return prepare_response(processed_images, text, 200)
    except Exception as e:
        # Логируем ошибку, возвращаем понятное сообщение
        print(f"Ошибка обработки: {e}")  # Пока print, потом заменим на logging
        return prepare_response(None, f'Ошибка обработки: {str(e)}', 500)
```

**Проверка:** Отправить некорректное изображение, убедиться что не падает

**Откат:** Удалить try-except

---

## ✅ ШАГ 3: Вынести константы в начало файла (5 минут)

**Цель:** Убрать магические числа

**Что делать:** В `app.py` и `ML_server.py`

```python
# В начале файла app.py
MAX_IMAGE_SIZE = 1024
MAX_PICTURES = 4
SERVER_URL = "http://127.0.0.1:9001/process"

# Использовать:
if image.size[0] > MAX_IMAGE_SIZE or image.size[1] > MAX_IMAGE_SIZE:
    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
```

**Проверка:** Всё работает как раньше

**Откат:** Вернуть числа обратно в код

---

## ✅ ШАГ 4: Убрать дубликаты импортов (2 минуты)

**Цель:** Чистота кода

**Что делать:** В `app.py` удалить дубликаты:
```python
# Удалить:
import gradio as gr  # второй раз
import random  # второй раз  
import os  # второй раз
```

**Проверка:** Всё работает

**Откат:** Вернуть импорты

---

## ✅ ШАГ 5: Добавить абстракцию для ОДНОЙ модели (BLIP) (30 минут)

**Цель:** Подготовить к Triton, но только для одной модели

**Что делать:** Создать файл `ML_SERVER/inference_wrapper.py`

```python
# ML_SERVER/inference_wrapper.py
"""Обертка для инференса моделей - подготовка к Triton"""

class BlipInference:
    """Обертка для BLIP инференса"""
    
    def __init__(self):
        # Используем существующий код
        from ML_SERVER.blip import BlipImageCaptioner
        self.model = BlipImageCaptioner()
    
    def describe_object(self, image):
        """Описание объекта"""
        return self.model.question(image)

# Глобальный экземпляр (как сейчас)
_blip_inference = None

def get_blip_inference():
    """Получить экземпляр BLIP инференса (singleton)"""
    global _blip_inference
    if _blip_inference is None:
        _blip_inference = BlipInference()
    return _blip_inference
```

**Изменить в `ML_SERVER/blip.py`:**
```python
# Добавить в конец файла (не трогая существующий код):
from ML_SERVER.inference_wrapper import get_blip_inference

def describe_center_object_v2(image):
    """Новая версия через обертку"""
    inference = get_blip_inference()
    return inference.describe_object(image)
```

**Использовать в `ML_SERVER/processor.py`:**
```python
# Пока оставить старый вызов, но добавить комментарий:
# TODO: Перейти на describe_center_object_v2
description = describe_center_object(image)  # Старый способ
```

**Проверка:** Всё работает как раньше

**Откат:** Удалить файл `inference_wrapper.py`, убрать изменения

---

## ✅ ШАГ 6: Переключить ОДНУ функцию на новую обертку (5 минут)

**Цель:** Протестировать абстракцию на практике

**Что делать:** В `ML_SERVER/processor.py` заменить вызов:

```python
# Было:
from ML_SERVER.blip import describe_center_object

# Стало:
from ML_SERVER.inference_wrapper import get_blip_inference

# В функции:
inference = get_blip_inference()
description = inference.describe_object(image)
```

**Проверка:** Всё работает

**Откат:** Вернуть старый вызов

---

## ✅ ШАГ 7: Добавить конфигурацию через переменные окружения (10 минут)

**Цель:** Убрать хардкод

**Что делать:** Создать `config.py`

```python
# config.py
import os

# Сервер
ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://127.0.0.1:9001/process")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))
MAX_OBJECTS = int(os.getenv("MAX_OBJECTS", "4"))

# Окружение
ENV = os.getenv("ENV", "local")

# SSL (только для продакшена)
CERTIFICATE_PATH = os.getenv("CERTIFICATE_PATH", "/home/yrsolo/tg-det/https-cert/certificate.pem")
PRIVATE_KEY_PATH = os.getenv("PRIVATE_KEY_PATH", "/home/yrsolo/tg-det/https-cert/private_key.pem")
```

**Использовать в `app.py`:**
```python
from config import ML_SERVER_URL, MAX_IMAGE_SIZE

SERVER_URL = ML_SERVER_URL  # Вместо хардкода
```

**Проверка:** Всё работает

**Откат:** Вернуть хардкод

---

## ✅ ШАГ 8: Заменить print на logging (15 минут)

**Цель:** Профессиональное логирование

**Что делать:** В начале `ML_server.py`:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Заменить print на logger:
# print(request.form) → logger.info(f"Request params: {request.form}")
```

**Проверка:** Логи появляются в консоли

**Откат:** Вернуть print

---

## ✅ ШАГ 9: Добавить валидацию размера файла (10 минут)

**Цель:** Защита от DoS

**Что делать:** В `ML_server.py`:

```python
from config import MAX_IMAGE_SIZE

@app.route('/process', methods=['POST'])
def process():
    image_file = request.files['image']
    
    # Проверка размера файла
    image_file.seek(0, 2)  # Переместиться в конец
    file_size = image_file.tell()
    image_file.seek(0)  # Вернуться в начало
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    if file_size > MAX_FILE_SIZE:
        return prepare_response(None, 'Файл слишком большой', 400)
    
    # Остальной код...
```

**Проверка:** Большой файл отклоняется

**Откат:** Удалить проверку

---

## ✅ ШАГ 10: Выключить debug в продакшене (2 минуты)

**Цель:** Безопасность

**Что делать:** В `ML_server.py`:

```python
from config import ENV

if __name__ == "__main__":
    debug_mode = ENV != "production"
    app.run(debug=debug_mode, host="0.0.0.0", port=9001)
```

**Проверка:** В production debug выключен

**Откат:** Вернуть debug=True

---

## Порядок выполнения

Рекомендую делать по порядку, проверяя каждый шаг:

1. ✅ Шаг 1 (баг) - обязательно
2. ✅ Шаг 2 (обработка ошибок) - важно
3. ✅ Шаг 3-4 (константы, импорты) - легко
4. ✅ Шаг 5-6 (абстракция для BLIP) - подготовка к Triton
5. ✅ Шаг 7-10 (конфиг, логирование, безопасность) - улучшения

## Правила безопасности

1. **Делать один шаг за раз**
2. **Тестировать после каждого шага**
3. **Коммитить после каждого рабочего шага** (легко откатить)
4. **Не трогать рабочий код** - только добавлять/заменять маленькими кусочками

## Если что-то сломалось

1. Откатить последний шаг (git revert)
2. Разобраться почему
3. Исправить и повторить

