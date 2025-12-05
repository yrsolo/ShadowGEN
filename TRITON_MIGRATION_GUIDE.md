# Руководство по подготовке к миграции на Triton Inference Server

## Что такое Triton Inference Server?

**Triton** - это сервер для ML инференса от NVIDIA, который:
- Деплоит модели как отдельные сервисы
- Автоматически делает батчинг запросов
- Поддерживает разные фреймворки (PyTorch, TensorRT, ONNX, TensorFlow)
- Масштабируется горизонтально (несколько GPU/серверов)
- Оптимизирует использование GPU автоматически
- Предоставляет REST/gRPC API для инференса

## Текущая архитектура vs Triton

### Сейчас:
```
Python процесс → Глобальные модели в памяти → Прямой вызов model.forward()
```

### С Triton:
```
Python процесс → HTTP/gRPC запрос → Triton Server → Модель на GPU → Ответ
```

## Ключевые изменения в архитектуре

### 1. Модели становятся отдельными сервисами

**Сейчас:**
```python
# Модель загружается при импорте модуля
print('Loading Blip model...')
blip = BlipImageCaptioner()  # Глобальная переменная
print('Blip model loaded')

def describe_center_object(image):
    return blip.question(image)  # Прямой вызов
```

**С Triton:**
```python
# Модель работает как отдельный сервис
import tritonclient.http as httpclient

triton_client = httpclient.InferenceServerClient(url="localhost:8000")

def describe_center_object(image):
    # Отправляем запрос в Triton
    inputs = prepare_blip_inputs(image)
    outputs = triton_client.infer("blip2_model", inputs)
    return parse_blip_outputs(outputs)
```

### 2. Разделение на preprocessing → inference → postprocessing

**Важно:** Triton работает только с тензорами, поэтому:

```
[Preprocessing в Python] → [Triton Inference] → [Postprocessing в Python]
```

**Пример:**
```python
# Preprocessing (в Python)
image_array = pic2float(image)
inputs = processor(image_array, prompt, return_tensors="pt")

# Inference (в Triton)
outputs = triton_client.infer("blip2_model", inputs)

# Postprocessing (в Python)
caption = processor.decode(outputs[0], skip_special_tokens=True)
```

### 3. Батчинг становится автоматическим

**Сейчас:** Вы сами управляете батчами
```python
for bbox in bboxs:
    proc_image, mask = biref_process(image, bbox)  # По одному
```

**С Triton:** Triton автоматически собирает батчи из разных запросов
```python
# Можно отправлять запросы асинхронно, Triton соберет их в батч
tasks = [
    triton_client.async_infer("birefnet_model", prepare_input(image, bbox))
    for bbox in bboxs
]
results = await asyncio.gather(*tasks)
```

## Что нужно изменить в архитектуре пайплайна

### 1. Абстракция инференса моделей

**Создать интерфейс для инференса:**

```python
# inference/base.py
from abc import ABC, abstractmethod

class ModelInference(ABC):
    """Абстракция для инференса модели"""
    
    @abstractmethod
    def infer(self, inputs):
        """Выполняет инференс"""
        pass

# inference/direct.py - текущая реализация
class DirectInference(ModelInference):
    """Прямой вызов модели (текущий способ)"""
    def __init__(self, model):
        self.model = model
    
    def infer(self, inputs):
        with torch.no_grad():
            return self.model(**inputs)

# inference/triton.py - будущая реализация
class TritonInference(ModelInference):
    """Инференс через Triton"""
    def __init__(self, model_name, triton_client):
        self.model_name = model_name
        self.client = triton_client
    
    def infer(self, inputs):
        return self.client.infer(self.model_name, inputs)
```

### 2. Шаги пайплайна должны использовать абстракцию

```python
# pipeline/steps.py
class DescriptionStep(PipelineStep):
    def __init__(self, inference: ModelInference):
        self.inference = inference  # Может быть Direct или Triton
    
    def process(self, context: PipelineContext) -> PipelineContext:
        inputs = self._prepare_inputs(context.image)
        outputs = self.inference.infer(inputs)  # Работает с любой реализацией
        context.description = self._parse_outputs(outputs)
        return context
```

### 3. Конфигурация через dependency injection

```python
# config.py
import os

INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "direct")  # "direct" или "triton"
TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")

# pipeline/factory.py
def create_inference_backend(model_name: str):
    if INFERENCE_BACKEND == "triton":
        client = tritonclient.http.InferenceServerClient(url=TRITON_URL)
        return TritonInference(model_name, client)
    else:
        model = load_model_directly(model_name)
        return DirectInference(model)
```

## Преимущества такой архитектуры

1. **Плавный переход:** Можно переключаться между direct и Triton через переменную окружения
2. **Тестируемость:** Легко мокировать инференс для тестов
3. **Гибкость:** Можно использовать разные бэкенды для разных моделей
4. **Готовность к Triton:** Архитектура уже подготовлена

## Что нужно учесть при переходе на Triton

### 1. Формат данных

Triton работает с numpy arrays, а не PyTorch tensors:
```python
# Сейчас
inputs = processor(image, return_tensors="pt").to("cuda")

# С Triton
inputs_numpy = processor(image, return_tensors="np")  # numpy!
inputs = tritonclient.http.InferInput("input", inputs_numpy.shape, "FP32")
inputs.set_data_from_numpy(inputs_numpy)
```

### 2. Конфигурация моделей (model repository)

Каждая модель в Triton требует конфиг файл:
```python
# model_repository/blip2_model/config.pbtxt
name: "blip2_model"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {
    name: "pixel_values"
    data_type: TYPE_FP16
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "generated_ids"
    data_type: TYPE_INT64
    dims: [ 100 ]
  }
]
```

### 3. Асинхронность

Triton поддерживает асинхронные запросы:
```python
# Можно отправлять запросы параллельно
async def process_multiple_objects(bboxes):
    tasks = [
        triton_client.async_infer("birefnet", prepare_input(bbox))
        for bbox in bboxes
    ]
    return await asyncio.gather(*tasks)
```

### 4. Мониторинг и метрики

Triton предоставляет метрики:
- Latency (p50, p95, p99)
- Throughput
- GPU utilization
- Queue depth

## Рекомендуемая архитектура для перехода

```
┌─────────────────────────────────────────┐
│         Application Layer              │
│  (Flask/Gradio - ваш текущий код)      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      Pipeline Layer                      │
│  (PipelineStep с абстракцией инференса) │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌─────▼─────────┐
│ Direct      │  │ Triton        │
│ Inference   │  │ Inference      │
│ (сейчас)    │  │ (будущее)      │
└─────────────┘  └────────────────┘
```

## План миграции

### Этап 1: Подготовка (сейчас)
- ✅ Создать абстракцию `ModelInference`
- ✅ Рефакторить шаги пайплайна для использования абстракции
- ✅ Добавить конфигурацию для переключения бэкендов

### Этап 2: Экспорт моделей
- Экспортировать модели в формат для Triton (TorchScript/ONNX)
- Создать конфиги для каждой модели
- Настроить model repository

### Этап 3: Интеграция Triton
- Запустить Triton Server с моделями
- Реализовать `TritonInference`
- Переключить через конфиг

### Этап 4: Оптимизация
- Настроить батчинг в Triton
- Оптимизировать preprocessing/postprocessing
- Мониторинг и метрики

## Важные моменты для текущей архитектуры

1. **Не привязывайтесь к прямым вызовам моделей** - используйте абстракцию
2. **Разделяйте preprocessing/inference/postprocessing** - Triton работает только с inference
3. **Готовьтесь к асинхронности** - Triton эффективнее работает с async запросами
4. **Думайте о батчинге** - Triton автоматически батчит, но нужно правильно формировать запросы
5. **Учитывайте форматы данных** - Triton работает с numpy, не PyTorch tensors

## Пример готовой архитектуры

```python
# inference/__init__.py
from inference.base import ModelInference
from inference.direct import DirectInference
from inference.triton import TritonInference

def create_inference(model_name: str, backend: str = "direct"):
    if backend == "triton":
        return TritonInference(model_name)
    return DirectInference(model_name)

# pipeline/steps.py
class DescriptionStep(PipelineStep):
    def __init__(self):
        # Может быть Direct или Triton в зависимости от конфига
        self.inference = create_inference("blip2", INFERENCE_BACKEND)
    
    def process(self, context):
        inputs = self._prepare_inputs(context.image)
        outputs = self.inference.infer(inputs)  # Работает с любым бэкендом
        context.description = self._parse_outputs(outputs)
        return context
```

## Вывод

**Да, переход на Triton имеет значение для архитектуры!**

Но если правильно спроектировать пайплайн сейчас:
- ✅ Использовать абстракцию для инференса
- ✅ Разделять preprocessing/inference/postprocessing
- ✅ Готовиться к асинхронности
- ✅ Не привязываться к прямым вызовам моделей

То переход на Triton будет **плавным** и **безболезненным** - просто замените реализацию `ModelInference`!

