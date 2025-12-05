# Предложение по улучшению архитектуры пайплайна

## Текущая архитектура (проблемы)

```
HTTP Request → Flask → process_image() → [BLIP → GDINO → BiRefNet → Center → Decontaminate → Shadow] → Response
                ↓
            Синхронно, последовательно, монолитно
```

## Предлагаемая архитектура

### Вариант 1: Pipeline Pattern с шагами

```python
# pipeline/steps.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PipelineContext:
    """Контекст обработки, передается между шагами"""
    image: np.ndarray
    description: Optional[str] = None
    bboxes: Optional[List] = None
    masks: Optional[List] = None
    processed_objects: Optional[List] = None
    params: dict = None

class PipelineStep(ABC):
    """Базовый класс для шага пайплайна"""
    
    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """Обрабатывает контекст и возвращает обновленный"""
        pass
    
    @abstractmethod
    def can_skip(self, context: PipelineContext) -> bool:
        """Можно ли пропустить этот шаг"""
        pass

class DescriptionStep(PipelineStep):
    """BLIP-2: генерация описания объекта"""
    def process(self, context: PipelineContext) -> PipelineContext:
        if context.description is None:
            context.description = describe_center_object(context.image)
        return context
    
    def can_skip(self, context: PipelineContext) -> bool:
        return context.description is not None

class DetectionStep(PipelineStep):
    """GroundingDINO: детекция объектов"""
    def process(self, context: PipelineContext) -> PipelineContext:
        if context.bboxes is None:
            objects = detect_objects(context.image, context.description)
            context.bboxes = [bbox.tolist() for bbox in objects['boxes']]
        return context
    
    def can_skip(self, context: PipelineContext) -> bool:
        return context.bboxes is not None

class SegmentationStep(PipelineStep):
    """BiRefNet: сегментация объектов"""
    def process(self, context: PipelineContext) -> PipelineContext:
        if context.masks is None:
            context.masks = []
            for bbox in context.bboxes[:context.params.get('max_objects', 4)]:
                image, mask = biref_process(context.image, bbox)
                context.masks.append((image, mask))
        return context
    
    def can_skip(self, context: PipelineContext) -> bool:
        return context.masks is not None

# pipeline/pipeline.py
class ImageProcessingPipeline:
    """Пайплайн обработки изображений"""
    
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Выполняет все шаги пайплайна"""
        for step in self.steps:
            if not step.can_skip(context):
                try:
                    context = step.process(context)
                except Exception as e:
                    # Обработка ошибок на уровне шага
                    context.error = str(e)
                    if not context.params.get('continue_on_error', False):
                        raise
        return context
```

### Вариант 2: Асинхронный пайплайн с очередью

```python
# pipeline/async_pipeline.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Callable, List

class AsyncPipeline:
    """Асинхронный пайплайн с возможностью параллельной обработки"""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_objects_parallel(
        self, 
        objects: List[tuple], 
        process_fn: Callable
    ) -> List:
        """Параллельная обработка объектов"""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, process_fn, obj)
            for obj in objects
        ]
        return await asyncio.gather(*tasks)
    
    async def process_image(self, image, params):
        """Асинхронная обработка изображения"""
        context = PipelineContext(image=image, params=params)
        
        # Последовательные шаги (зависят друг от друга)
        context = await self.description_step(context)
        context = await self.detection_step(context)
        
        # Параллельная обработка объектов
        if context.bboxes:
            results = await self.process_objects_parallel(
                context.bboxes,
                lambda bbox: self.process_single_object(image, bbox, params)
            )
            context.processed_objects = results
        
        return context
```

### Вариант 3: Pipeline с кэшированием промежуточных результатов

```python
# pipeline/cached_pipeline.py
from functools import lru_cache
import hashlib

class CachedPipelineStep(PipelineStep):
    """Шаг пайплайна с кэшированием"""
    
    def __init__(self, cache_size: int = 128):
        self.cache = {}
        self.cache_size = cache_size
    
    def _get_cache_key(self, context: PipelineContext) -> str:
        """Генерирует ключ кэша на основе входных данных"""
        # Хэшируем только релевантные данные для этого шага
        data = self._get_relevant_data(context)
        return hashlib.md5(str(data).encode()).hexdigest()
    
    def process(self, context: PipelineContext) -> PipelineContext:
        cache_key = self._get_cache_key(context)
        
        if cache_key in self.cache:
            # Используем кэшированный результат
            cached_result = self.cache[cache_key]
            context = self._apply_cached_result(context, cached_result)
        else:
            # Выполняем обработку
            context = self._do_process(context)
            
            # Сохраняем в кэш
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))  # Удаляем старый
            self.cache[cache_key] = self._extract_result(context)
        
        return context
```

### Вариант 4: Pipeline с батчингом

```python
# pipeline/batch_pipeline.py
from typing import List
import torch

class BatchPipeline:
    """Пайплайн с поддержкой батчинга для GPU"""
    
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.pending_requests = []
    
    def add_request(self, image, params):
        """Добавляет запрос в очередь"""
        self.pending_requests.append((image, params))
        
        if len(self.pending_requests) >= self.batch_size:
            return self.process_batch()
        return None
    
    def process_batch(self) -> List:
        """Обрабатывает батч запросов"""
        batch_images = [req[0] for req in self.pending_requests]
        batch_params = [req[1] for req in self.pending_requests]
        
        # Батчинг для GPU
        # Например, для сегментации:
        batched_masks = self.biref_predictor.predict_batch(batch_images)
        
        results = []
        for i, (image, params) in enumerate(self.pending_requests):
            mask = batched_masks[i]
            result = self.process_single(image, mask, params)
            results.append(result)
        
        self.pending_requests.clear()
        return results
```

## Рекомендуемая архитектура (комбинированная)

```python
# pipeline/processing_pipeline.py
from typing import List, Optional
from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class ProcessingResult:
    """Результат обработки одного объекта"""
    image: np.ndarray
    mask: np.ndarray
    shadow_image: Optional[np.ndarray] = None
    error: Optional[str] = None

@dataclass
class PipelineContext:
    """Контекст обработки"""
    image: np.ndarray
    params: dict = field(default_factory=dict)
    
    # Промежуточные результаты (кэшируются)
    description: Optional[str] = None
    bboxes: Optional[List] = None
    objects: List[ProcessingResult] = field(default_factory=list)
    
    # Метаданные
    processing_time: dict = field(default_factory=dict)

class ImageProcessingPipeline:
    """
    Улучшенный пайплайн обработки изображений с:
    - Разделением на шаги
    - Кэшированием промежуточных результатов
    - Параллельной обработкой объектов
    - Обработкой ошибок
    """
    
    def __init__(
        self,
        enable_caching: bool = True,
        max_parallel_objects: int = 4,
        cache_ttl: int = 3600
    ):
        self.enable_caching = enable_caching
        self.max_parallel_objects = max_parallel_objects
        self.executor = ThreadPoolExecutor(max_workers=max_parallel_objects)
        self.cache = {}  # Можно использовать Redis для продакшена
    
    async def process(self, image: np.ndarray, params: dict) -> PipelineContext:
        """Главный метод обработки"""
        context = PipelineContext(image=image, params=params)
        
        # Шаг 1: Описание (можно кэшировать по хэшу изображения)
        context = await self._step_description(context)
        
        # Шаг 2: Детекция (зависит от описания)
        context = await self._step_detection(context)
        
        # Шаг 3: Параллельная обработка объектов
        if context.bboxes:
            context.objects = await self._step_process_objects_parallel(context)
        
        return context
    
    async def _step_description(self, context: PipelineContext) -> PipelineContext:
        """Шаг 1: Генерация описания"""
        cache_key = self._get_image_hash(context.image)
        
        if self.enable_caching and cache_key in self.cache:
            context.description = self.cache[cache_key].get('description')
        
        if context.description is None:
            context.description = await asyncio.to_thread(
                describe_center_object, context.image
            )
            if self.enable_caching:
                self.cache.setdefault(cache_key, {})['description'] = context.description
        
        return context
    
    async def _step_detection(self, context: PipelineContext) -> PipelineContext:
        """Шаг 2: Детекция объектов"""
        cache_key = self._get_image_hash(context.image)
        
        if self.enable_caching:
            cached = self.cache.get(cache_key, {})
            if 'bboxes' in cached:
                context.bboxes = cached['bboxes']
        
        if context.bboxes is None:
            objects = await asyncio.to_thread(
                detect_objects, context.image, context.description
            )
            context.bboxes = [bbox.tolist() for bbox in objects['boxes'][:context.params.get('max_objects', 4)]]
            
            if self.enable_caching:
                self.cache.setdefault(cache_key, {})['bboxes'] = context.bboxes
        
        return context
    
    async def _step_process_objects_parallel(
        self, 
        context: PipelineContext
    ) -> List[ProcessingResult]:
        """Шаг 3: Параллельная обработка объектов"""
        tasks = [
            self._process_single_object(context.image, bbox, context.params)
            for bbox in context.bboxes
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_object(
        self, 
        image: np.ndarray, 
        bbox: List, 
        params: dict
    ) -> ProcessingResult:
        """Обработка одного объекта"""
        try:
            # Сегментация
            proc_image, mask = await asyncio.to_thread(
                biref_process, image, bbox
            )
            
            # Центрирование
            proc_image, mask = await asyncio.to_thread(
                center, proc_image, mask
            )
            
            # Деконтаминация
            proc_image = await asyncio.to_thread(
                decontaminate, proc_image, mask
            )
            
            # Генерация тени
            shadow_image = await asyncio.to_thread(
                generate_shadow, proc_image, mask, params.get('rot')
            )
            
            return ProcessingResult(
                image=proc_image,
                mask=mask,
                shadow_image=shadow_image
            )
        except Exception as e:
            return ProcessingResult(
                image=proc_image if 'proc_image' in locals() else None,
                mask=mask if 'mask' in locals() else None,
                error=str(e)
            )
    
    def _get_image_hash(self, image: np.ndarray) -> str:
        """Генерирует хэш изображения для кэширования"""
        import hashlib
        return hashlib.md5(image.tobytes()).hexdigest()
```

## Использование в Flask

```python
# ML_server.py (улучшенная версия)
from flask import Flask, request
from pipeline.processing_pipeline import ImageProcessingPipeline
import asyncio

app = Flask(__name__)
pipeline = ImageProcessingPipeline(
    enable_caching=True,
    max_parallel_objects=4
)

@app.route('/process', methods=['POST'])
def process():
    image = Image.open(request.files['image'])
    params = request.form.to_dict()
    
    # Асинхронная обработка
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    context = loop.run_until_complete(
        pipeline.process(pic2float(image), params)
    )
    loop.close()
    
    # Формирование ответа
    processed_images = [
        pic2pil(obj.shadow_image) for obj in context.objects 
        if obj.error is None
    ]
    
    return prepare_response(processed_images, "Success", 200)
```

## Преимущества новой архитектуры

1. **Модульность**: Каждый шаг - отдельный класс, легко тестировать
2. **Переиспользование**: Можно использовать промежуточные результаты
3. **Параллелизм**: Объекты обрабатываются параллельно
4. **Кэширование**: Промежуточные результаты кэшируются
5. **Обработка ошибок**: Ошибка в одном объекте не ломает весь пайплайн
6. **Гибкость**: Легко добавить/убрать/изменить шаги
7. **Масштабируемость**: Можно добавить очередь задач, батчинг, распределенную обработку

