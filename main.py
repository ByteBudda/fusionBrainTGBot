# -*- coding: utf-8 -*-

# --- Импорты ---
import logging          # Для логирования событий и ошибок
import os               # Для работы с переменными окружения
import time             # Используется в синхронном коде (хотя стараемся избегать)
import requests         # Для HTTP-запросов к Fusion Brain API (будет выполняться в executor'е)
import base64           # Для декодирования изображений из base64
import json             # Для работы с JSON-данными в API запросах/ответах
from io import BytesIO  # Для работы с бинарными данными изображений в памяти
from pathlib import Path # Для удобной работы с путями к файлам (для .env)
from dotenv import load_dotenv # Для загрузки переменных окружения из файла .env
from PIL import Image   # Для анализа изображений и определения MIME-типа
import asyncio          # Основная библиотека для асинхронного программирования
import functools        # Для использования functools.partial в run_in_executor

# Библиотеки для Telegram
from telegram import Update, InputFile, constants
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO # Уровень логирования (INFO, WARNING, ERROR, CRITICAL)
)
# Уменьшаем "шум" от низкоуровневых HTTP библиотек
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.vendor.ptb_urllib3.urllib3").setLevel(logging.WARNING)


logger = logging.getLogger(__name__) # Получаем логгер для нашего модуля

# Библиотека для Google Gemini API
import google.generativeai as genai

# --- Конфигурация ---

# Загрузка переменных окружения из файла .env в текущей директории
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Основные токены и URL
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FUSION_BRAIN_URL = 'https://api-key.fusionbrain.ai/'
# Имя модели Gemini (Flash обычно быстрее и дешевле для Vision)
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash-latest')

# Загрузка НЕСКОЛЬКИХ пар ключей Fusion Brain
# Ищет переменные FUSION_BRAIN_API_KEY_1, FUSION_BRAIN_SECRET_KEY_1, ..._2 и т.д.
FUSION_BRAIN_KEY_PAIRS = []
i = 1
while True:
    api_key = os.getenv(f"FUSION_BRAIN_API_KEY_{i}")
    secret_key = os.getenv(f"FUSION_BRAIN_SECRET_KEY_{i}")
    if api_key and secret_key:
        FUSION_BRAIN_KEY_PAIRS.append({"api_key": api_key, "secret_key": secret_key, "id": i})
        logger.info(f"Загружена пара ключей Fusion Brain ID: {i}")
        i += 1
    else:
        break # Прекращаем поиск, если пара с таким номером не найдена

# Если нумерованные ключи не найдены, пытаемся загрузить ключи без номера
if not FUSION_BRAIN_KEY_PAIRS:
    logger.info("Нумерованные ключи Fusion Brain (_1, _2...) не найдены. Попытка загрузить ключи без номера...")
    # Попытка загрузить старый формат имен (API_KEY / SECRET_KEY)
    api_key_old = os.getenv("API_KEY")
    secret_key_old = os.getenv("SECRET_KEY")
    if api_key_old and secret_key_old:
         FUSION_BRAIN_KEY_PAIRS.append({"api_key": api_key_old, "secret_key": secret_key_old, "id": 0}) # Используем ID 0
         logger.info("Загружена пара ключей Fusion Brain (API_KEY/SECRET_KEY)")
    else:
        # Попытка загрузить стандартные имена (FUSION_BRAIN_API_KEY / FUSION_BRAIN_SECRET_KEY)
        api_key_std = os.getenv("FUSION_BRAIN_API_KEY")
        secret_key_std = os.getenv("FUSION_BRAIN_SECRET_KEY")
        if api_key_std and secret_key_std:
            FUSION_BRAIN_KEY_PAIRS.append({"api_key": api_key_std, "secret_key": secret_key_std, "id": 0})
            logger.info("Загружена пара ключей Fusion Brain (FUSION_BRAIN_API_KEY/...)")



# --- Класс для работы с API Fusion Brain (адаптированный для asyncio) ---
class FusionBrainAPI:
    """
    Класс для асинхронного взаимодействия с Fusion Brain API (генерация изображений).
    Использует requests через run_in_executor для неблокирующей работы.
    """
    def __init__(self, url: str, api_key: str, secret_key: str, key_id: int | str):
        self.URL = url
        self.key_id = key_id # Идентификатор для логирования, какой ключ используется
        if not api_key or not secret_key:
             raise ValueError(f"API Key и Secret Key для Fusion Brain (ID: {key_id}) должны быть установлены.")
        self.AUTH_HEADERS = {
            'X-Key': f'Key {api_key}',
            'X-Secret': f'Secret {secret_key}',
        }
        self.pipeline_id = None # ID модели (пайплайна) будет получен при инициализации

    async def _run_blocking(self, func, *args, **kwargs):
        """Вспомогательная функция для запуска блокирующих вызовов (как requests) в executor'е."""
        loop = asyncio.get_running_loop()
        # functools.partial нужен, чтобы правильно передать именованные аргументы (kwargs)
        partial_func = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, partial_func) # None означает использование стандартного ThreadPoolExecutor

    async def initialize_pipeline_id(self) -> bool:
        """Асинхронно получает и сохраняет ID пайплайна (модели) Fusion Brain."""
        try:
            logger.info(f"[FB Client ID: {self.key_id}] Запрос ID пайплайна...")
            response = await self._run_blocking(
                requests.get, # Блокирующая функция
                self.URL + 'key/api/v1/pipelines',
                headers=self.AUTH_HEADERS,
                timeout=25 # Увеличим таймаут
            )
            response.raise_for_status() # Проверка на HTTP ошибки 4xx/5xx
            data = response.json()
            # Ищем активную модель TEXT2IMAGE
            active_pipeline = next((p for p in data if p.get("status") == "ACTIVE" and p.get("type") == "TEXT2IMAGE"), None)

            if active_pipeline and 'id' in active_pipeline:
                self.pipeline_id = active_pipeline['id']
                logger.info(f"[FB Client ID: {self.key_id}] Получен ID пайплайна: {self.pipeline_id} (Name: {active_pipeline.get('name', 'N/A')})")
                return True
            else:
                logger.error(f"[FB Client ID: {self.key_id}] Не найден активный TEXT2IMAGE пайплайн в ответе: {data}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"[FB Client ID: {self.key_id}] Ошибка сети при получении ID пайплайна: {e}")
            return False
        except Exception as e:
            logger.error(f"[FB Client ID: {self.key_id}] Неожиданная ошибка при получении ID пайплайна: {e}", exc_info=True)
            return False

    async def generate(self, prompt: str, images: int = 1, width: int = 1024, height: int = 1024, style: str = "DEFAULT") -> str | None:
        """
        Асинхронно отправляет запрос на генерацию изображения.
        Возвращает UUID задачи или None в случае ошибки.
        """
        if not self.pipeline_id:
            logger.error(f"[FB Client ID: {self.key_id}] ID пайплайна не установлен. Генерация невозможна.")
            # Попытка инициализировать ID снова перед отказом
            if not await self.initialize_pipeline_id():
                logger.error(f"[FB Client ID: {self.key_id}] Повторная инициализация ID пайплайна не удалась.")
                return None
            # Если инициализация удалась, продолжаем
            logger.info(f"[FB Client ID: {self.key_id}] ID пайплайна успешно инициализирован во время запроса генерации.")


        params = {
            "type": "GENERATE",
            "numImages": images,
            "width": width,
            "height": height,
            "style": style,
            "generateParams": { "query": f"{prompt}" }
        }
        files = {
            'pipeline_id': (None, self.pipeline_id),
            'params': (None, json.dumps(params), 'application/json')
        }

        try:
            logger.info(f"[FB Client ID: {self.key_id}] Отправка запроса на генерацию. Промпт: '{prompt[:60]}...'")
            response = await self._run_blocking(
                requests.post, # Блокирующая функция
                self.URL + 'key/api/v1/pipeline/run',
                headers=self.AUTH_HEADERS,
                files=files,
                timeout=45 # Таймаут для запроса
            )
            response.raise_for_status()
            data = response.json()

            if 'uuid' in data:
                logger.info(f"[FB Client ID: {self.key_id}] Запрос на генерацию принят, UUID: {data['uuid']}")
                return data['uuid']
            elif 'errorDescription' in data:
                 logger.error(f"[FB Client ID: {self.key_id}] Ошибка API Fusion Brain при генерации: {data['errorDescription']}")
                 return None
            elif 'pipeline_status' in data:
                 logger.warning(f"[FB Client ID: {self.key_id}] Fusion Brain сервис недоступен: {data['pipeline_status']}")
                 return None
            else:
                logger.error(f"[FB Client ID: {self.key_id}] Неожиданный ответ от Fusion Brain при запуске генерации: {data}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"[FB Client ID: {self.key_id}] Таймаут при отправке запроса на генерацию в Fusion Brain.")
            return None
        except requests.exceptions.RequestException as e:
            # Логируем ошибку и тело ответа, если оно есть
            error_details = f"[FB Client ID: {self.key_id}] Ошибка сети/HTTP при запросе к Fusion Brain: {e}"
            if e.response is not None:
                 try:
                     error_details += f" | Status: {e.response.status_code} | Body: {e.response.text[:500]}"
                 except Exception: pass # Игнорируем ошибки чтения тела ответа
            logger.error(error_details)
            return None
        except Exception as e:
            logger.error(f"[FB Client ID: {self.key_id}] Неожиданная ошибка при запросе к Fusion Brain: {e}", exc_info=True)
            return None

    async def check_generation(self, request_id: str, attempts: int = 30, delay: int = 7) -> list[str] | str | None:
        """
        Асинхронно проверяет статус генерации изображения.
        Возвращает список строк base64, строку статуса ("censored", "error", "timeout") или None.
        Увеличен attempts и уменьшен delay для более частого опроса.
        """
        logger.info(f"[FB Client ID: {self.key_id}] Проверка статуса для UUID: {request_id} (Попыток: {attempts}, Задержка: {delay}с)")
        current_attempt = 0
        while current_attempt < attempts:
            current_attempt += 1
            try:
                response = await self._run_blocking(
                     requests.get, # Блокирующая функция
                     self.URL + 'key/api/v1/pipeline/status/' + request_id,
                     headers=self.AUTH_HEADERS,
                     timeout=25 # Таймаут для запроса статуса
                )
                response.raise_for_status()
                data = response.json()
                status = data.get('status')
                logger.debug(f"[FB Client ID: {self.key_id}] Статус для UUID {request_id} (Попытка {current_attempt}/{attempts}): {status}")

                if status == 'DONE':
                    logger.info(f"[FB Client ID: {self.key_id}] Генерация для UUID {request_id} УСПЕШНО завершена.")
                    result_data = data.get('result', {})
                    if result_data.get('censored', False):
                         logger.warning(f"[FB Client ID: {self.key_id}] Изображение {request_id} было ЗАЦЕНЗУРЕНО.")
                         return "censored"
                    files = result_data.get('files')
                    if files and isinstance(files, list):
                        return files # Успешный результат
                    else:
                        logger.error(f"[FB Client ID: {self.key_id}] Статус DONE, но результат некорректен ('files' отсутствуют или не список) для {request_id}: {data}")
                        return "error" # Ошибка в структуре ответа

                elif status == 'FAIL':
                    error_desc = data.get('errorDescription', 'Нет описания ошибки')
                    logger.error(f"[FB Client ID: {self.key_id}] Генерация для UUID {request_id} ПРОВАЛЕНА: {error_desc}")
                    return "error"

                elif status in ['INITIAL', 'PROCESSING']:
                    # Статус нормальный, ждем следующей проверки
                    pass # Продолжаем цикл while
                else:
                    # Неизвестный или неожиданный статус
                    logger.warning(f"[FB Client ID: {self.key_id}] Обнаружен неизвестный статус '{status}' для UUID {request_id}. Ответ: {data}")
                    # Решаем продолжать или считать ошибкой. Пока продолжаем.

            except requests.exceptions.Timeout:
                logger.warning(f"[FB Client ID: {self.key_id}] Таймаут при проверке статуса UUID {request_id} (Попытка {current_attempt}/{attempts}).")
                # Продолжаем попытки при таймауте
            except requests.exceptions.RequestException as e:
                logger.error(f"[FB Client ID: {self.key_id}] Ошибка сети/HTTP при проверке статуса UUID {request_id} (Попытка {current_attempt}/{attempts}): {e}")
                # При сетевой ошибке стоит немного подождать и попробовать снова
            except Exception as e:
                logger.error(f"[FB Client ID: {self.key_id}] Неожиданная ошибка при проверке статуса UUID {request_id} (Попытка {current_attempt}/{attempts}): {e}", exc_info=True)
                return "error" # При других ошибках лучше прервать цикл

            # Ждем перед следующей попыткой (только если еще остались попытки)
            if current_attempt < attempts:
                await asyncio.sleep(delay)

        # Если цикл завершился без результата
        logger.warning(f"[FB Client ID: {self.key_id}] Превышено максимальное количество ({attempts}) попыток проверки статуса для UUID {request_id}.")
        return "timeout"

# --- Функции для работы с Gemini API (адаптированные для asyncio) ---

async def _run_blocking_gemini(func, *args, **kwargs):
    """Вспомогательная функция для запуска блокирующих вызовов Gemini API в executor'е."""
    loop = asyncio.get_running_loop()
    partial_func = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, partial_func)

async def enhance_prompt_with_gemini(original_prompt: str, gemini_model) -> str | None:
    """
    Асинхронно улучшает текстовый промпт с помощью Gemini.
    Возвращает улучшенный промпт или None в случае ошибки/блокировки.
    """
    # Промпт для Gemini с инструкциями по улучшению
    instruction = (
        "Ты - ИИ-ассистент, улучшающий промпты для генерации изображений. "
        "Сделай промпт пользователя более детальным и ярким, добавив описание освещения, окружения, стиля (реализм/фотореализм, если не указано иное), настроения, композиции, деталей. "
        "Сохрани основную идею. Если промпт уже хорош, дополни немного или верни как есть. "
        "Ограничь длину ответа примерно 990 символами. " # Ограничение Fusion Brain + запас
        "Выдай ТОЛЬКО улучшенный промпт, без своих комментариев и извинений.\n\n"
        f"Промпт пользователя: \"{original_prompt}\"\n\n"
        "Улучшенный промпт:"
    )
    logger.info(f"Отправка промпта в Gemini для улучшения: '{original_prompt[:60]}...'")
    try:
        # Асинхронный запуск блокирующего вызова Gemini
        response = await _run_blocking_gemini(
            gemini_model.generate_content, # Блокирующая функция
            instruction,
            request_options={'timeout': 45} # Таймаут для запроса
        )

        # Проверка ответа Gemini
        if not response.parts:
             block_reason = response.prompt_feedback.block_reason
             if block_reason:
                  logger.warning(f"Запрос к Gemini (enhance) заблокирован по причине: {block_reason}")
             else:
                 logger.warning("Ответ Gemini (enhance) пуст без явной причины блокировки.")
             return None # Не удалось получить текст

        enhanced_prompt = response.text.strip()
        if not enhanced_prompt:
            logger.warning("Gemini (enhance) вернул пустой текст после обработки.")
            return None

        logger.info(f"Улучшенный промпт от Gemini получен: '{enhanced_prompt[:60]}...'")
        return enhanced_prompt

    except Exception as e:
        logger.error(f"Ошибка при обращении к Gemini API (enhance): {e}", exc_info=True)
        return None

async def describe_image_with_gemini(image_bytes: bytes, gemini_model) -> str | None:
    """
    Асинхронно анализирует изображение с помощью Gemini Vision.
    Возвращает текстовое описание, специальный маркер "vision_blocked" или None.
    """
    # Промпт для Gemini Vision с инструкциями и ограничением длины
    prompt = (
        "Опиши это изображение максимально подробно, чтобы можно было создать похожее с помощью другой нейросети (например, Kandinsky или Stable Diffusion). "
        "Сконцентрируйся на следующих аспектах:\n"
        "- Главный объект(ы): что это, как выглядит, поза, эмоции.\n"
        "- Фон/Окружение: где происходит действие, важные детали.\n"
        "- Стиль: фотореализм, иллюстрация, арт, 3D-рендер и т.д.\n"
        "- Освещение: тип, тени, блики.\n"
        "- Цветовая палитра: преобладающие цвета, контраст.\n"
        "- Композиция: ракурс, план.\n"
        "- Атмосфера/Настроение.\n"
        "Предоставь только само описание в виде связного текста (50-150 слов), без вступлений. "
        "Крайне важно: Итоговое описание должно быть НЕ БОЛЕЕ 990 символов. " # Ограничение Fusion Brain + запас
        "Избегай упоминания текста на картинке."
    )
    logger.info("Отправка изображения в Gemini Vision для анализа...")
    try:
        # Определение MIME-типа (синхронно, т.к. быстро)
        mime_type = 'image/jpeg' # Значение по умолчанию
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                img_format = img.format
                if img_format:
                    mime_type = f"image/{img_format.lower()}"
                    logger.info(f"Определен MIME-тип через Pillow: {mime_type}")
                else:
                    raise ValueError("Pillow не смог определить формат")
        except Exception as pil_e:
             logger.warning(f"Ошибка определения MIME через Pillow ({pil_e}), пробуем по сигнатуре...")
             # Проверка сигнатур (можно добавить больше форматов)
             if image_bytes.startswith(b'\xff\xd8\xff'): mime_type = 'image/jpeg'
             elif image_bytes.startswith(b'\x89PNG\r\n\x1a\n'): mime_type = 'image/png'
             elif image_bytes.startswith(b'GIF8'): mime_type = 'image/gif'
             elif image_bytes.startswith(b'RIFF') and image_bytes[8:12] == b'WEBP': mime_type = 'image/webp'
             logger.warning(f"Предполагаемый MIME-тип по сигнатуре: {mime_type}")

        image_part = {"mime_type": mime_type, "data": image_bytes}

        # Асинхронный запуск блокирующего вызова Gemini
        response = await _run_blocking_gemini(
            gemini_model.generate_content, # Блокирующая функция
            [prompt, image_part], # Передаем промпт и изображение
            request_options={'timeout': 60} # Увеличенный таймаут для Vision
        )

        # Обработка ответа Gemini Vision
        if not response.parts:
             block_reason = response.prompt_feedback.block_reason
             if block_reason:
                 logger.warning(f"Запрос к Gemini Vision заблокирован по причине: {block_reason}")
                 return "vision_blocked" # Явный блок промпта/картинки

             # Проверяем safety ratings, если нет явного блока
             for rating in response.prompt_feedback.safety_ratings:
                 # Считаем блок, если рейтинг HIGH или MEDIUM
                 if rating.probability.name in ('HIGH', 'MEDIUM'):
                     logger.warning(f"Gemini Vision заблокировал из-за safety ratings: {rating.category.name} - {rating.probability.name}")
                     return "vision_blocked" # Блок из-за содержимого

             logger.warning("Ответ Gemini Vision пуст без явной причины блокировки или опасного контента.")
             return None # Не удалось получить ответ

        description = response.text.strip()

        # Проверка на нерелевантный ответ
        if not description or len(description) < 20 or "не могу проанализировать" in description.lower() or "cannot fulfill" in description.lower():
            logger.warning(f"Gemini Vision вернул слишком короткое или нерелевантное описание: '{description[:60]}...'")
            return None

        logger.info(f"Описание от Gemini Vision получено: '{description[:60]}...'")
        return description

    except ImportError:
        logger.error("Библиотека Pillow не найдена. Установите: pip install Pillow")
        return None
    except Exception as e:
        logger.error(f"Ошибка при обращении к Gemini Vision API: {e}", exc_info=True)
        return None

# --- Фоновая задача для полного цикла генерации ---

async def run_generation_task(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    prompt: str,
    processing_message, # Сообщение "Генерирую..." для редактирования/удаления
    original_input_text: str | None = None # Исходный текст пользователя, если был
):
    """
    Выполняет весь процесс генерации асинхронно в фоновом режиме.
    Выбирает клиента Fusion Brain, генерирует, проверяет статус и отправляет результат.
    """
    # Получаем список доступных клиентов Fusion Brain и текущий индекс для round-robin
    fusion_clients: list[FusionBrainAPI] = context.bot_data.get('fusion_clients', [])
    client_index: int = context.bot_data.get('fusion_client_index', 0)

    if not fusion_clients:
        logger.error("Нет доступных клиентов Fusion Brain для запуска задачи генерации.")
        try: await processing_message.edit_text("Ошибка: Сервис генерации изображений сейчас недоступен.")
        except Exception: pass # Игнорируем ошибки, если сообщение уже удалено
        return

    # Выбираем клиента для этого запроса по стратегии Round-Robin
    client_index_to_use = client_index % len(fusion_clients)
    selected_client = fusion_clients[client_index_to_use]
    # Обновляем индекс в bot_data для следующего запроса (потокобезопасно для asyncio)
    context.bot_data['fusion_client_index'] = (client_index + 1) % len(fusion_clients)

    logger.info(f"Запуск задачи генерации для chat_id={chat_id} с использованием клиента FB ID: {selected_client.key_id}")

    try:
        # Обрезаем промпт до максимальной длины Fusion Brain (подстраховка)
        final_prompt_for_fusion = prompt[:1000]
        if len(prompt) > 1000:
            logger.warning(f"[FB Client ID: {selected_client.key_id}] Промпт был обрезан с {len(prompt)} до 1000 символов перед отправкой в Fusion Brain.")

        # 1. Асинхронный запуск генерации в Fusion Brain
        uuid = await selected_client.generate(final_prompt_for_fusion)
        if uuid is None:
            # Ошибка уже залогирована внутри generate()
            await processing_message.edit_text(f"😕 Не удалось запустить генерацию (ключ ID: {selected_client.key_id}). Попробуйте позже.")
            return

        # 2. Асинхронная проверка результата генерации
        result = await selected_client.check_generation(uuid)

        # 3. Обработка результата и отправка пользователю
        if result == "error":
            await processing_message.edit_text(f"❌ Ошибка во время генерации изображения (ключ ID: {selected_client.key_id}).")
        elif result == "timeout":
            await processing_message.edit_text(f"⏳ Время ожидания генерации истекло (ключ ID: {selected_client.key_id}).")
        elif result == "censored":
            await processing_message.edit_text(f"🔞 Сгенерированное изображение не прошло модерацию (ключ ID: {selected_client.key_id}).")
        elif isinstance(result, list) and len(result) > 0:
            # Успех! Получили список base64 строк (обычно одна)
            image_base64 = result[0]
            try:
                # Декодирование base64
                image_data = base64.b64decode(image_base64)
                if not image_data:
                    raise ValueError("Декодирование base64 дало пустой результат.")

                image_stream = BytesIO(image_data)
                image_stream.name = 'generated_image.png' # Даем имя файлу для Telegram

                # Отправляем статус "загрузка фото"
                await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.UPLOAD_PHOTO)

                # Формируем подпись к изображению
                caption_base = "Готово! ✨\n\n"
                if original_input_text: # Если генерация была по тексту
                    caption_base += f"Оригинал: «{original_input_text}»\n"
                    # Показываем полный промпт (обрезанный до лимита TG), если он не равен оригиналу
                    if prompt != original_input_text:
                        caption_base += f"Промпт: «{prompt[:900]}...»" # Ограничиваем длину для подписи
                    else:
                         caption_base += f"Промпт: «{prompt[:900]}...»"
                else: # Если генерация была по фото
                     caption_base += f"Сгенерировано по вашему изображению.\n\nПромпт (из Vision): «{prompt[:900]}...»"

                # Отправляем фото с подписью (обрезанной до лимита Telegram)
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=image_stream,
                    caption=caption_base[:1024] # Лимит подписи Telegram - 1024 символа
                )
                # Удаляем сообщение "Генерирую..."
                await processing_message.delete()

            except (base64.binascii.Error, ValueError) as decode_err:
                logger.error(f"Ошибка декодирования Base64 от Fusion Brain (UUID {uuid}, ключ {selected_client.key_id}): {decode_err}")
                await processing_message.edit_text("😕 Ошибка обработки полученного изображения (неверный формат base64).")
            except Exception as send_err:
                logger.error(f"Ошибка отправки фото (UUID {uuid}, ключ {selected_client.key_id}): {send_err}", exc_info=True)
                # Пытаемся сообщить об ошибке, если можем редактировать сообщение
                try: await processing_message.edit_text("😕 Произошла ошибка при отправке сгенерированного изображения.")
                except Exception: logger.warning("Не удалось отредактировать сообщение об ошибке отправки фото.")
        else:
            # Если check_generation вернул что-то неожиданное
            logger.error(f"Неожиданный результат от check_generation (UUID {uuid}, ключ {selected_client.key_id}): {result}")
            await processing_message.edit_text("😕 Внутренняя ошибка при получении результата генерации от Fusion Brain.")

    except Exception as task_err:
        # Ловим любые другие ошибки внутри фоновой задачи
        logger.exception(f"Критическая ошибка в задаче генерации run_generation_task (ключ {selected_client.key_id}, chat {chat_id}): {task_err}")
        try:
            # Пытаемся сообщить пользователю об общей ошибке
            await processing_message.edit_text("Произошла серьезная внутренняя ошибка во время генерации.")
        except Exception:
            # Если редактирование не удалось, просто логируем
            logger.error("Не удалось отредактировать сообщение о критической ошибке в задаче генерации.")


# --- Обработчики команд и сообщений Telegram ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    user = update.effective_user
    await update.message.reply_html(
        f"Привет, {user.mention_html()}!\n\nЯ бот для генерации изображений с использованием нейросетей Kandinsky и Gemini.\n\n"
        "➡️ Отправь мне <b>текстовое описание</b> того, что хочешь увидеть.\n"
        "➡️ Или отправь <b>картинку</b> (как фото), чтобы я сгенерировал похожую.",
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
     """Обработчик команды /help."""
     await update.message.reply_text(
         "ℹ️ *Как пользоваться ботом:*\n\n"
         "1️⃣ *Генерация по тексту:*\n"
         "   Просто отправь мне текстовое описание (промпт). Я постараюсь улучшить его с помощью Gemini и затем сгенерирую изображение с помощью Kandinsky.\n"
         "   _Пример:_ `рыжий кот в очках сидит за компьютером, стиль киберпанк`\n\n"
         "2️⃣ *Генерация по картинке:*\n"
         "   Отправь мне изображение как обычное фото (не как файл). Я проанализирую его с помощью Gemini Vision и создам новое изображение, похожее по стилю и содержанию.\n\n"
         "⏳ *Ожидание:*\n"
         "   Генерация может занять некоторое время (обычно до минуты), так как включает анализ, улучшение запроса и саму генерацию. Я сообщу, когда начну работу, и пришлю результат по готовности.",
         parse_mode='Markdown'
     )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обрабатывает текстовые сообщения: улучшает промпт через Gemini
    и запускает фоновую задачу генерации через Fusion Brain.
    """
    # Проверяем наличие сообщения и текста
    if not update.message or not update.message.text: return
    chat_id = update.effective_chat.id
    original_prompt = update.message.text
    user = update.effective_user
    logger.info(f"Получено текстовое сообщение от {user.id} ({user.username}): '{original_prompt[:60]}...'")

    # Базовые проверки промпта
    if original_prompt.isspace():
        await update.message.reply_text("Пожалуйста, отправь непустой текст.")
        return
    if len(original_prompt) > 900: # Оставляем запас для улучшения Gemini
        await update.message.reply_text("Твой запрос слишком длинный. Попробуй короче (макс. 900 символов).")
        return

    # Проверка доступности сервисов (модели Gemini и клиентов Fusion Brain)
    gemini_model = context.bot_data.get('gemini_model')
    fusion_clients = context.bot_data.get('fusion_clients')
    if not gemini_model:
        logger.error("Модель Gemini не инициализирована в контексте бота.")
        await update.message.reply_text("⚠️ Ошибка конфигурации: сервис улучшения запросов недоступен.")
        return
    if not fusion_clients:
        logger.error("Клиенты Fusion Brain не инициализированы в контексте бота.")
        await update.message.reply_text("⚠️ Ошибка конфигурации: сервис генерации изображений недоступен.")
        return

    # Сообщение о начале работы
    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
    processing_message = await update.message.reply_text("⏳ Понял тебя! Улучшаю запрос с помощью Gemini...")

    # 1. Асинхронное улучшение промпта через Gemini
    enhanced_prompt = await enhance_prompt_with_gemini(original_prompt, gemini_model)
    final_prompt = enhanced_prompt if enhanced_prompt else original_prompt # Используем оригинал, если улучшение не удалось

    # Обновляем статус для пользователя перед запуском долгой задачи
    status_text = "✨ Запрос принят! Начинаю генерацию в фоновом режиме..."
    if final_prompt == original_prompt and enhanced_prompt is None:
        logger.warning("Не удалось улучшить промпт через Gemini, используется оригинальный.")
        status_text = "⚠️ Не удалось улучшить запрос. Использую оригинальный.\n✨ Начинаю генерацию в фоновом режиме..."
    elif final_prompt != original_prompt:
        logger.info("Промпт успешно улучшен Gemini.")
        status_text = f"✅ Запрос улучшен!\n✨ Начинаю генерацию в фоновом режиме..."

    try:
        await processing_message.edit_text(status_text) # Не используем Markdown здесь для простоты
    except Exception as e:
        # Ошибка редактирования не критична, просто логируем
        logger.warning(f"Не удалось отредактировать статусное сообщение перед запуском задачи: {e}")

    # 2. Запуск фоновой задачи для генерации и отправки изображения
    # Используем asyncio.create_task, чтобы не ждать завершения генерации здесь
    asyncio.create_task(run_generation_task(
        context=context,
        chat_id=chat_id,
        prompt=final_prompt, # Передаем финальный (возможно, улучшенный) промпт
        processing_message=processing_message, # Передаем сообщение для редактирования/удаления
        original_input_text=original_prompt # Передаем исходный текст для подписи
    ))
    logger.info(f"Фоновая задача генерации по тексту для chat_id={chat_id} запущена.")


async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обрабатывает полученные фотографии: анализирует через Gemini Vision
    и запускает фоновую задачу генерации похожего изображения через Fusion Brain.
    """
    # Проверяем наличие сообщения и фото
    if not update.message or not update.message.photo: return
    chat_id = update.effective_chat.id
    user = update.effective_user
    logger.info(f"Получено фото от {user.id} ({user.username})")

    # Проверка доступности сервисов
    gemini_model = context.bot_data.get('gemini_model')
    fusion_clients = context.bot_data.get('fusion_clients')
    if not gemini_model:
        logger.error("Модель Gemini не инициализирована в контексте бота.")
        await update.message.reply_text("⚠️ Ошибка конфигурации: сервис анализа изображений недоступен.")
        return
    if not fusion_clients:
        logger.error("Клиенты Fusion Brain не инициализированы в контексте бота.")
        await update.message.reply_text("⚠️ Ошибка конфигурации: сервис генерации изображений недоступен.")
        return

    # Получаем файл фото (выбираем наибольшее доступное разрешение)
    photo_file = update.message.photo[-1]

    # Скачиваем фото в память
    image_bytes = None
    try:
        # Сообщение о начале работы
        await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
        processing_message = await update.message.reply_text("🖼️ Фото получено. Анализирую его с помощью Gemini Vision...")

        logger.debug(f"Начало загрузки файла фото {photo_file.file_id}...")
        tg_file = await photo_file.get_file()
        # Скачиваем фото как bytearray, затем преобразуем в bytes
        image_bytearray = await tg_file.download_as_bytearray()
        image_bytes = bytes(image_bytearray)
        if not image_bytes:
            raise ValueError("Скачанные байты изображения пусты.")
        logger.debug(f"Файл фото {photo_file.file_id} успешно загружен ({len(image_bytes)} байт).")

    except Exception as e:
        logger.error(f"Ошибка при загрузке фото {photo_file.file_id} от {user.id}: {e}", exc_info=True)
        await update.message.reply_text("Не удалось загрузить твое фото из Телеграм. Попробуй еще раз.")
        return

    # 1. Асинхронный анализ изображения с помощью Gemini Vision
    description = await describe_image_with_gemini(image_bytes, gemini_model)

    # Обработка результата анализа
    if description == "vision_blocked":
         await processing_message.edit_text("🔞 К сожалению, не могу обработать это изображение из-за ограничений безопасности. Попробуй другое фото.")
         return
    elif description is None:
        # Ошибка уже залогирована внутри describe_image_with_gemini
        await processing_message.edit_text("😕 Не удалось проанализировать изображение с помощью Gemini Vision. Возможно, фото слишком сложное или сервис временно недоступен.")
        return

    # Обновляем статус для пользователя перед запуском долгой задачи
    status_text = "✅ Изображение проанализировано! Начинаю генерацию похожего в фоновом режиме..."
    try:
        await processing_message.edit_text(status_text)
    except Exception as e:
        logger.warning(f"Не удалось отредактировать статусное сообщение (Vision) перед запуском задачи: {e}")


    # 2. Запуск фоновой задачи для генерации и отправки изображения
    asyncio.create_task(run_generation_task(
        context=context,
        chat_id=chat_id,
        prompt=description, # Используем описание от Gemini как промпт
        processing_message=processing_message,
        original_input_text=None # Передаем None, чтобы показать, что исходником было фото
    ))
    logger.info(f"Фоновая задача генерации по фото для chat_id={chat_id} запущена.")


# --- Функции инициализации и запуска бота ---

async def initialize_fusion_clients(key_pairs: list) -> list[FusionBrainAPI]:
    """
    Асинхронно инициализирует клиентов Fusion Brain для каждой пары ключей
    и параллельно получает для них ID пайплайнов.
    Возвращает список только успешно инициализированных клиентов.
    """
    clients = []
    tasks = []
    temp_clients = [] # Временный список для хранения экземпляров до проверки pipeline ID

    if not key_pairs:
         logger.critical("Список пар ключей Fusion Brain пуст!")
         return []

    logger.info(f"Найдено {len(key_pairs)} пар ключей Fusion Brain. Запуск инициализации клиентов...")

    # Создаем экземпляры клиентов и задачи для получения их pipeline ID
    for pair in key_pairs:
        try:
            client = FusionBrainAPI(FUSION_BRAIN_URL, pair['api_key'], pair['secret_key'], pair['id'])
            temp_clients.append(client)
            # Создаем задачу для асинхронного вызова initialize_pipeline_id()
            tasks.append(asyncio.create_task(client.initialize_pipeline_id(), name=f"init_pipeline_{pair['id']}"))
            logger.debug(f"Создан клиент и задача инициализации для FB ID: {pair['id']}")
        except ValueError as e:
            # Ошибка при создании экземпляра (например, пустые ключи)
            logger.error(f"Ошибка создания экземпляра клиента Fusion Brain (ID: {pair['id']}): {e}")
        except Exception as create_e:
             logger.error(f"Неожиданная ошибка при создании клиента Fusion Brain (ID: {pair['id']}): {create_e}", exc_info=True)


    if not tasks:
         logger.error("Не создано ни одной задачи инициализации клиентов Fusion Brain.")
         return []

    # Ожидаем завершения всех задач инициализации pipeline ID
    logger.info(f"Ожидание завершения {len(tasks)} задач инициализации pipeline ID...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Задачи инициализации pipeline ID завершены.")

    # Собираем только успешно инициализированных клиентов
    successful_clients = 0
    for client, result in zip(temp_clients, results):
        if isinstance(result, Exception):
            # Ошибка произошла внутри задачи initialize_pipeline_id()
            logger.error(f"Ошибка при выполнении инициализации pipeline ID для клиента FB ID {client.key_id}: {result}", exc_info=isinstance(result, Exception))
        elif result is True:
            # Инициализация прошла успешно
            clients.append(client)
            successful_clients += 1
            logger.info(f"Клиент Fusion Brain (ID: {client.key_id}) успешно инициализирован (Pipeline ID: {client.pipeline_id}).")
        else: # Если initialize_pipeline_id вернул False
             logger.warning(f"Не удалось получить pipeline ID для клиента Fusion Brain (ID: {client.key_id}). Клиент не будет использоваться.")

    if not clients:
        logger.critical("Не удалось инициализировать ни одного рабочего клиента Fusion Brain! Генерация изображений будет невозможна.")
    else:
         logger.info(f"Итого успешно инициализировано: {successful_clients} из {len(temp_clients)} клиентов Fusion Brain.")

    return clients


async def post_init(application: Application):
    """Функция, выполняемая после инициализации приложения."""
    bot_info = await application.bot.get_me()
    logger.info(f"Бот {bot_info.username} (ID: {bot_info.id}) успешно инициализирован.")
    # Здесь можно добавить отправку сообщения админу о запуске, если нужно

async def on_shutdown(application: Application):
     """Функция, выполняемая при остановке бота."""
     logger.info("Начало процедуры остановки бота...")
     # Здесь можно добавить логику для сохранения состояния, если необходимо
     await asyncio.sleep(1) # Небольшая пауза для завершения текущих операций
     logger.info("Бот успешно остановлен.")


# --- Основная функция запуска бота ---
async def main() -> None:
    """Асинхронная основная функция для настройки и запуска бота."""
    global application # Делаем application глобальной, чтобы использовать в обработчике сигналов

    # 1. Проверка наличия всех необходимых токенов/ключей
    # ... (код проверки токенов без изменений) ...
    if not TELEGRAM_BOT_TOKEN: logger.critical("!!! Переменная TELEGRAM_BOT_TOKEN не установлена !!!"); return
    if not GEMINI_API_KEY: logger.critical("!!! Переменная GEMINI_API_KEY не установлена !!!"); return
    if not FUSION_BRAIN_KEY_PAIRS: logger.critical("!!! Не найдены API ключи Fusion Brain (используйте FUSION_BRAIN_API_KEY_n/SECRET_KEY_n) !!!"); return
    logger.info("Все необходимые API ключи и токены найдены в конфигурации.")


    # 2. Настройка клиента Gemini API
    # ... (код настройки Gemini без изменений) ...
    try:
        safety_settings = {} # Можно настроить безопасность Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME, safety_settings=safety_settings)
        logger.info(f"Клиент Gemini API успешно настроен. Используется модель: {GEMINI_MODEL_NAME}")
    except Exception as e:
        logger.critical(f"Не удалось настроить Gemini API или инициализировать модель {GEMINI_MODEL_NAME}: {e}", exc_info=True)
        return

    # 3. Асинхронная инициализация клиентов Fusion Brain
    # ... (код инициализации fusion_clients без изменений) ...
    fusion_clients = await initialize_fusion_clients(FUSION_BRAIN_KEY_PAIRS)
    if not fusion_clients:
        logger.error("Запуск бота невозможен без рабочих клиентов Fusion Brain.")
        return

    # 4. Создание и настройка приложения Telegram Bot
    # ... (код создания application без изменений, но убираем .post_shutdown, т.к. будем делать вручную) ...
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)      # Функция после инициализации
        # .post_shutdown(on_shutdown) # Убираем, будем вызывать stop/shutdown вручную
        .build()
    )


    # 5. Сохранение инициализированных клиентов и других данных в контекст бота
    # ... (код сохранения в bot_data без изменений) ...
    application.bot_data['gemini_model'] = gemini_model
    application.bot_data['fusion_clients'] = fusion_clients
    application.bot_data['fusion_client_index'] = 0


    # 6. Регистрация обработчиков команд и сообщений
    # ... (код регистрации обработчиков без изменений) ...
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))


    # 7. ИЗМЕНЕННЫЙ ЗАПУСК БОТА
    logger.info("Запуск компонентов бота...")
    try:
        # Инициализация приложения (выполняет post_init)
        await application.initialize()
        logger.info("Приложение инициализировано.")

        # Запуск самого updater'а для получения обновлений
        await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("Updater запущен в режиме polling.")

        # Запуск обработчиков и других фоновых задач приложения
        await application.start()
        logger.info("Application запущено, бот готов к работе.")

        # --- Ожидание сигнала остановки ---
        # Теперь главный цикл asyncio будет работать, пока его не прервут извне (Ctrl+C)
        # или пока не будет вызван stop() для application/updater
        # Можно добавить более сложную логику ожидания, но для простоты оставим так.
        # Цикл событий будет жить благодаря запущенному updater'у.
        while True:
            await asyncio.sleep(3600) # Просто спим, чтобы не завершать main()

    except (KeyboardInterrupt, SystemExit):
        logger.info("Получен сигнал KeyboardInterrupt или SystemExit.")
    except Exception as e:
        logger.critical(f"Критическая ошибка в основном цикле работы бота: {e}", exc_info=True)
    finally:
        # --- КОРРЕКТНОЕ ЗАВЕРШЕНИЕ РАБОТЫ ---
        logger.info("Начало процедуры штатной остановки бота...")
        if application.updater and application.updater.running:
            await application.updater.stop()
            logger.info("Updater остановлен.")
        if application.running:
            await application.stop()
            logger.info("Application остановлено.")
        # Завершаем работу всех компонентов, очищаем ресурсы
        await application.shutdown()
        logger.info("Приложение Telegram завершило работу (shutdown).")
        await on_shutdown(application) # Вызываем нашу функцию on_shutdown вручную

# --- Точка входа в программу ---
if __name__ == '__main__':
    # Выводим информацию о необходимых зависимостях
    # ... (print statements без изменений) ...
    print("--- Telegram Image Generation Bot ---")
    print("Убедитесь, что установлены зависимости:")
    print("pip install python-telegram-bot google-generativeai requests Pillow python-dotenv")
    print("---")
    print("Настройка конфигурации:")
    print("Создайте файл .env ...") # Сокращено для краткости
    print("---")
    print("Запуск бота (для остановки нажмите Ctrl+C)...")

    # Запуск асинхронной функции main через asyncio.run()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # asyncio.run() должен корректно обработать KeyboardInterrupt внутри main()
        # Этот блок скорее для случаев, если Ctrl+C нажат до старта основного цикла
        print("\nОбнаружен KeyboardInterrupt на верхнем уровне. Завершение...")
    except Exception as e:
        print(f"\n!!! Фатальная ошибка при запуске: {e}")
        logging.critical(f"Фатальная ошибка: {e}", exc_info=True)

    print("--- Бот окончательно завершил работу ---")

# Определяем application как глобальную переменную для доступа в on_shutdown
application: Application | None = None

# Определяем функцию on_shutdown, если она нужна
async def on_shutdown(app: Application):
     """Функция, выполняемая при остановке бота."""
     logger.info("Выполнение пользовательских действий при остановке...")
     # Здесь можно добавить логику, например, уведомление админа
     await asyncio.sleep(0.5) # Небольшая пауза
     logger.info("Пользовательские действия при остановке завершены.")