# -*- coding: utf-8 -*-

# --- Импорты ---
import logging
import os
import time
import requests
import base64
import json
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import asyncio
import functools

# Библиотеки для Telegram
from telegram import Update, InputFile, constants, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler,
    ConversationHandler,
)

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.vendor.ptb_urllib3.urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Библиотека для Google Gemini API
import google.generativeai as genai

# --- Конфигурация ---

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FUSION_BRAIN_URL = 'https://api-key.fusionbrain.ai/'
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash-latest')

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
        break

if not FUSION_BRAIN_KEY_PAIRS:
    logger.info("Нумерованные ключи Fusion Brain (_1, _2...) не найдены. Попытка загрузить ключи без номера...")
    api_key_old = os.getenv("API_KEY")
    secret_key_old = os.getenv("SECRET_KEY")
    if api_key_old and secret_key_old:
         FUSION_BRAIN_KEY_PAIRS.append({"api_key": api_key_old, "secret_key": secret_key_old, "id": 0})
         logger.info("Загружена пара ключей Fusion Brain (API_KEY/SECRET_KEY)")
    else:
        api_key_std = os.getenv("FUSION_BRAIN_API_KEY")
        secret_key_std = os.getenv("FUSION_BRAIN_SECRET_KEY")
        if api_key_std and secret_key_std:
            FUSION_BRAIN_KEY_PAIRS.append({"api_key": api_key_std, "secret_key": secret_key_std, "id": 0})
            logger.info("Загружена пара ключей Fusion Brain (FUSION_BRAIN_API_KEY/...)")

# --- Состояния для ConversationHandler ---
STYLE_SETTINGS = 2
ASPECT_RATIO_SETTINGS = 3
SETTINGS_MAIN_MENU = 4
PROMPT_ENHANCEMENT_SETTINGS = 5

# --- Доступные стили и соотношения сторон ---
AVAILABLE_STYLES = ["DEFAULT", "CINEMATIC", "PHOTOREALISTIC", "ANIME", "DIGITAL_ART", "COMIC_BOOK", "PENCIL_DRAWING", "PASTEL_ART"]
AVAILABLE_ASPECT_RATIOS = {"1:1": (1024, 1024), "16:9": (1024, 576), "9:16": (576, 1024), "3:2": (1024, 683), "2:3": (683, 1024)}
PROMPT_ENHANCEMENT_OPTIONS = {"Включено": True, "Выключено": False}

# --- Класс для работы с API Fusion Brain (адаптированный для asyncio) ---
class FusionBrainAPI:
    def __init__(self, url: str, api_key: str, secret_key: str, key_id: int | str):
        self.URL = url
        self.key_id = key_id
        if not api_key or not secret_key:
             raise ValueError(f"API Key и Secret Key для Fusion Brain (ID: {key_id}) должны быть установлены.")
        self.AUTH_HEADERS = {
            'X-Key': f'Key {api_key}',
            'X-Secret': f'Secret {secret_key}',
        }
        self.pipeline_id = None

    async def _run_blocking(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        partial_func = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, partial_func)

    async def initialize_pipeline_id(self) -> bool:
        try:
            logger.info(f"[FB Client ID: {self.key_id}] Запрос ID пайплайна...")
            response = await self._run_blocking(
                requests.get,
                self.URL + 'key/api/v1/pipelines',
                headers=self.AUTH_HEADERS,
                timeout=25
            )
            response.raise_for_status()
            data = response.json()
            active_pipeline = next((
                p for p in data
                if p.get("status") == "ACTIVE"
                and p.get("type") == "TEXT2IMAGE"
                and p.get("version") == 3.1
            ), None)

            if active_pipeline and 'id' in active_pipeline:
                self.pipeline_id = active_pipeline['id']
                logger.info(f"[FB Client ID: {self.key_id}] Получен ID пайплайна: {self.pipeline_id} (Name: {active_pipeline.get('name', 'N/A')})")
                return True
            else:
                logger.error(f"[FB Client ID: {self.key_id}] Не найден активный TEXT2IMAGE пайплайн версии 3.1 в ответе: {data}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"[FB Client ID: {self.key_id}] Ошибка сети при получении ID пайплайна: {e}")
            return False
        except Exception as e:
            logger.error(f"[FB Client ID: {self.key_id}] Неожиданная ошибка при получении ID пайплайна: {e}", exc_info=True)
            return False

    async def generate(self, prompt: str, images: int = 1, width: int = 1024, height: int = 1024, style: str = "DEFAULT") -> str | None:
        if not self.pipeline_id:
            logger.error(f"[FB Client ID: {self.key_id}] ID пайплайна не установлен. Генерация невозможна.")
            if not await self.initialize_pipeline_id():
                logger.error(f"[FB Client ID: {self.key_id}] Повторная инициализация ID пайплайна не удалась.")
                return None
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
                requests.post,
                self.URL + 'key/api/v1/pipeline/run',
                headers=self.AUTH_HEADERS,
                files=files,
                timeout=45
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
            error_details = f"[FB Client ID: {self.key_id}] Ошибка сети/HTTP при запросе к Fusion Brain: {e}"
            if e.response is not None:
                 try:
                     error_details += f" | Status: {e.response.status_code} | Body: {e.response.text[:500]}"
                 except Exception: pass
            logger.error(error_details)
            return None
        except Exception as e:
            logger.error(f"[FB Client ID: {self.key_id}] Неожиданная ошибка при запросе к Fusion Brain: {e}", exc_info=True)
            return None

    async def check_generation(self, request_id: str, attempts: int = 30, delay: int = 7) -> list[str] | str | None:
        logger.info(f"[FB Client ID: {self.key_id}] Проверка статуса для UUID: {request_id} (Попыток: {attempts}, Задержка: {delay}с)")
        current_attempt = 0
        while current_attempt < attempts:
            current_attempt += 1
            try:
                response = await self._run_blocking(
                     requests.get,
                     self.URL + 'key/api/v1/pipeline/status/' + request_id,
                     headers=self.AUTH_HEADERS,
                     timeout=25
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
                        return files
                    else:
                        logger.error(f"[FB Client ID: {self.key_id}] Статус DONE, но результат некорректен ('files' отсутствуют или не список) для {request_id}: {data}")
                        return "error"

                elif status == 'FAIL':
                    error_desc = data.get('errorDescription', 'Нет описания ошибки')
                    logger.error(f"[FB Client ID: {self.key_id}] Генерация для UUID {request_id} ПРОВАЛЕНА: {error_desc}")
                    return "error"

                elif status in ['INITIAL', 'PROCESSING']:
                    pass
                else:
                    logger.warning(f"[FB Client ID: {self.key_id}] Обнаружен неизвестный статус '{status}' для UUID {request_id}. Ответ: {data}")

            except requests.exceptions.Timeout:
                logger.warning(f"[FB Client ID: {self.key_id}] Таймаут при проверке статуса UUID {request_id} (Попытка {current_attempt}/{attempts}).")
            except requests.exceptions.RequestException as e:
                logger.error(f"[FB Client ID: {self.key_id}] Ошибка сети/HTTP при проверке статуса UUID {request_id} (Попытка {current_attempt}/{attempts}): {e}")
            except Exception as e:
                logger.error(f"[FB Client ID: {self.key_id}] Неожиданная ошибка при проверке статуса UUID {request_id} (Попытка {current_attempt}/{attempts}): {e}", exc_info=True)
                return "error"

            if current_attempt < attempts:
                await asyncio.sleep(delay)

        logger.warning(f"[FB Client ID: {self.key_id}] Превышено максимальное количество ({attempts}) попыток проверки статуса для UUID {request_id}.")
        return "timeout"

# --- Функции для работы с Gemini API (адаптированные для asyncio) ---

async def _run_blocking_gemini(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    partial_func = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, partial_func)

async def enhance_prompt_with_gemini(original_prompt: str, gemini_model, style: str = "DEFAULT", aspect_ratio: tuple[int, int] = (1024, 1024)) -> str | None:
    aspect_ratio_str = f"{aspect_ratio[0]}x{aspect_ratio[1]}"
    instruction = (
        "Ты - ИИ-ассистент, улучшающий промпты для генерации изображений. "
        "Сделай промпт пользователя более детальным и ярким, добавив описание освещения, окружения, стиля (реализм/фотореализм, если не указано иное), настроения, композиции, деталей. "
        "Сохрани основную идею. Если промпт уже хорош, дополни немного или верни как есть. "
        "Ограничь длину ответа примерно 990 символами. "
        "Выдай ТОЛЬКО улучшенный промпт, без своих комментариев и извинений.\n\n"
        f"Промпт пользователя: \"{original_prompt}\"\n\n"
    )
    if style and style != "DEFAULT":
        instruction += f"Сгенерируй промпт в стиле: {style}.\n\n"
    instruction += f"Учти желаемое соотношение сторон изображения: {aspect_ratio_str}.\n\n"
    instruction += "Улучшенный промпт:"

    logger.info(f"Отправка промпта в Gemini для улучшения: '{original_prompt[:60]}...', стиль: {style}, соотношение сторон: {aspect_ratio_str}")
    try:
        response = await _run_blocking_gemini(
            gemini_model.generate_content,
            instruction,
            request_options={'timeout': 45}
        )

        if not response.parts:
             block_reason = response.prompt_feedback.block_reason
             if block_reason:
                  logger.warning(f"Запрос к Gemini (enhance) заблокирован по причине: {block_reason}")
             else:
                 logger.warning("Ответ Gemini (enhance) пуст без явной причины блокировки.")
             return None

        enhanced_prompt = response.text.strip()
        if not enhanced_prompt:
            logger.warning("Gemini (enhance) вернул пустой текст после обработки.")
            return None

        logger.info(f"Улучшенный промпт от Gemini получен: '{enhanced_prompt[:60]}...'")
        return enhanced_prompt

    except Exception as e:
        logger.error(f"Ошибка при обращении к Gemini API (enhance): {e}", exc_info=True)
        return None

async def describe_image_with_gemini(image_bytes: bytes, gemini_model, caption: str = None) -> str | None:
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
        "Крайне важно: Итоговое описание должно быть НЕ БОЛЕЕ 990 символов. "
        "Избегай упоминания текста на картинке."
    )
    if caption:
        prompt += f"\n\nТакже учти следующее описание к изображению: '{caption}'."

    logger.info("Отправка изображения в Gemini Vision для анализа...")
    try:
        mime_type = 'image/jpeg'
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
             if image_bytes.startswith(b'\xff\xd8\xff'): mime_type = 'image/jpeg'
             elif image_bytes.startswith(b'\x89PNG\r\n\x1a\n'): mime_type = 'image/png'
             elif image_bytes.startswith(b'GIF8'): mime_type = 'image/gif'
             elif image_bytes.startswith(b'RIFF') and image_bytes[8:12] == b'WEBP': mime_type = 'image/webp'
             logger.warning(f"Предполагаемый MIME-тип по сигнатуре: {mime_type}")

        image_part = {"mime_type": mime_type, "data": image_bytes}

        response = await _run_blocking_gemini(
            gemini_model.generate_content,
            [prompt, image_part],
            request_options={'timeout': 60}
        )

        if not response.parts:
             block_reason = response.prompt_feedback.block_reason
             if block_reason:
                 logger.warning(f"Запрос к Gemini Vision заблокирован по причине: {block_reason}")
                 return "vision_blocked"

             for rating in response.prompt_feedback.safety_ratings:
                 if rating.probability.name in ('HIGH', 'MEDIUM'):
                     logger.warning(f"Gemini Vision заблокировал из-за safety ratings: {rating.category.name} - {rating.probability.name}")
                     return "vision_blocked"

             logger.warning("Ответ Gemini Vision пуст без явной причины блокировки или опасного контента.")
             return None

        description = response.text.strip()

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
    processing_message,
    style: str = "DEFAULT",
    aspect_ratio: tuple[int, int] = (1024, 1024),
):
    fusion_clients: list[FusionBrainAPI] = context.bot_data.get('fusion_clients', [])
    client_index: int = context.bot_data.get('fusion_client_index', 0)

    if not fusion_clients:
        logger.error("Нет доступных клиентов Fusion Brain для запуска задачи генерации.")
        try: await processing_message.edit_text("Ошибка: Сервис генерации изображений сейчас недоступен.")
        except Exception: pass
        return

    client_index_to_use = client_index % len(fusion_clients)
    selected_client = fusion_clients[client_index_to_use]
    context.bot_data['fusion_client_index'] = (client_index + 1) % len(fusion_clients)

    logger.info(f"Запуск задачи генерации для chat_id={chat_id} с использованием клиента FB ID: {selected_client.key_id}, стиль: {style}, соотношение сторон: {aspect_ratio}")

    try:
        final_prompt_for_fusion = prompt[:1000]
        if len(prompt) > 1000:
            logger.warning(f"[FB Client ID: {selected_client.key_id}] Промпт был обрезан с {len(prompt)} до 1000 символов перед отправкой в Fusion Brain.")

        uuid = await selected_client.generate(final_prompt_for_fusion, width=aspect_ratio[0], height=aspect_ratio[1], style=style)
        if uuid is None:
            await processing_message.edit_text(f"😕 Не удалось запустить генерацию (ключ ID: {selected_client.key_id}). Попробуйте позже.")
            return

        result = await selected_client.check_generation(uuid)

        if result == "error":
            await processing_message.edit_text(f"❌ Ошибка во время генерации изображения (ключ ID: {selected_client.key_id}).")
        elif result == "timeout":
            await processing_message.edit_text(f"⏳ Время ожидания генерации истекло (ключ ID: {selected_client.key_id}).")
        elif result == "censored":
            await processing_message.edit_text(f"🔞 Сгенерированное изображение не прошло модерацию (ключ ID: {selected_client.key_id}).")
        elif isinstance(result, list) and len(result) > 0:
            image_base64 = result[0]
            try:
                image_data = base64.b64decode(image_base64)
                if not image_data:
                    raise ValueError("Декодирование base64 дало пустой результат.")

                image_stream = BytesIO(image_data)
                image_stream.name = 'generated_image.png'

                await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.UPLOAD_PHOTO)

                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=image_stream,
                )
                await processing_message.delete()

            except (base64.binascii.Error, ValueError) as decode_err:
                logger.error(f"Ошибка декодирования Base64 от Fusion Brain (UUID {uuid}, ключ {selected_client.key_id}): {decode_err}")
                await processing_message.edit_text("😕 Ошибка обработки полученного изображения (неверный формат base64).")
            except Exception as send_err:
                logger.error(f"Ошибка отправки фото (UUID {uuid}, ключ {selected_client.key_id}): {send_err}", exc_info=True)
                try: await processing_message.edit_text("😕 Произошла ошибка при отправке сгенерированного изображения.")
                except Exception: logger.warning("Не удалось отредактировать сообщение об ошибке отправки фото.")
        else:
            logger.error(f"Неожиданный результат от check_generation (UUID {uuid}, ключ {selected_client.key_id}): {result}")
            await processing_message.edit_text("😕 Внутренняя ошибка при получении результата генерации от Fusion Brain.")

    except Exception as task_err:
        logger.exception(f"Критическая ошибка в задаче генерации run_generation_task (ключ {selected_client.key_id}, chat {chat_id}): {task_err}")
        try:
            await processing_message.edit_text("Произошла серьезная внутренняя ошибка во время генерации.")
        except Exception:
            logger.error("Не удалось отредактировать сообщение об ошибке отправки фото.")

# --- Обработчики команд и сообщений Telegram ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_html(
        f"Привет, {user.mention_html()}!\n\nЯ бот для генерации изображений с использованием нейросетей Fusion Brain и Gemini.\n\n"
        "➡️ Отправь мне <b>текстовое описание</b> того, что хочешь увидеть.\n"
        "➡️ Или отправь <b>картинку</b> (как фото), чтобы я сгенерировал похожую.\n\n"
        "Используй команду /settings для настройки стиля, соотношения сторон и улучшения промпта.",
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
     await update.message.reply_text(
         "ℹ️ *Как пользоваться ботом:*\n\n"
         "1️⃣ *Генерация по тексту:*\n"
         "   Просто отправь мне текстовое описание (промпт). Я могу улучшить его с помощью Gemini перед генерацией.\n"
         "   _Пример:_ `рыжий кот в очках сидит за компьютером, стиль киберпанк`\n\n"
         "2️⃣ *Генерация по картинке:*\n"
         "   Отправь мне изображение как обычное фото (не как файл). Я проанализирую его с помощью Gemini Vision и создам новое изображение, похожее по стилю и содержанию. Ты также можешь добавить описание к фото.\n\n"
         "⚙️ *Настройки:*\n"
         "   Используй команду /settings, чтобы выбрать стиль изображения, соотношение сторон и включить/выключить улучшение промпта.\n\n"
         "⏳ *Ожидание:*\n"
         "   Генерация может занять некоторое время (обычно до минуты). Я сообщу, когда начну работу, и пришлю результат по готовности.",
         parse_mode='Markdown'
     )

async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Стиль", callback_data='set_style')],
        [InlineKeyboardButton("Соотношение сторон", callback_data='set_aspect')],
        [InlineKeyboardButton("Улучшение промпта", callback_data='set_prompt_enhancement')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("⚙️ Настройки генерации:", reply_markup=reply_markup)
    return SETTINGS_MAIN_MENU

async def set_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    keyboard = [[InlineKeyboardButton(style, callback_data=f'style_{style}')] for style in AVAILABLE_STYLES]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text("🎨 Выбери стиль изображения:", reply_markup=reply_markup)
    return STYLE_SETTINGS

async def set_aspect(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    keyboard = [[InlineKeyboardButton(ratio, callback_data=f'aspect_{ratio}')] for ratio in AVAILABLE_ASPECT_RATIOS]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text("📐 Выбери соотношение сторон:", reply_markup=reply_markup)
    return ASPECT_RATIO_SETTINGS

async def set_prompt_enhancement(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    keyboard = [[InlineKeyboardButton(option, callback_data=f'enhance_{value}')] for option, value in PROMPT_ENHANCEMENT_OPTIONS.items()]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text("✨ Включить или выключить улучшение промпта с помощью Gemini?", reply_markup=reply_markup)
    return PROMPT_ENHANCEMENT_SETTINGS

async def handle_style_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    style = query.data.split('_')[1]
    context.user_data['style'] = style
    await query.edit_message_text(f"✅ Стиль изображения установлен на: {style}")
    return ConversationHandler.END

async def handle_aspect_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    ratio = query.data.split('_')[1]
    context.user_data['aspect_ratio'] = AVAILABLE_ASPECT_RATIOS[ratio]
    await query.edit_message_text(f"✅ Соотношение сторон установлено на: {ratio} ({AVAILABLE_ASPECT_RATIOS[ratio][0]}x{AVAILABLE_ASPECT_RATIOS[ratio][1]})")
    return ConversationHandler.END

async def handle_prompt_enhancement_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    value = query.data.split('_')[1]
    context.user_data['prompt_enhancement'] = value == 'True'
    status = "включено" if context.user_data['prompt_enhancement'] else "выключено"
    await query.edit_message_text(f"⚙️ Улучшение промпта установлено на: {status}")
    return ConversationHandler.END

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    chat_id = update.effective_chat.id
    original_prompt = update.message.text
    user = update.effective_user
    logger.info(f"Получено текстовое сообщение от {user.id} ({user.username}): '{original_prompt[:60]}...'")

    if original_prompt.isspace():
        await update.message.reply_text("Пожалуйста, отправь непустой текст.")
        return
    if len(original_prompt) > 900:
        await update.message.reply_text("Твой запрос слишком длинный. Попробуй короче (макс. 900 символов).")
        return

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

    user_prompt_enhancement = context.user_data.get('prompt_enhancement', True)
    user_style = context.user_data.get('style', "DEFAULT")
    user_aspect_ratio = context.user_data.get('aspect_ratio', (1024, 1024))

    if user_prompt_enhancement:
        await update.message.reply_text("⏳ Улучшаю запрос с помощью Gemini...")
        enhanced_prompt = await enhance_prompt_with_gemini(original_prompt, gemini_model, style=user_style, aspect_ratio=user_aspect_ratio)

        if enhanced_prompt:
            final_prompt = enhanced_prompt
            await update.message.reply_text(f"✨ Улучшенный запрос:\n\n«{final_prompt[:900]}...»")
        else:
            final_prompt = original_prompt
            await update.message.reply_text(f"⚠️ Не удалось улучшить запрос. Будет использован оригинальный запрос:\n\n«{final_prompt[:900]}...»")
    else:
        final_prompt = original_prompt
        await update.message.reply_text("⚙️ Улучшение промпта выключено. Начинаю генерацию...")

    processing_message = await update.message.reply_text("⏳ Генерирую...")
    asyncio.create_task(run_generation_task(
        context=context,
        chat_id=chat_id,
        prompt=final_prompt,
        processing_message=processing_message,
        style=user_style,
        aspect_ratio=user_aspect_ratio
    ))
    logger.info(f"Запущена генерация для chat_id={chat_id}, улучшение: {user_prompt_enhancement}, стиль: {user_style}, соотношение сторон: {user_aspect_ratio}")
    return ConversationHandler.END

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo: return
    chat_id = update.effective_chat.id
    user = update.effective_user
    logger.info(f"Получено фото от {user.id} ({user.username})")

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

    photo_file = update.message.photo[-1]
    caption = update.message.caption

    image_bytes = None
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
        processing_message = await update.message.reply_text("🖼️ Фото получено. Анализирую его...")

        logger.debug(f"Начало загрузки файла фото {photo_file.file_id}...")
        tg_file = await photo_file.get_file()
        image_bytearray = await tg_file.download_as_bytearray()
        image_bytes = bytes(image_bytearray)
        if not image_bytes:
            raise ValueError("Скачанные байты изображения пусты.")
        logger.debug(f"Файл фото {photo_file.file_id} успешно загружен ({len(image_bytes)} байт).")

    except Exception as e:
        logger.error(f"Ошибка при загрузке фото {photo_file.file_id} от {user.id}: {e}", exc_info=True)
        await update.message.reply_text("Не удалось загрузить твое фото из Телеграм. Попробуй еще раз.")
        return

    description = await describe_image_with_gemini(image_bytes, gemini_model, caption)

    if description == "vision_blocked":
         await processing_message.edit_text("🔞 К сожалению, не могу обработать это изображение из-за ограничений безопасности. Попробуй другое фото.")
         return
    elif description is None:
        await processing_message.edit_text("😕 Не удалось проанализировать изображение. Возможно, фото слишком сложное или сервис временно недоступен.")
        return

    status_text = "✅ Изображение проанализировано! Начинаю генерацию похожего..."
    try:
        await processing_message.edit_text(status_text)
    except Exception as e:
        logger.warning(f"Не удалось отредактировать статусное сообщение (Vision) перед запуском задачи: {e}")

    user_style = context.user_data.get('style', "DEFAULT")
    user_aspect_ratio = context.user_data.get('aspect_ratio', (1024, 1024))
    asyncio.create_task(run_generation_task(
        context=context,
        chat_id=chat_id,
        prompt=description,
        processing_message=processing_message,
        style=user_style,
        aspect_ratio=user_aspect_ratio
    ))
    logger.info(f"Фоновая задача генерации по фото для chat_id={chat_id} запущена.")

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    logger.info(f"User {user.first_name} canceled the conversation.")
    await update.message.reply_text(
        "Действие отменено.", reply_markup=None
    )
    return ConversationHandler.END

# --- Функции инициализации и запуска бота ---

async def initialize_fusion_clients(key_pairs: list) -> list[FusionBrainAPI]:
    clients = []
    tasks = []
    temp_clients = []

    if not key_pairs:
         logger.critical("Список пар ключей Fusion Brain пуст!")
         return []

    logger.info(f"Найдено {len(key_pairs)} пар ключей Fusion Brain. Запуск инициализации клиентов...")

    for pair in key_pairs:
        try:
            client = FusionBrainAPI(FUSION_BRAIN_URL, pair['api_key'], pair['secret_key'], pair['id'])
            temp_clients.append(client)
            tasks.append(asyncio.create_task(client.initialize_pipeline_id(), name=f"init_pipeline_{pair['id']}"))
            logger.debug(f"Создан клиент и задача инициализации для FB ID: {pair['id']}")
        except ValueError as e:
            logger.error(f"Ошибка создания экземпляра клиента Fusion Brain (ID: {pair['id']}): {e}")
        except Exception as create_e:
             logger.error(f"Неожиданная ошибка при создании клиента Fusion Brain (ID: {pair['id']}): {create_e}", exc_info=True)


    if not tasks:
         logger.error("Не создано ни одной задачи инициализации клиентов Fusion Brain.")
         return []

    logger.info(f"Ожидание завершения {len(tasks)} задач инициализации pipeline ID...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Задачи инициализации pipeline ID завершены.")

    successful_clients = 0
    for client, result in zip(temp_clients, results):
        if isinstance(result, Exception):
            logger.error(f"Ошибка при выполнении инициализации pipeline ID для клиента FB ID {client.key_id}: {result}", exc_info=isinstance(result, Exception))
        elif result is True:
            clients.append(client)
            successful_clients += 1
            logger.info(f"Клиент Fusion Brain (ID: {client.key_id}) успешно инициализирован (Pipeline ID: {client.pipeline_id}).")
        else:
             logger.warning(f"Не удалось получить pipeline ID для клиента Fusion Brain (ID: {client.key_id}). Клиент не будет использоваться.")

    if not clients:
        logger.critical("Не удалось инициализировать ни одного рабочего клиента Fusion Brain! Генерация изображений будет невозможна.")
    else:
         logger.info(f"Итого успешно инициализировано: {successful_clients} из {len(temp_clients)} клиентов Fusion Brain.")

    return clients


async def post_init(application: Application):
    bot_info = await application.bot.get_me()
    logger.info(f"Бот {bot_info.username} (ID: {bot_info.id}) успешно инициализирован.")

async def on_shutdown(application: Application):
     logger.info("Начало процедуры остановки бота...")
     await asyncio.sleep(1)
     logger.info("Бот успешно остановлен.")


# --- Основная функция запуска бота ---
async def main() -> None:
    global application

    if not TELEGRAM_BOT_TOKEN: logger.critical("!!! Переменная TELEGRAM_BOT_TOKEN не установлена !!!"); return
    if not GEMINI_API_KEY: logger.critical("!!! Переменная GEMINI_API_KEY не установлена !!!"); return
    if not FUSION_BRAIN_KEY_PAIRS: logger.critical("!!! Не найдены API ключи Fusion Brain (используйте FUSION_BRAIN_API_KEY_n/SECRET_KEY_n) !!!"); return
    logger.info("Все необходимые API ключи и токены найдены в конфигурации.")

    try:
        safety_settings = {}
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME, safety_settings=safety_settings)
        logger.info(f"Клиент Gemini API успешно настроен. Используется модель: {GEMINI_MODEL_NAME}")
    except Exception as e:
        logger.critical(f"Не удалось настроить Gemini API или инициализировать модель {GEMINI_MODEL_NAME}: {e}", exc_info=True)
        return

    fusion_clients = await initialize_fusion_clients(FUSION_BRAIN_KEY_PAIRS)
    if not fusion_clients:
        logger.error("Запуск бота невозможен без рабочих клиентов Fusion Brain.")
        return

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    application.bot_data['gemini_model'] = gemini_model
    application.bot_data['fusion_clients'] = fusion_clients
    application.bot_data['fusion_client_index'] = 0

    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message)],
        states={},
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    settings_handler = ConversationHandler(
        entry_points=[CommandHandler('settings', settings)],
        states={
            SETTINGS_MAIN_MENU: [
                CallbackQueryHandler(set_style, pattern='^set_style$'),
                CallbackQueryHandler(set_aspect, pattern='^set_aspect$'),
                CallbackQueryHandler(set_prompt_enhancement, pattern='^set_prompt_enhancement$'),
            ],
            STYLE_SETTINGS: [CallbackQueryHandler(handle_style_choice, pattern='^style_')],
            ASPECT_RATIO_SETTINGS: [CallbackQueryHandler(handle_aspect_choice, pattern='^aspect_')],
            PROMPT_ENHANCEMENT_SETTINGS: [CallbackQueryHandler(handle_prompt_enhancement_choice, pattern='^enhance_')],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(settings_handler)
    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))

    logger.info("Запуск компонентов бота...")
    try:
        await application.initialize()
        logger.info("Приложение инициализировано.")

        await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("Updater запущен в режиме polling.")

        await application.start()
        logger.info("Application запущено, бот готов к работе.")

        while True:
            await asyncio.sleep(3600)

    except (KeyboardInterrupt, SystemExit):
        logger.info("Получен сигнал KeyboardInterrupt или SystemExit.")
    except Exception as e:
        logger.critical(f"Критическая ошибка в основном цикле работы бота: {e}", exc_info=True)
    finally:
        logger.info("Начало процедуры штатной остановки бота...")
        if application.updater and application.updater.running:
            await application.updater.stop()
            logger.info("Updater остановлен.")
        if application.running:
            await application.stop()
            logger.info("Application остановлено.")
        await application.shutdown()
        logger.info("Приложение Telegram завершило работу (shutdown).")
        await on_shutdown(application)

if __name__ == '__main__':
    print("--- Telegram Image Generation Bot ---")
    print("Убедитесь, что установлены зависимости:")
    print("pip install python-telegram-bot google-generativeai requests Pillow python-dotenv")
    print("---")
    print("Настройка конфигурации:")
    print("Создайте файл .env ...")
    print("---")
    print("Запуск бота (для остановки нажмите Ctrl+C)...")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nОбнаружен KeyboardInterrupt на верхнем уровне. Завершение...")
    except Exception as e:
        print(f"\n!!! Фатальная ошибка при запуске: {e}")
        logging.critical(f"Фатальная ошибка: {e}", exc_info=True)

    print("--- Бот окончательно завершил работу ---")

application: Application | None = None

async def on_shutdown(app: Application):
     logger.info("Выполнение пользовательских действий при остановке...")
     await asyncio.sleep(0.5)
     logger.info("Пользовательские действия при остановке завершены.")
