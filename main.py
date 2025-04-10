import logging
import os
import time
import requests
import base64
import json
from io import BytesIO
from telegram import Update, InputFile, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# --- Конфигурация ---
# !!! ВАЖНО: Замените на свои значения или используйте переменные окружения !!!
# Никогда не храните токены и ключи прямо в коде в реальных проектах!
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_TELEGRAM_BOT_TOKEN') # <--- Замени или используй env
FUSION_BRAIN_API_KEY = os.getenv('FUSION_BRAIN_API_KEY', 'YOUR_FUSION_BRAIN_API_KEY') # <--- Замени или используй env
FUSION_BRAIN_SECRET_KEY = os.getenv('FUSION_BRAIN_SECRET_KEY', 'YOUR_SECRET_KEY') # <--- Замени или используй env
FUSION_BRAIN_URL = 'https://api-key.fusionbrain.ai/'

# Включаем логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Класс для работы с API Fusion Brain (исправленный) ---
class FusionBrainAPI:
    def __init__(self, url, api_key, secret_key):
        self.URL = url
        if not api_key or not secret_key or api_key == 'YOUR_FUSION_BRAIN_API_KEY':
             raise ValueError("API Key и Secret Key должны быть установлены и отличаться от плейсхолдеров.")
        self.AUTH_HEADERS = {
            'X-Key': f'Key {api_key}',
            'X-Secret': f'Secret {secret_key}',
        }

    def get_pipeline_id(self):
        """Получает ID доступного пайплайна (Kandinsky)."""
        try:
            logger.info("Запрос ID пайплайна...")
            response = requests.get(self.URL + 'key/api/v1/pipelines', headers=self.AUTH_HEADERS, timeout=20)
            response.raise_for_status()  # Проверка на HTTP ошибки (4xx, 5xx)
            data = response.json()
            # Предполагаем, что нужная модель первая в списке
            if data and isinstance(data, list) and len(data) > 0 and 'id' in data[0]:
                pipeline_id = data[0]['id']
                logger.info(f"Получен ID пайплайна: {pipeline_id}")
                return pipeline_id
            else:
                logger.error(f"Неожиданный ответ при получении ID пайплайна: {data}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка сети при получении ID пайплайна: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при получении ID пайплайна: {e}")
            return None

    def generate(self, prompt, pipeline_id, images=1, width=1024, height=1024, style="DEFAULT"):
        """Отправляет запрос на генерацию изображения."""
        params = {
            "type": "GENERATE",
            "numImages": images,
            "width": width,
            "height": height,
            "style": style, # Стиль как отдельный параметр
            "generateParams": {
                "query": f"{prompt}"
            }
            # Сюда можно добавить "negativePromptDecoder": "...", если нужно
        }

        # Используем multipart/form-data для отправки ID пайплайна и параметров
        files = {
            'pipeline_id': (None, pipeline_id), # ID пайплайна передается так
            'params': (None, json.dumps(params), 'application/json')
        }

        try:
            logger.info(f"Отправка запроса на генерацию для пайплайна {pipeline_id}...")
            # Используем эндпоинт /pipeline/run
            response = requests.post(self.URL + 'key/api/v1/pipeline/run', headers=self.AUTH_HEADERS, files=files, timeout=30)
            response.raise_for_status() # Проверка на HTTP ошибки

            data = response.json()
            if 'uuid' in data:
                logger.info(f"Запрос на генерацию принят, UUID: {data['uuid']}")
                return data['uuid']
            elif 'errorDescription' in data:
                 logger.error(f"Ошибка от API при генерации: {data['errorDescription']}")
                 return None
            elif 'pipeline_status' in data:
                 logger.warning(f"Сервис недоступен: {data['pipeline_status']}")
                 return None # Возвращаем None, чтобы показать, что задача не принята
            else:
                logger.error(f"Неожиданный ответ при запуске генерации: {data}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка сети при отправке запроса на генерацию: {e}")
            # Попытка прочитать тело ответа при ошибке
            try: logger.error(f"Тело ответа: {e.response.text}")
            except: pass
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при отправке запроса на генерацию: {e}")
            return None

    def check_generation(self, request_id, attempts=20, delay=10):
        """Проверяет статус генерации и возвращает результат."""
        logger.info(f"Проверка статуса для UUID: {request_id}")
        while attempts > 0:
            try:
                # Используем эндпоинт /pipeline/status/
                response = requests.get(self.URL + 'key/api/v1/pipeline/status/' + request_id, headers=self.AUTH_HEADERS, timeout=20)
                response.raise_for_status() # Проверка на HTTP ошибки
                data = response.json()
                status = data.get('status')
                logger.debug(f"Статус для UUID {request_id}: {status}")

                if status == 'DONE':
                    logger.info(f"Генерация для UUID {request_id} завершена.")
                    # Проверяем цензуру
                    if data.get('result', {}).get('censored', False):
                         logger.warning(f"Изображение для UUID {request_id} зацензурено.")
                         return "censored"
                    # Возвращаем список файлов base64
                    return data.get('result', {}).get('files') # Используем 'files'

                elif status == 'FAIL':
                    error_desc = data.get('errorDescription', 'Нет описания ошибки')
                    logger.error(f"Генерация для UUID {request_id} провалена: {error_desc}")
                    return "error"

                elif status in ['INITIAL', 'PROCESSING']:
                    logger.info(f"Генерация для UUID {request_id} еще в процессе ({status}). Осталось попыток: {attempts-1}")
                    # Статус в норме, продолжаем ждать
                else:
                    logger.warning(f"Неизвестный статус '{status}' для UUID {request_id}. Ответ: {data}")
                    # Можно считать ошибкой или продолжить

            except requests.exceptions.RequestException as e:
                logger.error(f"Ошибка сети при проверке статуса {request_id}: {e}")
                # При сетевой ошибке стоит немного подождать и попробовать снова
            except Exception as e:
                logger.error(f"Неожиданная ошибка при проверке статуса {request_id}: {e}")
                return "error" # При других ошибках лучше прервать

            attempts -= 1
            time.sleep(delay) # Ожидание перед следующей попыткой

        logger.warning(f"Превышено количество попыток проверки статуса для UUID {request_id}.")
        return "timeout" # Возвращаем маркер таймаута

# --- Обработчики команд Telegram ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    user = update.effective_user
    await update.message.reply_html(
        f"Привет, {user.mention_html()}!\n\nЯ бот для генерации изображений с помощью Kandinsky.\n"
        "Просто отправь мне текстовое описание того, что хочешь увидеть.",
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
     """Обработчик команды /help."""
     await update.message.reply_text(
         "Отправь мне текст (промпт), и я постараюсь сгенерировать по нему изображение.\n"
         "Пример: 'рыжий кот в скафандре летит к марсу'\n\n"
         "Генерация может занять некоторое время (до минуты). Пожалуйста, будь терпелив."
     )

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений для генерации изображений."""
    chat_id = update.effective_chat.id
    prompt = update.message.text

    # Проверка на пустой промпт или слишком длинный (из документации)
    if not prompt:
        await update.message.reply_text("Пожалуйста, отправь непустое описание.")
        return
    if len(prompt) > 1000:
        await update.message.reply_text("Запрос слишком длинный (макс. 1000 символов).")
        return

    # Получаем API клиент и ID пайплайна из контекста бота
    api_client: FusionBrainAPI = context.bot_data['api_client']
    pipeline_id = context.bot_data.get('pipeline_id')

    # Если ID пайплайна нет, пытаемся получить его снова
    if not pipeline_id:
        logger.warning("ID пайплайна отсутствует в контексте, пытаюсь получить снова...")
        pipeline_id = api_client.get_pipeline_id()
        if pipeline_id:
            context.bot_data['pipeline_id'] = pipeline_id
            logger.info("ID пайплайна успешно получен и сохранен.")
        else:
            logger.error("Не удалось получить ID пайплайна при обработке запроса.")
            await update.message.reply_text("Не могу связаться с сервисом генерации изображений. Попробуйте позже.")
            return

    # Отправляем сообщение о начале работы и "печатаем..."
    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
    processing_message = await update.message.reply_text("✨ Начинаю колдовать... Генерация может занять до минуты.")

    try:
        # 1. Запускаем генерацию
        uuid = api_client.generate(prompt, pipeline_id)

        if uuid is None:
            # Ошибка уже на этапе запуска генерации
            await processing_message.edit_text("😕 Не удалось запустить генерацию. Возможно, сервис перегружен или проверьте промпт. Попробуйте позже.")
            return

        # 2. Проверяем результат (эта функция уже содержит цикл ожидания)
        result = api_client.check_generation(uuid)

        # 3. Обрабатываем результат
        if result == "error":
            await processing_message.edit_text("❌ Произошла ошибка во время генерации. Попробуйте изменить запрос или повторить позже.")
        elif result == "timeout":
            await processing_message.edit_text("⏳ Время ожидания генерации истекло. Сервер мог быть перегружен. Попробуйте позже.")
        elif result == "censored":
            await processing_message.edit_text("🔞 Упс! Сгенерированное изображение не прошло модерацию. Попробуйте другой, более нейтральный запрос.")
        elif isinstance(result, list) and len(result) > 0:
            # Успех! Получили список base64 строк
            image_base64 = result[0] # Берем первое изображение
            try:
                # Декодируем base64
                image_data = base64.b64decode(image_base64)
                image_stream = BytesIO(image_data)
                image_stream.name = 'generated_image.png' # Даем имя файлу

                # Отправляем фото
                await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.UPLOAD_PHOTO)
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=image_stream,
                    caption=f"Готово! ✨\n\nПромпт: «{prompt}»" # Используем кавычки-елочки для красоты
                )
                # Удаляем сообщение "Начинаю колдовать..."
                await processing_message.delete()

            except (base64.binascii.Error, ValueError) as e:
                logger.error(f"Ошибка декодирования Base64 для UUID {uuid}: {e}")
                await processing_message.edit_text("😕 Не удалось обработать полученное изображение (ошибка формата).")
            except Exception as e:
                logger.error(f"Ошибка отправки фото для UUID {uuid}: {e}")
                await processing_message.edit_text("😕 Произошла ошибка при отправке изображения.")
        else:
            # Неожиданный результат от check_generation (не список, не ошибка, не таймаут)
            logger.error(f"Неожиданный результат от check_generation для UUID {uuid}: {result}")
            await processing_message.edit_text("😕 Произошла внутренняя ошибка при получении результата.")

    except Exception as e:
        # Общая обработка непредвиденных ошибок в процессе
        logger.exception(f"Непредвиденная ошибка в обработчике generate_image: {e}")
        try:
            await processing_message.edit_text(f"Произошла серьезная ошибка: {e}")
        except: # Если даже редактирование не удалось
             await update.message.reply_text(f"Произошла серьезная ошибка: {e}")


# --- Основная функция запуска ---
def main() -> None:
    """Запуск бота."""
    # Проверка токенов перед стартом
    if TELEGRAM_BOT_TOKEN == 'YOUR_TELEGRAM_BOT_TOKEN':
        logger.critical("!!! НЕОБХОДИМО УСТАНОВИТЬ TELEGRAM_BOT_TOKEN !!!")
        return
    if FUSION_BRAIN_API_KEY == 'YOUR_FUSION_BRAIN_API_KEY' or FUSION_BRAIN_SECRET_KEY == 'YOUR_SECRET_KEY':
        logger.critical("!!! НЕОБХОДИМО УСТАНОВИТЬ FUSION_BRAIN_API_KEY и FUSION_BRAIN_SECRET_KEY !!!")
        return

    # Создаем экземпляр API клиента
    try:
        api_client = FusionBrainAPI(FUSION_BRAIN_URL, FUSION_BRAIN_API_KEY, FUSION_BRAIN_SECRET_KEY)
    except ValueError as e:
        logger.critical(f"Ошибка инициализации API: {e}")
        return

    # Пытаемся получить ID пайплайна при старте
    pipeline_id = api_client.get_pipeline_id()
    if not pipeline_id:
        logger.warning("Не удалось получить ID пайплайна при запуске бота. Бот будет пытаться получить его при первом запросе.")
        # Можно завершить работу, если это критично:
        # logger.critical("Не удалось получить ID пайплайна. Запуск бота отменен.")
        # return

    # Создаем приложение
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Сохраняем клиент API и ID пайплайна в контекст бота
    application.bot_data['api_client'] = api_client
    application.bot_data['pipeline_id'] = pipeline_id # Может быть None, если при старте не получили

    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_image))

    # Запускаем бота
    logger.info("Бот запускается...")
    application.run_polling()
    logger.info("Бот остановлен.")


if __name__ == '__main__':
    main()
