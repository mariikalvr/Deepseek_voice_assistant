import os
import logging
import tempfile
from io import BytesIO

from aiogram import Bot, Dispatcher, types
from aiogram.types import Message, ContentType
from aiogram.utils.executor import start_webhook
from dotenv import load_dotenv
import openai
import requests
from pydub import AudioSegment

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
WEBHOOK_HOST = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME', 'localhost')}"
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

# Инициализация бота
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(bot)

# Настройка DeepSeek API (через OpenAI совместимый интерфейс)
openai.api_key = DEEPSEEK_API_KEY
openai.api_base = "https://api.deepseek.com/v1"

async def on_startup(dp):
    """При запуске бота устанавливаем webhook"""
    await bot.set_webhook(WEBHOOK_URL)
    logger.info(f"Webhook установлен на {WEBHOOK_URL}")

async def on_shutdown(dp):
    """При остановке удаляем webhook"""
    await bot.delete_webhook()
    logger.info("Webhook удален")

def transcribe_voice(file_path: str) -> str:
    """Расшифровка голосового сообщения через DeepSeek Whisper"""
    try:
        with open(file_path, "rb") as audio_file:
            response = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file
            )
        return response.text
    except Exception as e:
        logger.error(f"Ошибка расшифровки: {e}")
        return None

def analyze_with_deepseek(text: str) -> str:
    """Анализ текста через DeepSeek API"""
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Ты полезный ассистент, который анализирует текст и дает содержательные ответы."},
                {"role": "user", "content": text}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Ошибка DeepSeek API: {e}")
        return "Извините, не удалось обработать запрос."

@dp.message_handler(commands=['start'])
async def start_command(message: Message):
    """Обработчик команды /start"""
    welcome_text = (
        "👋 Привет! Я бот, который умеет:\n"
        "✅ Расшифровывать голосовые сообщения\n"
        "✅ Анализировать текст через нейросеть DeepSeek\n\n"
        "Просто отправь мне голосовое сообщение или текст!"
    )
    await message.answer(welcome_text)

@dp.message_handler(content_types=ContentType.VOICE)
async def handle_voice(message: Message):
    """Обработчик голосовых сообщений"""
    # Отправляем статус "печатает"
    await bot.send_chat_action(message.chat.id, "typing")
    
    # Отправляем временное сообщение
    processing_msg = await message.answer("🎙️ Расшифровываю голосовое сообщение...")
    
    try:
        # Скачиваем голосовое сообщение
        file = await bot.get_file(message.voice.file_id)
        file_bytes = await bot.download_file(file.file_path)
        
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_ogg:
            tmp_ogg.write(file_bytes.getvalue())
            ogg_path = tmp_ogg.name
        
        # Конвертируем из OGG в WAV (pydub требует ffmpeg)
        wav_path = ogg_path.replace(".ogg", ".wav")
        audio = AudioSegment.from_ogg(ogg_path)
        audio.export(wav_path, format="wav")
        
        # Расшифровываем
        await processing_msg.edit_text("🤔 Расшифровываю и анализирую...")
        transcribed_text = transcribe_voice(wav_path)
        
        if not transcribed_text:
            await processing_msg.edit_text("❌ Не удалось распознать голосовое сообщение. Попробуй еще раз.")
            return
        
        # Показываем расшифрованный текст
        await processing_msg.edit_text(f"📝 Расшифровано:\n\n_{transcribed_text}_\n\n🧠 Анализирую...")
        
        # Анализируем через DeepSeek
        analysis = analyze_with_deepseek(transcribed_text)
        
        # Отправляем результат
        await message.answer(
            f"🎯 **Результат анализа:**\n\n{analysis}\n\n"
            f"📝 **Расшифровка:**\n{transcribed_text}"
        )
        await processing_msg.delete()
        
    except Exception as e:
        logger.error(f"Ошибка обработки голосового: {e}")
        await processing_msg.edit_text("❌ Произошла ошибка при обработке. Попробуйте позже.")
    finally:
        # Удаляем временные файлы
        for path in [ogg_path, wav_path]:
            if os.path.exists(path):
                os.unlink(path)

@dp.message_handler(content_types=ContentType.TEXT)
async def handle_text(message: Message):
    """Обработчик текстовых сообщений"""
    await bot.send_chat_action(message.chat.id, "typing")
    
    response = analyze_with_deepseek(message.text)
    await message.answer(response)

if __name__ == "__main__":
    # Запуск через webhook
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
