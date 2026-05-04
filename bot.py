import os
import logging
import tempfile
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message, ContentType
from aiogram.utils.executor import start_webhook
from groq import Groq
from pydub import AudioSegment

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Переменные окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Определяем URL для вебхука (ВАЖНО!)
# Render сам подставляет переменную RENDER_EXTERNAL_HOSTNAME
RENDER_HOSTNAME = os.getenv("RENDER_EXTERNAL_HOSTNAME")
if not RENDER_HOSTNAME:
    logger.error("Переменная RENDER_EXTERNAL_HOSTNAME не найдена!")
    RENDER_HOSTNAME = "voice-assistant-djk2.onrender.com"  # Запасной вариант

WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"https://{RENDER_HOSTNAME}{WEBHOOK_PATH}"

# Инициализация
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(bot)

# Инициализация Groq
try:
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    if not groq_client:
        logger.error("GROQ_API_KEY не задан!")
except Exception as e:
    logger.error(f"Ошибка инициализации Groq: {e}")
    groq_client = None

@dp.message_handler(commands=['start'])
async def start_command(message: Message):
    await message.answer(
        "✅ Бот успешно запущен!\n"
        "Отправь голосовое сообщение или просто напиши текст."
    )

@dp.message_handler(content_types=ContentType.VOICE)
async def handle_voice(message: Message):
    if not groq_client:
        await message.answer("❌ Ошибка: API не настроен.")
        return
    
    processing_msg = await message.answer("🎙️ Обрабатываю голосовое...")
    ogg_path = None
    wav_path = None
    
    try:
        # Скачиваем
        file = await bot.get_file(message.voice.file_id)
        file_bytes = await bot.download_file(file.file_path)
        
        # Сохраняем .ogg
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(file_bytes.getvalue())
            ogg_path = tmp.name
        
        # Конвертируем в .wav
        wav_path = ogg_path.replace(".ogg", ".wav")
        audio = AudioSegment.from_ogg(ogg_path)
        audio.export(wav_path, format="wav")
        
        await processing_msg.edit_text("🤔 Расшифровываю...")
        
        # Расшифровка через Groq Whisper
        with open(wav_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                language="ru"
            )
        transcribed_text = transcription.text
        
        if not transcribed_text:
            await processing_msg.edit_text("❌ Не удалось распознать речь.")
            return
        
        await processing_msg.edit_text("🧠 Анализирую...")
        
        # Ответ через Groq Llama
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Ты полезный помощник. Отвечай кратко и по-русски."},
                {"role": "user", "content": transcribed_text}
            ],
            max_tokens=500
        )
        
        await message.answer(
            f"📝 **Расшифровка:**\n{transcribed_text}\n\n"
            f"🤖 **Ответ:**\n{response.choices[0].message.content}"
        )
        await processing_msg.delete()
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        await processing_msg.edit_text(f"❌ Ошибка: {str(e)[:100]}")
    finally:
        for path in [ogg_path, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

@dp.message_handler(content_types=ContentType.TEXT)
async def handle_text(message: Message):
    if not groq_client:
        await message.answer("❌ Ошибка: API не настроен.")
        return
    
    thinking_msg = await message.answer("🤔 Думаю...")
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": message.text}],
            max_tokens=500
        )
        await thinking_msg.delete()
        await message.answer(response.choices[0].message.content)
    except Exception as e:
        await thinking_msg.edit_text(f"❌ Ошибка: {str(e)[:100]}")

# Запуск с вебхуком
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    
    # Функции для старта и остановки
    async def on_startup(dp):
        logger.info(f"Устанавливаю вебхук на {WEBHOOK_URL}")
        await bot.set_webhook(WEBHOOK_URL)
        logger.info("Вебхук установлен успешно!")
    
    async def on_shutdown(dp):
        logger.info("Удаляю вебхук...")
        await bot.delete_webhook()
        logger.info("Вебхук удален")
    
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True,
        host="0.0.0.0",
        port=port
    )
