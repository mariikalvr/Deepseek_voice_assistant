import os
import logging
import tempfile
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message, ContentType
from aiogram.utils.executor import start_webhook
from dotenv import load_dotenv
from groq import Groq
from pydub import AudioSegment

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEBHOOK_HOST = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME', 'localhost')}"
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(bot)
groq_client = Groq(api_key=GROQ_API_KEY)

async def on_startup(dp):
    await bot.set_webhook(WEBHOOK_URL)
    logger.info(f"Webhook установлен на {WEBHOOK_URL}")

async def on_shutdown(dp):
    await bot.delete_webhook()
    logger.info("Webhook удален")

@dp.message_handler(commands=['start'])
async def start_command(message: Message):
    await message.answer("👋 Привет! Я бот на Groq. Отправь голосовое или текст!")

@dp.message_handler(content_types=ContentType.VOICE)
async def handle_voice(message: Message):
    await message.answer("🎙️ Обрабатываю голосовое...")
    try:
        file = await bot.get_file(message.voice.file_id)
        file_bytes = await bot.download_file(file.file_path)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(file_bytes.getvalue())
            tmp_path = tmp.name
        wav_path = tmp_path.replace(".ogg", ".wav")
        audio = AudioSegment.from_ogg(tmp_path)
        audio.export(wav_path, format="wav")
        with open(wav_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                language="ru"
            )
        os.unlink(tmp_path)
        os.unlink(wav_path)
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Ты помощник. Отвечай кратко на русском."},
                {"role": "user", "content": transcription.text}
            ]
        )
        await message.answer(f"📝 {transcription.text}\n\n🤖 {response.choices[0].message.content}")
    except Exception as e:
        await message.answer(f"❌ Ошибка: {str(e)[:100]}")
        logger.error(f"Ошибка: {e}")

@dp.message_handler()
async def handle_text(message: Message):
    await message.answer("🤔 Думаю...")
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": message.text}]
        )
        await message.answer(response.choices[0].message.content)
    except Exception as e:
        await message.answer(f"❌ Ошибка: {str(e)[:100]}")

if __name__ == "__main__":
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
