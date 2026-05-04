import os
import tempfile
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message, ContentType
from aiogram.utils.executor import start_webhook
from groq import Groq
from pydub import AudioSegment

# --- Конфигурация ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RENDER_HOSTNAME = os.getenv("RENDER_EXTERNAL_HOSTNAME")
if not RENDER_HOSTNAME:
    raise RuntimeError("Переменная RENDER_EXTERNAL_HOSTNAME не найдена!")
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"https://{RENDER_HOSTNAME}{WEBHOOK_PATH}"

# --- Инициализация ---
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(bot)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

if not groq_client:
    raise RuntimeError("GROQ_API_KEY не задан!")

# --- Обработчики ---
@dp.message_handler(commands=['start'])
async def start_command(message: Message):
    await message.answer("✅ Бот работает! Отправь голосовое или текст.")

@dp.message_handler(content_types=ContentType.VOICE)
async def handle_voice(message: Message):
    msg = await message.answer("🎙️ Обрабатываю голосовое...")
    try:
        file = await bot.get_file(message.voice.file_id)
        file_bytes = await bot.download_file(file.file_path)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(file_bytes.getvalue())
            ogg_path = tmp.name
        wav_path = ogg_path.replace(".ogg", ".wav")
        AudioSegment.from_ogg(ogg_path).export(wav_path, format="wav")
        with open(wav_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                language="ru"
            )
        os.unlink(ogg_path)
        os.unlink(wav_path)
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": transcription.text}]
        )
        await msg.delete()
        await message.answer(f"📝 {transcription.text}\n\n🤖 {response.choices[0].message.content}")
    except Exception as e:
        await msg.edit_text(f"❌ Ошибка: {str(e)[:100]}")

@dp.message_handler(content_types=ContentType.TEXT)
async def handle_text(message: Message):
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": message.text}]
        )
        await message.answer(response.choices[0].message.content)
    except Exception as e:
        await message.answer(f"❌ Ошибка: {str(e)[:100]}")

# --- Запуск сервера с правильным обработчиком /webhook ---
async def on_startup(dp):
    await bot.set_webhook(WEBHOOK_URL)
    print(f"Webhook set to {WEBHOOK_URL}")

async def on_shutdown(dp):
    await bot.delete_webhook()
    print("Webhook deleted")

if __name__ == "__main__":
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,  # ЭТО ЭНДПОЙНТ НА /WEBHOOK
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
