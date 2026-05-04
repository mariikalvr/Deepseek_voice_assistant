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

# Инициализация клиента Groq (теперь с правильной версией)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(bot)

async def on_startup(dp):
    if not groq_client:
        logger.error("КРИТИЧЕСКАЯ ОШИБКА: GROQ_API_KEY не задан!")
        return
    await bot.set_webhook(WEBHOOK_URL)
    logger.info(f"Вебхук успешно установлен на {WEBHOOK_URL}")

async def on_shutdown(dp):
    await bot.delete_webhook()
    logger.info("Вебхук удален")

@dp.message_handler(commands=['start'])
async def start_command(message: Message):
    await message.answer(
        "✅ Бот успешно запущен и работает через бесплатный Groq API!\n\n"
        "📝 Отправь мне голосовое сообщение, и я расшифрую его и отвечу.\n"
        "💬 Или просто напиши текст."
    )

@dp.message_handler(content_types=ContentType.VOICE)
async def handle_voice(message: Message):
    if not groq_client:
        await message.answer("❌ Ошибка: API не настроен. Пожалуйста, сообщите администратору.")
        return

    processing_msg = await message.answer("🎙️ Получил голосовое. Скачиваю и готовлю к расшифровке...")
    ogg_path = None
    wav_path = None
    try:
        file = await bot.get_file(message.voice.file_id)
        file_bytes = await bot.download_file(file.file_path)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_ogg:
            tmp_ogg.write(file_bytes.getvalue())
            ogg_path = tmp_ogg.name

        wav_path = ogg_path.replace(".ogg", ".wav")
        audio = AudioSegment.from_ogg(ogg_path)
        audio.export(wav_path, format="wav")

        await processing_msg.edit_text("🤔 Отправляю на расшифровку...")

        with open(wav_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                language="ru"
            )
        transcribed_text = transcription.text

        if not transcribed_text:
            await processing_msg.edit_text("❌ Не удалось распознать речь. Попробуйте говорить четче или короче.")
            return

        await processing_msg.edit_text("🧠 Расшифровал. Анализирую и придумываю ответ...")

        chat_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Ты дружелюбный и полезный ассистент. Отвечай по-русски, кратко и ясно по существу вопроса."},
                {"role": "user", "content": transcribed_text}
            ]
        )
        answer_text = chat_response.choices[0].message.content

        await message.answer(
            f"💬 **Ответ:**\n{answer_text}\n\n"
            f"📝 **Расшифровка сообщения:**\n{transcribed_text}"
        )
        await processing_msg.delete()

    except Exception as e:
        logger.exception("Ошибка при обработке голосового:")
        await processing_msg.edit_text(f"❌ Произошла ошибка при обработке: {str(e)[:150]}\nПопробуйте еще раз.")
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

    thinking_msg = await message.answer("🤔 Анализирую текст...")
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": message.text}]
        )
        await thinking_msg.delete()
        await message.answer(response.choices[0].message.content)
    except Exception as e:
        logger.exception("Ошибка при обработке текста:")
        await thinking_msg.edit_text(f"❌ Ошибка: {str(e)[:100]}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True,
        host="0.0.0.0",
        port=port
    )
