from aiogram import Bot, Dispatcher
from dotenv import load_dotenv
from app.handlers_analyzer import router as analyzer_router

import logging
import os
import asyncio

logging.basicConfig(level=logging.INFO)

load_dotenv()
TG_TOKEN = os.getenv('TG_TOKEN', 0)

async def main():
    bot = Bot(token=TG_TOKEN)
    dp = Dispatcher()
    
    # Включаем роутер анализатора
    dp.include_router(analyzer_router)
    
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Exit')
