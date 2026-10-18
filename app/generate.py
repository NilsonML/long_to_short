"""generate.py — запросы к LLM через OpenRouter (OpenAI-совместимый API) с перебором моделей и ретраями."""

from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
import logging
import re

load_dotenv()
AI_TOKEN = os.getenv('AI_TOKEN')

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=AI_TOKEN,
)

logger = logging.getLogger(__name__)

# Актуальный список работающих бесплатных моделей
MODELS = [
    "qwen/qwen-2.5-72b-instruct:free",
    "deepseek/deepseek-r1:free", 
    "google/gemma-7b-it:free",
    "meta-llama/llama-3.1-8b-instruct:free",
]

async def ai_generate(text: str, max_retries=3):
    """Генерация ответа с улучшенной обработкой ошибок"""
    last_error = None
    
    for attempt in range(max_retries):
        model = MODELS[attempt % len(MODELS)]
        
        try:
            logger.info(f"Попытка {attempt + 1} с моделью {model}")
            
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Ты помощник для анализа сообщений. Ты должен строго следовать формату ответа. Отвечай ТОЛЬКО на русском языке. Не добавляй ничего от себя. Сохраняй оригинальные местоимения и детали из сообщения."},
                    {"role": "user", "content": text}
                ],
                timeout=60,
                max_tokens=800
            )
            
            result = completion.choices[0].message.content
            
            # Проверяем, что ответ на русском языке
            if result and len(result.strip()) > 10:
                # Проверяем наличие кириллицы в ответе
                if not re.search(r'[а-яА-Я]', result):
                    logger.warning(f"Ответ не содержит кириллицы, возможно не на русском: {result}")
                    continue
                return result
            else:
                raise Exception("Пустой или слишком короткий ответ от модели")
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            logger.warning(f"Ошибка с моделью {model}: {error_str}")
            
            if "429" in error_str or "rate" in error_str.lower():
                wait_time = min(2 ** attempt + 5, 30)
                logger.info(f"Лимит запросов, ждем {wait_time} секунд")
                await asyncio.sleep(wait_time)
            elif "404" in error_str:
                await asyncio.sleep(1)
                continue
            elif "timeout" in error_str.lower():
                await asyncio.sleep(5)
            else:
                await asyncio.sleep(2)
    
    raise Exception(f"Не удалось получить ответ после {max_retries} попыток")

def clean_ai_response(text: str) -> str:
    """Очищает ответ от AI от нежелательного содержимого"""
    if not text:
        return text
    
    # Удаляем китайские и другие не-русские символы (кроме пунктуации)
    cleaned = re.sub(r'[^\u0400-\u04FF\s\.\,\!\?\-\:\(\)\d]', '', text)
    
    # Удаляем лишние пробелы
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

async def ai_analyze_message(text: str):
    """Анализирует сообщение и возвращает основную мысль и ответ"""
    # Для очень длинных текстов делаем умное сокращение
    original_length = len(text)
    if original_length > 4000:
        # Берем начало, середину и конец текста для сохранения контекста
        part1 = text[:1500]
        part2 = text[original_length//2 - 500:original_length//2 + 500]
        part3 = text[-1500:]
        text = f"{part1}... [пропущена средняя часть] ...{part2}... [пропущена конечная часть] ...{part3}"
        logger.info(f"Текст сокращен с {original_length} до ~{len(text)} символов")
    
    prompt = f"""
    Анализируй сообщение и строго следуй формату:

    Сообщение: {text}

    Твоя задача:
    1. ОСНОВНАЯ МЫСЛЬ: [1-2 предложения, выдели главное]
    2. ОТВЕТ: [1-3 предложения, естественный человеческий ответ]

    ВАЖНЫЕ ПРАВИЛА ДЛЯ ОТВЕТА:
    - Сохраняй оригинальные местоимения (я/ты/вы/мы) из сообщения
    - Не меняй пол говорящего, если он явно указан
    - Не добавляй информацию, которой нет в сообщении
    - Будь точным в деталях
    - Ответ должен звучать естественно и человечно

    ПРАВИЛА ФОРМАТА:
    - Каждая часть должна быть на отдельной строке
    - Начинай с "ОСНОВНАЯ МЫСЛЬ:", затем с новой строки "ОТВЕТ:"
    - Отвечай ТОЛЬКО на русском
    - Не добавляй лишних слов, только указанный формат
    - Ответ должен быть готов для отправки собеседнику
    """
    
    response = await ai_generate(prompt)
    return clean_ai_response(response)

async def analyze_long_text(text: str):
    """Анализирует очень длинные тексты с разбиением на части"""
    if len(text) <= 4000:
        return await ai_analyze_message(text)
    
    # Для очень длинных текстов разбиваем на части и анализируем каждую
    chunk_size = 3500
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    if len(chunks) > 3:
        chunks = chunks[:3]
        chunks[-1] = chunks[-1] + "... [текст сокращен]"
    
    analyses = []
    for i, chunk in enumerate(chunks):
        try:
            analysis = await ai_analyze_message(f"Часть {i+1} из {len(chunks)}: {chunk}")
            analyses.append(analysis)
        except Exception as e:
            logger.error(f"Ошибка анализа части {i+1}: {e}")
            analyses.append(f"ОСНОВНАЯ МЫСЛЬ: Не удалось проанализировать часть {i+1}\nОТВЕТ: ")
    
    # Объединяем анализы
    combined_analysis = "\n\n".join(analyses)
    
    # Делаем финальный анализ объединенных результатов
    final_prompt = f"""
    Проанализируй объединенные части длинного сообщения:

    {combined_analysis}

    Выдели ОСНОВНУЮ мысль всего сообщения и придумай ОТВЕТ.

    ВАЖНО: 
    - Сохраняй оригинальные местоимения и детали из сообщения
    - Не меняй пол говорящего
    - Будь точным в деталях

    Строго соблюдай формат:
    ОСНОВНАЯ МЫСЛЬ: [2-3 предложения]
    ОТВЕТ: [2-4 предложения]

    Отвечай только на русском, без лишних слов.
    """
    
    response = await ai_generate(final_prompt)
    return clean_ai_response(response)

async def simple_text_analysis(text: str):
    """Умный резервный анализ без использования AI"""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if len(sentences) > 3:
        main_idea = sentences[0] + ". " + sentences[len(sentences)//2] + ". " + sentences[-1] + "."
        
        # Анализируем тон и содержание сообщения
        if any(word in text.lower() for word in ['проблема', 'сложно', 'трудно', 'не знаю']):
            answer = "Понимаю, что ситуация непростая. Давайте вместе подумаем над решением."
        elif any(word in text.lower() for word in ['вопрос', 'интересно', 'хочу знать']):
            answer = "Это действительно интересный вопрос. Я думаю, стоит рассмотреть разные аспекты."
        elif any(word in text.lower() for word in ['рад', 'хорошо', 'отлично', 'спасибо']):
            answer = "Рад это слышать! Продолжайте в том же духе."
        else:
            answer = "Спасибо за развернутое сообщение. Я готов помочь вам разобраться в этом вопросе."
    else:
        main_idea = text[:200] + "..." if len(text) > 200 else text
        answer = "Понимаю вашу ситуацию. Давайте обсудим возможные варианты."
    
    return f"ОСНОВНАЯ МЫСЛЬ: {main_idea}\nОТВЕТ: {answer}"