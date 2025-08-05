import json
import time
import google.generativeai as genai
import pandas as pd
from datetime import datetime
import re

def load_api_keys():
    with open('Key.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        return {
            'gemini_api_key': data.get('gemini_api_key')
        }

def initialize_gemini(api_keys):
    genai.configure(api_key=api_keys['gemini_api_key'])
    return genai.GenerativeModel('gemini-2.5-flash')

def get_excerpt_by_gemini(gemini_model):
    prompt = """Выбери значительный отрывок из романа "Мастер и Маргарита" Михаила Булгакова (примерно 500-800 слов). 
    Отрывок должен быть содержательным и подходящим для создания вопросов. 
    Верни только текст отрывка без дополнительных комментариев."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Ошибка при получении отрывка: {e}")
        return None

def generate_questions_with_gemini(gemini_model, excerpt):
    prompt = f"""На основе этого отрывка из "Мастера и Маргариты" создай 20 разнообразных вопросов для тестирования понимания текста:

{excerpt}

Создай вопросы разного типа:
- Вопросы на понимание сюжета
- Вопросы на анализ персонажей  
- Вопросы на интерпретацию
- Вопросы на детали

Верни результат в формате JSON:
{{
    "questions": [
        {{"question": "вопрос", "answer": "правильный ответ"}},
        ...
    ]
}}

Важно: верни только валидный JSON без дополнительного текста."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            if not response_text.startswith('{'):
                start = response_text.find('{')
                if start != -1:
                    response_text = response_text[start:]
            if not response_text.endswith('}'):
                end = response_text.rfind('}')
                if end != -1:
                    response_text = response_text[:end+1]
            response_text = response_text.replace('\n', '')
            response_text = re.sub(r',\s*}', '}', response_text)
            response_text = re.sub(r',\s*]', ']', response_text)
            result = json.loads(response_text)
            questions = result.get("questions", [])
            if not questions and "questions" not in result:
                questions = result if isinstance(result, list) else []
            valid_questions = []
            for qa in questions[:20]:
                if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                    question = str(qa['question']).strip()
                    answer = str(qa['answer']).strip()
                    if len(question) > 10 and len(answer) > 5:
                        valid_questions.append({
                            'question': question,
                            'answer': answer
                        })
            return valid_questions
        except Exception as e:
            print(f"Попытка {attempt + 1}/{max_retries}: Ошибка при генерации вопросов: {e}")
            if 'response_text' in locals():
                print(f"Ответ модели (фрагмент): {response_text[:500]}")
            if "429" in str(e) or "quota" in str(e).lower():
                wait_time = (attempt + 1) * 60
                print(f"Превышена квота API. Ожидание {wait_time} секунд...")
                time.sleep(wait_time)
            else:
                if attempt == max_retries - 1:
                    print(f"Ответ модели: {response.text[:500] if 'response' in locals() else 'Нет ответа'}...")
                break
    return []

def get_answers_from_gemini(gemini_model, questions, excerpt):
    answers = []
    for i, qa in enumerate(questions):
        prompt = f"""Ответь на вопрос на основе следующего отрывка из "Мастера и Маргариты":

ОТРЫВОК:
{excerpt}

ВОПРОС: {qa['question']}

ИНСТРУКЦИИ:
- Отвечай ТОЛЬКО на основе предоставленного отрывка
- Если в отрывке нет информации для ответа, так и скажи
- Дай краткий и точный ответ
- Не добавляй информацию, которой нет в отрывке"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = gemini_model.generate_content(prompt)
                answer = response.text.strip()
                if answer and len(answer) > 10:
                    low_quality_indicators = [
                        "не могу ответить", "нет информации", "нужно больше контекста",
                        "не предоставлен", "нет данных", "не знаю", "отсутствует",
                        "невозможно определить", "не указано", "не упоминается"
                    ]
                    is_low_quality = any(indicator in answer.lower() for indicator in low_quality_indicators)
                    if not is_low_quality:
                        answers.append(answer)
                    else:
                        answers.append("Недостаточно информации в отрывке для ответа")
                else:
                    answers.append("Недостаточно информации в отрывке для ответа")
                print(f"Вопрос {i+1}/{len(questions)}: {qa['question'][:50]}...")
                time.sleep(10)
                break
            except Exception as e:
                print(f"Попытка {attempt + 1}/{max_retries}: Ошибка при запросе к Gemini: {e}")
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = (attempt + 1) * 60
                    print(f"Превышена квота API. Ожидание {wait_time} секунд...")
                    time.sleep(wait_time)
                else:
                    if attempt == max_retries - 1:
                        answers.append("Ошибка получения ответа")
                    break
    return answers

def prepare_test_data():
    print("=" * 50)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ ТЕСТИРОВАНИЯ")
    print("=" * 50)
    print("\nЗагрузка API ключей...")
    api_keys = load_api_keys()
    print("Инициализация Gemini...")
    gemini_model = initialize_gemini(api_keys)
    print("Получение отрывка из 'Мастера и Маргариты'...")
    excerpt = get_excerpt_by_gemini(gemini_model)
    if not excerpt:
        print("ОШИБКА: Не удалось получить отрывок")
        return None, None, None
    print(f"Выбранный отрывок:\n{excerpt}\n")
    print("-" * 50)
    print("Генерация вопросов и ответов...")
    reference_qa = generate_questions_with_gemini(gemini_model, excerpt)
    if not reference_qa:
        print("ОШИБКА: Не удалось сгенерировать вопросы")
        return None, None, None
    print(f"Сгенерировано {len(reference_qa)} валидных вопросов")
    print("\nОтправка вопросов в модель Gemini...")
    model_answers = get_answers_from_gemini(gemini_model, reference_qa, excerpt)
    if len(model_answers) < len(reference_qa):
        model_answers += ["Ошибка получения ответа"] * (len(reference_qa) - len(model_answers))
    print(f"Получено {len(model_answers)} ответов от модели")
    return excerpt, reference_qa, model_answers

if __name__ == "__main__":
    excerpt, reference_qa, model_answers = prepare_test_data()
    if excerpt and reference_qa and model_answers:
        print("\nДанные успешно подготовлены!")
        print(f"Отрывок: {len(excerpt)} символов")
        print(f"Вопросов: {len(reference_qa)}")
        print(f"Ответов: {len(model_answers)}")
    else:
        print("\nОшибка при подготовке данных!") 