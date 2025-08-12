import json
import time
from datetime import datetime
from data_preparation import load_api_keys, initialize_text_model


def generate_answers(model, questions, excerpt):
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ОТВЕТОВ")
    print("=" * 60)
    
    answers = []
    for i, qa in enumerate(questions):
        question = qa['question']
        print(f"Обработка вопроса {i+1}/{len(questions)}: {question[:50]}...")
        
        prompt = f"""Ответь на вопрос на основе следующего отрывка из "Мастера и Маргарита":

ОТРЫВОК:
{excerpt}

ВОПРОС: {question}

ИНСТРУКЦИИ:
- Отвечай ТОЛЬКО на основе предоставленного отрывка
- Если в отрывке нет информации для ответа, так и скажи
- Дай краткий и точный ответ
- Не добавляй информацию, которой нет в отрывке
- Максимум 2-3 предложения в ответе"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                answer = response.text.strip()

                if answer and len(answer) > 10:
                    low_quality_indicators = [
                        "не могу ответить", "нет информации", "нужно больше контекста",
                        "не предоставлен", "нет данных", "не знаю", "отсутствует",
                        "невозможно определить", "не указано", "не упоминается"
                    ]
                    is_low_quality = any(indicator in answer.lower() for indicator in low_quality_indicators)

                    if not is_low_quality:
                        answers.append({
                            'question': question,
                            'gemini_answer': answer,
                            'reference_answer': qa.get('answer', ''),
                            'question_id': i
                        })
                    else:
                        answers.append({
                            'question': question,
                            'gemini_answer': "Недостаточно информации в отрывке для ответа",
                            'reference_answer': qa.get('answer', ''),
                            'question_id': i
                        })
                else:
                    answers.append({
                        'question': question,
                        'gemini_answer': "Недостаточно информации в отрывке для ответа",
                        'reference_answer': qa.get('answer', ''),
                        'question_id': i
                    })
                
                print(f"✅ Ответ получен: {answer[:50]}...")
                time.sleep(2)
                break
                
            except Exception as e:
                print(f"Попытка {attempt + 1}/{max_retries}: Ошибка при запросе к Gemini: {e}")
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = (attempt + 1) * 30
                    print(f"Превышена квота API. Ожидание {wait_time} секунд...")
                    time.sleep(wait_time)
                else:
                    if attempt == max_retries - 1:
                        answers.append({
                            'question': question,
                            'gemini_answer': "Ошибка получения ответа",
                            'reference_answer': qa.get('answer', ''),
                            'question_id': i
                        })
                    break
    
    print(f"✅ Успешно сгенерировано {len(answers)} ответов")
    return answers


def save_answers(answers, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"answers_{timestamp}.json"
    
    data = {
        "generator": "LLM",
        "timestamp": datetime.now().isoformat(),
        "total_answers": len(answers),
        "answers": answers
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return filename


def run_answer_generation(questions, excerpt):
    print("Запуск генерации ответов...")
    
    api_keys = load_api_keys()
    model = initialize_text_model(api_keys)
    
    answers = generate_answers(model, questions, excerpt)
    
    if answers:
        filename = save_answers(answers)
        print(f"\n✅ Ответы сохранены в файл: {filename}")
        
    
    return answers
    
