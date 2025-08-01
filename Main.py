import json
import time
import google.generativeai as genai
import requests

# Импортируем нашу альтернативную реализацию Giscard
try:
    from giscard_alternative import giscard_alt
    GISCARD_AVAILABLE = True
    print("✅ GiscardAlternative найден и готов к использованию")
except ImportError:
    GISCARD_AVAILABLE = False
    print("⚠️  GiscardAlternative не найден, будет использоваться только Gemini")

# --- 1. Загрузка ключей API из Key.json ---
def load_api_keys(path='Key.json'):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        'gemini_api_key': data.get('gemini_api_key'),  # Убираем неправильный strip()
    }

# --- 2. Инициализация моделей ---
def initialize_models(api_keys):
    # Инициализация Gemini
    genai.configure(api_key=api_keys['gemini_api_key'])
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Инициализация GiscardAlternative (если доступен)
    giscard = None
    if GISCARD_AVAILABLE:
        try:
            giscard = giscard_alt
            print("✅ GiscardAlternative инициализирован")
        except Exception as e:
            print(f"⚠️  Ошибка инициализации GiscardAlternative: {e}")
            giscard = None
    
    return gemini_model, giscard

# --- 3. Модель Gemini выбирает отрывок из Мастера и Маргариты ---
def get_excerpt_by_gemini(gemini_model):
    prompt = (
        "Выбери значительный, информативный отрывок из романа 'Мастер и Маргарита' Михаила Булгакова "
        "длиной примерно 300-500 слов. Отрывок должен быть самодостаточным для составления вопросов по содержанию. "
        "Верни только текст отрывка без дополнительных комментариев."
    )
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Ошибка при получении отрывка: {e}")
        # Fallback отрывок
        return (
            "В час жаркого весеннего заката на Патриарших прудах появились два гражданина. "
            "Первый из них — приблизительно сорокалетний, одетый в серенькую летнюю пару, — был маленького роста, "
            "темноволос, упитан, лыс, свою приличную шляпу пирожком нес в руке, а на хорошо выбритом лице его "
            "помещались сверхъестественных размеров очки в черной роговой оправе. Второй — плечистый, рыжеватый, "
            "вихрастый молодой человек в заломленной на затылок клетчатой кепке — был в ковбойке, жеваных белых "
            "брюках и в черных тапочках."
        )

# --- 4. GiscardAlternative составляет около 20 вопросов и ответов ---
def generate_reference_qa_giscard(text, giscard, gemini_model):
    if giscard is not None:
        try:
            print("Используем GiscardAlternative для генерации вопросов...")
            # Используем GiscardAlternative для генерации вопросов
            questions = giscard.generate_questions(text, num_questions=20)
            
            # Получаем эталонные ответы для каждого вопроса
            qa_list = []
            for question in questions:
                answer = giscard.generate_answer(text, question)
                qa_list.append({
                    "question": question,
                    "answer": answer
                })
            
            return qa_list
        except Exception as e:
            print(f"Ошибка при генерации вопросов через GiscardAlternative: {e}")
            print("Переключаемся на Gemini...")
    
    # Fallback через Gemini
    return generate_fallback_qa(text, gemini_model)

# --- 5. Fallback генерация вопросов через Gemini ---
def generate_fallback_qa(text, gemini_model):
    print("Используем Gemini для генерации вопросов...")
    prompt = f"""
    Проанализируй следующий отрывок из романа "Мастер и Маргарита" и составь 20 разнообразных вопросов с правильными ответами.
    
    Отрывок:
    {text}
    
    Создай вопросы разных типов:
    - Вопросы на понимание сюжета
    - Вопросы о персонажах
    - Вопросы о деталях
    - Вопросы на анализ текста
    
    Верни результат в формате JSON:
    [
        {{"question": "вопрос", "answer": "ответ"}},
        ...
    ]
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        
        response_text = response.text
        if '[' in response_text and ']' in response_text:
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            json_str = response_text[start:end]
            qa_list = json.loads(json_str)
            return qa_list[:20]
        else:
            raise ValueError("Не удалось найти JSON в ответе")
    except Exception as e:
        print(f"Ошибка при fallback генерации: {e}")
        # Базовые вопросы
        return [
            {"question": "Кто был на Патриарших прудах?", "answer": "Берлиоз и Бездомный"},
            {"question": "Как был одет первый гражданин?", "answer": "В серенькую летнюю пару"},
            {"question": "Сколько лет было первому гражданину?", "answer": "Приблизительно сорока лет"},
            {"question": "Какой был молодой человек?", "answer": "Плечистый, рыжеватый, вихрастый"},
            {"question": "Во что был одет молодой человек?", "answer": "В ковбойке, жеваных белых брюках и черных тапочках"}
        ]

# --- 6. Запросы к Gemini с задержкой ---
def ask_gemini(question, gemini_model):
    max_retries = 3
    retry_delay = 60  # Увеличиваем задержку до 60 секунд
    
    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(question)
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "quota" in error_str.lower():
                print(f"Превышена квота API. Ожидание {retry_delay} секунд...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Увеличиваем задержку с каждой попыткой
                continue
            else:
                print(f"Ошибка при запросе к Gemini: {e}")
                return "Ошибка получения ответа"
    
    return "Не удалось получить ответ после нескольких попыток"

# --- 7. Оценка ответов через GiscardAlternative или Gemini ---
def evaluate_answers(reference_qa, model_answers, giscard, gemini_model):
    results = {}
    
    for i, qa in enumerate(reference_qa):
        score = 0.5  # Значение по умолчанию
        
        if giscard is not None:
            try:
                # Используем GiscardAlternative для оценки
                score = giscard.evaluate_answer(
                    question=qa['question'],
                    reference_answer=qa['answer'],
                    model_answer=model_answers[i]
                )
            except Exception as e:
                print(f"Ошибка при оценке через GiscardAlternative: {e}")
                # Fallback оценка через Gemini
                score = evaluate_with_gemini(qa, model_answers[i], gemini_model)
        else:
            # Используем Gemini для оценки
            score = evaluate_with_gemini(qa, model_answers[i], gemini_model)
        
        results[qa['question']] = {
            'score': score,
            'reference': qa['answer'],
            'model': model_answers[i]
        }
    
    return results

# --- 8. Fallback оценка через Gemini ---
def evaluate_with_gemini(qa, model_answer, gemini_model):
    prompt = f"""
    Оцени ответ модели на вопрос. Используй шкалу от 0 до 1, где:
    0 - полностью неправильный ответ
    1 - полностью правильный ответ
    
    Вопрос: {qa['question']}
    Правильный ответ: {qa['answer']}
    Ответ модели: {model_answer}
    
    Верни только число от 0 до 1.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        score = float(response.text.strip())
        return max(0, min(1, score))  # Ограничиваем от 0 до 1
    except:
        return 0.5  # Средняя оценка при ошибке

# --- 9. Основной пайплайн ---
def main():
    print("="*50)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ GEMINI 2.0 FLASH")
    print("="*50)
    
    print("\nЗагрузка API ключей...")
    api_keys = load_api_keys()
    
    print("Инициализация моделей...")
    gemini_model, giscard = initialize_models(api_keys)
    
    print("Выбор отрывка из 'Мастера и Маргариты'...")
    excerpt = get_excerpt_by_gemini(gemini_model)
    print("Выбранный отрывок:\n", excerpt, "\n---\n")
    
    print("Генерация вопросов и ответов...")
    reference_qa = generate_reference_qa_giscard(excerpt, giscard, gemini_model)
    print(f"Сгенерировано {len(reference_qa)} вопросов\n")
    
    print("Отправка вопросов в модель Gemini...")
    model_answers = []
    for i, qa in enumerate(reference_qa):
        print(f"Вопрос {i+1}/{len(reference_qa)}: {qa['question'][:50]}...")
        answer = ask_gemini(qa['question'], gemini_model)
        model_answers.append(answer)
        time.sleep(2)  # Задержка между запросами
    
    print("\nОценка ответов...")
    results = evaluate_answers(reference_qa, model_answers, giscard, gemini_model)
    
    # Вывод результатов
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("="*50)
    
    total_score = 0
    for i, (question, res) in enumerate(results.items()):
        print(f"\n{i+1}. Вопрос: {question}")
        print(f"   Эталон: {res['reference']}")
        print(f"   Модель: {res['model'][:100]}...")
        print(f"   Оценка: {res['score']:.2f}")
        total_score += res['score']
    
    avg_score = total_score / len(results)
    print(f"\n" + "="*50)
    print(f"СРЕДНЯЯ ОЦЕНКА: {avg_score:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()
