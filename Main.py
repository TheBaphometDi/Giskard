import json
import time
import google.generativeai as genai
import giskard
import pandas as pd
from datetime import datetime
import re
from difflib import SequenceMatcher

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
            
            # Попытка исправить распространённые ошибки JSON
            import re
            response_text = response_text.replace('\n', '')
            response_text = re.sub(r',\s*}', '}', response_text)
            response_text = re.sub(r',\s*]', ']', response_text)
            
            result = json.loads(response_text)
            questions = result.get("questions", [])
            
            if not questions and "questions" not in result:
                questions = result if isinstance(result, list) else []
            
            return questions[:20]
            
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
                answers.append(response.text.strip())
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

def create_giskard_model():
    def prediction_function(df):
        return df['model_answer'].tolist()
    
    return giskard.Model(
        prediction_function,
        model_type="text_generation",
        name="gemini_qa_model",
        description="Модель для генерации ответов на вопросы",
        feature_names=["question", "reference_answer", "model_answer"]
    )

def calculate_text_similarity(text1, text2):
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def extract_key_phrases(text):
    text = text.lower()
    phrases = re.findall(r'\b\w+(?:\s+\w+){1,4}\b', text)
    return set(phrases)

def calculate_accuracy_score(model_answer, reference_answer, question):
    base_score = 0.0

    refusal_phrases = [
        "не могу ответить", "нет информации", "нужно больше контекста",
        "не предоставлен", "нет данных", "не знаю", "отсутствует",
        "невозможно определить", "не указано", "не упоминается"
    ]
    
    if any(phrase in model_answer.lower() for phrase in refusal_phrases):
        return 0.1  # Очень низкий балл за отказ
    

    similarity = calculate_text_similarity(model_answer, reference_answer)
    base_score += similarity * 0.5
    

    model_phrases = extract_key_phrases(model_answer)
    ref_phrases = extract_key_phrases(reference_answer)
    
    if ref_phrases:
        phrase_overlap = len(model_phrases.intersection(ref_phrases)) / len(ref_phrases)
        base_score += phrase_overlap * 0.3
    

    question_words = set(question.lower().split())
    answer_words = set(model_answer.lower().split())
    question_relevance = len(question_words.intersection(answer_words)) / max(len(question_words), 1)
    base_score += question_relevance * 0.2
    

    if 20 <= len(model_answer) <= 500:
        base_score += 0.1
    elif len(model_answer) > 1000:
        base_score -= 0.1  # Штраф за избыточность
    
    return min(1.0, max(0.0, base_score))

def evaluate_with_giskard(reference_qa, model_answers):
    test_data = []
    for i, qa in enumerate(reference_qa):
        test_data.append({
            "question": qa["question"],
            "reference_answer": qa["answer"],
            "model_answer": model_answers[i]
        })
    
    dataset = giskard.Dataset(
        df=pd.DataFrame(test_data),
        name="qa_test_dataset",
        target=None
    )
    
    model = create_giskard_model()
    test_suite = giskard.Suite()
    
    test_suite.add_test(
        giskard.testing.test_data_uniqueness(
            dataset=dataset,
            column="model_answer",
            threshold=0.3
        )
    )
    
    test_suite.add_test(
        giskard.testing.test_data_completeness(
            dataset=dataset,
            column_name="model_answer",
            threshold=0.8
        )
    )
    
    results = test_suite.run()
    
    scores = []
    base_score = 0.5
    

    passed_tests = 0
    total_tests = len(results.results)
    
    for test_result in results.results:
        if hasattr(test_result, 'passed') and test_result.passed:
            passed_tests += 1
    

    if total_tests > 0:
        test_success_rate = passed_tests / total_tests

        base_score += (test_success_rate - 0.5) * 0.1
    
    for i, qa in enumerate(reference_qa):
        accuracy_score = calculate_accuracy_score(
            model_answers[i], 
            qa["answer"], 
            qa["question"]
        )
        
        score = base_score + accuracy_score * 0.5
        
        model_answer = model_answers[i]
        

        if 20 <= len(model_answer) <= 500:
            score += 0.05  # Оптимальная длина
        elif len(model_answer) < 10:
            score -= 0.2   # Слишком короткий
        elif len(model_answer) > 1000:
            score -= 0.1   # Слишком длинный

        if model_answer.strip():
            score += 0.05
        else:
            score -= 0.3

        if i > 0 and model_answer == model_answers[i-1]:
            score -= 0.15

        low_quality_phrases = [
            "не знаю", "нет информации", "нужно больше контекста", 
            "не могу ответить", "не предоставлен", "нет данных",
            "отсутствует", "невозможно определить", "не указано"
        ]
        if any(phrase in model_answer.lower() for phrase in low_quality_phrases):
            score -= 0.2
        
        score = max(0.0, min(1.0, score))
        scores.append(score)
    
    return scores, results

def calculate_metrics(scores, reference_qa, model_answers):
    accuracy_scores = []
    for i, qa in enumerate(reference_qa):
        acc_score = calculate_accuracy_score(model_answers[i], qa["answer"], qa["question"])
        accuracy_scores.append(acc_score)
    
    metrics = {
        "average_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "std_score": pd.Series(scores).std(),
        "average_accuracy": sum(accuracy_scores) / len(accuracy_scores),
        "min_accuracy": min(accuracy_scores),
        "max_accuracy": max(accuracy_scores),
        "total_questions": len(scores),
        "high_quality_answers": len([s for s in scores if s >= 0.8]),
        "medium_quality_answers": len([s for s in scores if 0.6 <= s < 0.8]),
        "low_quality_answers": len([s for s in scores if s < 0.6]),
        "high_accuracy_answers": len([s for s in accuracy_scores if s >= 0.8]),
        "medium_accuracy_answers": len([s for s in accuracy_scores if 0.6 <= s < 0.8]),
        "low_accuracy_answers": len([s for s in accuracy_scores if s < 0.6]),
        "average_answer_length": sum(len(ans) for ans in model_answers) / len(model_answers),
        "unique_answers_ratio": len(set(model_answers)) / len(model_answers),
        "timestamp": datetime.now().isoformat()
    }
    
    return metrics

def save_results_to_file(results_data, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gemini_test_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    return filename

def main():
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ GEMINI С GISCARD")
    print("=" * 50)
    
    print("\nЗагрузка API ключей...")
    api_keys = load_api_keys()
    
    print("Инициализация Gemini...")
    gemini_model = initialize_gemini(api_keys)
    
    print("Получение отрывка из 'Мастера и Маргариты'...")
    excerpt = get_excerpt_by_gemini(gemini_model)
    if not excerpt:
        print("ОШИБКА: Не удалось получить отрывок")
        return
    
    print(f"Выбранный отрывок:\n{excerpt}\n")
    print("-" * 50)
    
    print("Генерация вопросов и ответов...")
    reference_qa = generate_questions_with_gemini(gemini_model, excerpt)
    if not reference_qa:
        print("ОШИБКА: Не удалось сгенерировать вопросы")
        return
    
    print(f"Сгенерировано {len(reference_qa)} вопросов")
    
    print("\nОтправка вопросов в модель Gemini...")
    model_answers = get_answers_from_gemini(gemini_model, reference_qa, excerpt)

    if len(model_answers) < len(reference_qa):
        model_answers += ["Ошибка получения ответа"] * (len(reference_qa) - len(model_answers))
    
    print("\nОценка ответов через Giskard...")
    scores, giskard_results = evaluate_with_giskard(reference_qa, model_answers)
    
    metrics = calculate_metrics(scores, reference_qa, model_answers)
    
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 50)
    
    for i, qa in enumerate(reference_qa):
        accuracy = calculate_accuracy_score(model_answers[i], qa["answer"], qa["question"])
        print(f"\n{i+1}. Вопрос: {qa['question']}")
        print(f"   Эталон: {qa['answer']}")
        print(f"   Модель: {model_answers[i][:100]}...")
        print(f"   Оценка: {scores[i]:.2f} (Точность: {accuracy:.2f})")
    
    print("\n" + "=" * 50)
    print("МЕТРИКИ")
    print("=" * 50)
    for key, value in metrics.items():
        if key != "timestamp":
            print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    
    passed_tests = 0
    failed_tests = 0
    print("\nРезультаты тестов Giskard:")
    for i, test_result in enumerate(giskard_results.results):
        test_name = getattr(test_result, 'test_name', f'Тест {i+1}')

        result_text = getattr(test_result, 'result', '')
        is_passed = 'Test succeeded' in str(result_text)
        if is_passed:
            passed_tests += 1
            print(f"  ✅ {test_name}: ПРОЙДЕН")
        else:
            failed_tests += 1
            print(f"  ❌ {test_name}: НЕ ПРОЙДЕН")
    
    print(f"\nИтого: {passed_tests} пройдено, {failed_tests} не пройдено")
    
    results_data = {
        "excerpt": excerpt,
        "questions_and_answers": [
            {
                "question": qa["question"],
                "reference_answer": qa["answer"],
                "model_answer": model_answers[i],
                "score": scores[i],
                "accuracy": calculate_accuracy_score(model_answers[i], qa["answer"], qa["question"])
            }
            for i, qa in enumerate(reference_qa)
        ],
        "metrics": metrics,
        "giskard_results": {
            "total_tests": len(giskard_results.results),
            "passed_tests": passed_tests,
            "failed_tests": failed_tests
        }
    }
    
    filename = save_results_to_file(results_data)
    print(f"\nРезультаты сохранены в файл: {filename}")

if __name__ == "__main__":
    main()
