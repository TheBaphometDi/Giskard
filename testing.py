import json
import giskard
import pandas as pd
from datetime import datetime
import re
from difflib import SequenceMatcher
from data_preparation import prepare_test_data

def create_giskard_model():
    """Создает модель для Giskard тестирования"""
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
    """Вычисляет схожесть между двумя текстами"""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def extract_key_phrases(text):
    """Извлекает ключевые фразы из текста"""
    if not text:
        return set()
    text = text.lower()
    phrases = re.findall(r'\b\w+(?:\s+\w+){1,4}\b', text)
    return set(phrases)

def calculate_accuracy_score(model_answer, reference_answer, question):
    """Вычисляет оценку точности ответа"""
    if not model_answer or not reference_answer or not question:
        return 0.0
    
    base_score = 0.0

    # Проверка на отказ от ответа
    refusal_phrases = [
        "не могу ответить", "нет информации", "нужно больше контекста",
        "не предоставлен", "нет данных", "не знаю", "отсутствует",
        "невозможно определить", "не указано", "не упоминается",
        "ошибка получения ответа", "недостаточно информации"
    ]
    
    if any(phrase in model_answer.lower() for phrase in refusal_phrases):
        return 0.1

    # Схожесть текста
    similarity = calculate_text_similarity(model_answer, reference_answer)
    base_score += similarity * 0.5

    # Пересечение ключевых фраз
    model_phrases = extract_key_phrases(model_answer)
    ref_phrases = extract_key_phrases(reference_answer)
    
    if ref_phrases:
        phrase_overlap = len(model_phrases.intersection(ref_phrases)) / len(ref_phrases)
        base_score += phrase_overlap * 0.3

    # Релевантность ответа вопросу
    question_words = set(question.lower().split())
    answer_words = set(model_answer.lower().split())
    question_relevance = len(question_words.intersection(answer_words)) / max(len(question_words), 1)
    base_score += question_relevance * 0.2

    # Оценка длины ответа
    if 20 <= len(model_answer) <= 500:
        base_score += 0.1
    elif len(model_answer) > 1000:
        base_score -= 0.1
    
    return min(1.0, max(0.0, base_score))

def validate_data_for_giskard(reference_qa, model_answers):
    """Валидирует данные для Giskard тестов"""
    valid_data = []
    
    for i, qa in enumerate(reference_qa):
        if i >= len(model_answers):
            continue
            
        question = str(qa.get("question", "")).strip()
        reference_answer = str(qa.get("answer", "")).strip()
        model_answer = str(model_answers[i]).strip()
        
        # Дополнительные проверки на валидность
        if (len(question) > 10 and 
            len(reference_answer) > 10 and 
            len(model_answer) > 10 and
            question and reference_answer and model_answer and
            not question.isspace() and 
            not reference_answer.isspace() and 
            not model_answer.isspace()):
            
            # Проверка на невалидные ответы
            invalid_phrases = [
                "ошибка получения ответа", "error", "недостаточно информации",
                "не могу ответить", "нет информации", "нужно больше контекста"
            ]
            
            is_valid = True
            for phrase in invalid_phrases:
                if phrase.lower() in model_answer.lower():
                    is_valid = False
                    break
            
            if is_valid:
                valid_data.append({
                    "question": question,
                    "reference_answer": reference_answer,
                    "model_answer": model_answer
                })
    
    return valid_data

def evaluate_with_giskard(reference_qa, model_answers):
    """Оценка через Giskard с улучшенной валидацией"""
    # Валидация данных
    valid_data = validate_data_for_giskard(reference_qa, model_answers)
    
    if not valid_data:
        print("ОШИБКА: Нет валидных данных для Giskard тестов")
        return [], None
    
    print(f"Валидных записей для Giskard: {len(valid_data)}")
    
    # Создание датасета
    dataset = giskard.Dataset(
        df=pd.DataFrame(valid_data),
        name="qa_test_dataset",
        target=None
    )
    
    model = create_giskard_model()
    test_suite = giskard.Suite()
    
    # Добавление тестов только если есть достаточно данных
    if len(valid_data) >= 5:
        try:
            test_suite.add_test(
                giskard.testing.test_data_uniqueness(
                    dataset=dataset,
                    column="model_answer",
                    threshold=0.3
                )
            )
        except Exception as e:
            print(f"Предупреждение: Не удалось добавить тест уникальности: {e}")
    
    # Тест полноты данных - всегда добавляем
    try:
        test_suite.add_test(
            giskard.testing.test_data_completeness(
                dataset=dataset,
                column_name="model_answer",
                threshold=0.8
            )
        )
    except Exception as e:
        print(f"Предупреждение: Не удалось добавить тест полноты: {e}")
    
    # Запуск тестов
    try:
        results = test_suite.run()
    except Exception as e:
        print(f"Ошибка при запуске Giskard тестов: {e}")
        return [], None
    
    # Вычисление оценок
    scores = []
    base_score = 0.5

    # Подсчет пройденных тестов
    passed_tests = 0
    total_tests = len(results.results)
    
    for test_result in results.results:
        if hasattr(test_result, 'passed') and test_result.passed:
            passed_tests += 1

    if total_tests > 0:
        test_success_rate = passed_tests / total_tests
        base_score += (test_success_rate - 0.5) * 0.1
    
    # Оценка каждого ответа
    for i, qa in enumerate(reference_qa):
        if i >= len(model_answers):
            scores.append(0.0)
            continue
            
        accuracy_score = calculate_accuracy_score(
            model_answers[i], 
            qa["answer"], 
            qa["question"]
        )
        
        score = base_score + accuracy_score * 0.5
        model_answer = model_answers[i]

        # Дополнительные критерии оценки
        if 20 <= len(model_answer) <= 500:
            score += 0.05
        elif len(model_answer) < 10:
            score -= 0.2
        elif len(model_answer) > 1000:
            score -= 0.1

        if model_answer.strip():
            score += 0.05
        else:
            score -= 0.3

        # Штраф за повторяющиеся ответы
        if i > 0 and model_answer == model_answers[i-1]:
            score -= 0.15

        # Штраф за низкокачественные ответы
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
    """Вычисляет метрики оценки"""
    if not scores:
        return {}
        
    accuracy_scores = []
    for i, qa in enumerate(reference_qa):
        if i < len(model_answers):
            acc_score = calculate_accuracy_score(model_answers[i], qa["answer"], qa["question"])
            accuracy_scores.append(acc_score)
    
    if not accuracy_scores:
        return {}
    
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
        "average_answer_length": sum(len(str(ans)) for ans in model_answers) / len(model_answers) if model_answers else 0,
        "unique_answers_ratio": len(set(str(ans) for ans in model_answers)) / len(model_answers) if model_answers else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    return metrics

def save_results_to_file(results_data, filename=None):
    """Сохраняет результаты в JSON файл"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gemini_test_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    return filename

def run_testing():
    """Основная функция тестирования"""
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ GEMINI С GISCARD")
    print("=" * 50)
    
    # Подготовка данных
    excerpt, reference_qa, model_answers = prepare_test_data()
    
    if not excerpt or not reference_qa or not model_answers:
        print("ОШИБКА: Не удалось подготовить данные для тестирования")
        return
    
    print("\nОценка ответов через Giskard...")
    scores, giskard_results = evaluate_with_giskard(reference_qa, model_answers)
    
    if not scores:
        print("ОШИБКА: Не удалось оценить ответы")
        return
    
    metrics = calculate_metrics(scores, reference_qa, model_answers)
    
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 50)
    
    for i, qa in enumerate(reference_qa):
        if i < len(model_answers) and i < len(scores):
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
    
    # Результаты Giskard тестов
    if giskard_results:
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
    else:
        print("\nGiskard тесты не были выполнены")
    
    # Сохранение результатов
    results_data = {
        "excerpt": excerpt,
        "questions_and_answers": [
            {
                "question": qa["question"],
                "reference_answer": qa["answer"],
                "model_answer": model_answers[i] if i < len(model_answers) else "Ошибка",
                "score": scores[i] if i < len(scores) else 0.0,
                "accuracy": calculate_accuracy_score(
                    model_answers[i] if i < len(model_answers) else "", 
                    qa["answer"], 
                    qa["question"]
                )
            }
            for i, qa in enumerate(reference_qa)
        ],
        "metrics": metrics,
        "giskard_results": {
            "total_tests": len(giskard_results.results) if giskard_results else 0,
            "passed_tests": passed_tests if giskard_results else 0,
            "failed_tests": failed_tests if giskard_results else 0
        }
    }
    
    filename = save_results_to_file(results_data)
    print(f"\nРезультаты сохранены в файл: {filename}")

if __name__ == "__main__":
    run_testing() 