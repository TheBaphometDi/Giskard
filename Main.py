from giskard_question_generation import run_giskard_generation
from giskard_evaluation import run_giskard_evaluation
from gemini_answer_generation import run_gemini_answer_generation

def main():
    print("=" * 60)
    print("ПОЛНЫЙ РАБОЧИЙ ПРОЦЕСС: GEMINI (выбор текста + ответы) + GISCARD (вопросы + оценка)")
    print("=" * 60)
    print("Использует модульную архитектуру:")
    print("1. giskard_question_generation.py: получение отрывка через Gemini + генерация вопросов через Giskard")
    print("2. gemini_answer_generation.py: генерация ответов через Gemini")
    print("3. giskard_evaluation.py: оценка ответов с помощью метрик Giskard")
    print("=" * 60)
    
    print("\n1️⃣ Запуск генерации вопросов через Giskard (с получением отрывка через Gemini)...")
    result = run_giskard_generation(return_data=True)
    if result and len(result) == 2:
        questions, excerpt = result
        if questions and excerpt:
            print(f"\n✅ Получено {len(questions)} вопросов от Giskard")
            print(f"✅ Отрывок получен через Gemini (длина: {len(excerpt)} символов)")
            
            print("\n2️⃣ Запуск генерации ответов через Gemini...")
            gemini_answers = run_gemini_answer_generation(questions, excerpt)
            
            if gemini_answers:
                print(f"\n✅ Получено {len(gemini_answers)} ответов от Gemini")
                
                print("\n3️⃣ Запуск оценки ответов с помощью метрик Giskard...")
                evaluation_results = run_giskard_evaluation(questions, excerpt, gemini_answers)
                
                if evaluation_results:
                    print("\n📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
                    print(f"   Вопросов сгенерировано (Giskard): {evaluation_results.get('total_questions', 0)}")
                    print(f"   Ответов получено (Gemini): {len(gemini_answers)}")
                    print(f"   Точность: {evaluation_results.get('accuracy', 0):.2%}")
                    print(f"   Процент успеха: {evaluation_results.get('success_rate', 0):.1f}%")
                    
                    if 'automatic_metrics' in evaluation_results:
                        auto_metrics = evaluation_results['automatic_metrics']
                        print(f"\n🤖 АВТОМАТИЧЕСКИЕ МЕТРИКИ GISKARD:")
                        print(f"   - Правильных ответов: {auto_metrics.get('correct_answers', 0)}/{auto_metrics.get('total_questions', 0)}")
                        print(f"   - Точность: {auto_metrics.get('accuracy', 0):.2%}")
                        print(f"   - Процент успеха: {auto_metrics.get('success_rate', 0):.1f}%")
                    
                    if 'evaluation_results' in evaluation_results:
                        print(f"\n📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ (первые 5 вопросов):")
                        for i, result in enumerate(evaluation_results['evaluation_results'][:5]):
                            print(f"   Вопрос {result['question_id']+1}: {'✅' if result.get('correctness', False) else '❌'}")
                            print(f"      Вопрос (Giskard): {result['question'][:60]}...")
                            print(f"      Ответ (Gemini): {result['gemini_answer'][:60]}...")
                            print()
                        
                        if len(evaluation_results['evaluation_results']) > 5:
                            print(f"   ... и еще {len(evaluation_results['evaluation_results']) - 5} вопросов")
                else:
                    print("❌ Не удалось оценить ответы")
            else:
                print("❌ Не удалось получить ответы от Gemini")
        else:
            print("❌ Не удалось получить данные для оценки")
    else:
        print("❌ Не удалось сгенерировать вопросы")

if __name__ == "__main__":
    main()
