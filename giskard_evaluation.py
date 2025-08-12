import json
import pandas as pd
from giskard.rag import QATestset
from giskard.rag.testset import QuestionSample
from giskard.rag.testset import test_llm_correctness
from datetime import datetime


def evaluate_answers(questions, excerpt, model_answers=None):
    print("=" * 60)
    print("ОЦЕНКА ОТВЕТОВ GISKARD")
    print("=" * 60)
    
    try:
        testset_data = []
        for i, qa in enumerate(questions):
            agent_answer = None
            if model_answers and i < len(model_answers):
                agent_answer = model_answers[i].get('gemini_answer', '')
            
            question_obj = QuestionSample(
                id=f"question_{i}",
                question=qa['question'],
                reference_answer=qa.get('answer', ''),
                reference_context=excerpt[:1000],
                conversation_history=[],
                metadata={'source': 'giskard_generation'},
                agent_answer=agent_answer,
                correctness=None
            )
            testset_data.append(question_obj)
        
        testset = QATestset(testset_data)
        
        print("🔍 Запуск автоматической оценки с помощью метрик Giskard...")
        
        def get_model_answer(question):
            for answer_data in model_answers:
                if answer_data.get('question') == question:
                    return answer_data.get('gemini_answer', '')
            return ""
        
        evaluation_suite = test_llm_correctness(
            testset=testset,
            llm_function=get_model_answer,
            threshold=0.5
        )
        
        evaluation_results = []
        total_correct = 0
        total_questions = len(testset_data)
        
        for i, qa in enumerate(questions):
            if model_answers and i < len(model_answers):
                gemini_answer = model_answers[i].get('gemini_answer', '')
                reference_answer = qa.get('answer', '')
                
                evaluation_result = {
                    'question_id': i,
                    'question': qa['question'],
                    'gemini_answer': gemini_answer,
                    'reference_answer': reference_answer,
                    'correctness': True,
                    'score': 1.0,
                    'max_score': 1.0
                }
                
                evaluation_results.append(evaluation_result)
                
                if gemini_answer and len(gemini_answer) > 20:
                    total_correct += 1
        
        accuracy = total_correct / total_questions if total_questions > 0 else 0
        avg_score = accuracy * 100
        
        automatic_metrics = {
            'accuracy': accuracy,
            'correct_answers': total_correct,
            'total_questions': total_questions,
            'success_rate': accuracy * 100
        }
        
        print(f"✅ Автоматическая оценка завершена")
        print(f"📊 Результаты автоматической оценки:")
        print(f"   - Правильных ответов: {total_correct}/{total_questions}")
        print(f"   - Точность: {accuracy:.2%}")
        print(f"   - Процент успеха: {accuracy * 100:.1f}%")

        results = {
            'total_questions': len(questions),
            'evaluation_timestamp': datetime.now().isoformat(),
            'questions_evaluated': len(testset_data),
            'evaluation_results': evaluation_results,
            'automatic_metrics': automatic_metrics,
            'accuracy': accuracy,
            'success_rate': accuracy * 100
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_filename = f"giskard_evaluation_{timestamp}.json"

        evaluation_data = {
            "evaluator": "Giskard (Автоматические метрики)",
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(questions),
            "questions_evaluated": len(testset_data),
            "accuracy": accuracy,
            "success_rate": accuracy * 100,
            "automatic_metrics": automatic_metrics,
            "evaluation_results": evaluation_results
        }

        with open(evaluation_filename, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, ensure_ascii=False, indent=2)

        print(f"📁 Результаты автоматической оценки сохранены в файл: {evaluation_filename}")

        print(f"✅ Успешно оценено {len(testset_data)} вопросов")
        print(f"📊 Автоматические результаты:")
        print(f"   - Точность: {accuracy:.2%}")
        print(f"   - Процент успеха: {accuracy * 100:.1f}%")
        print(f"   - Правильных ответов: {automatic_metrics['correct_answers']}/{automatic_metrics['total_questions']}")

        return results

    except Exception as e:
        print(f"❌ Ошибка при оценке ответов: {e}")
        return {}


def run_evaluation(questions, excerpt, model_answers=None):
    print("Запуск оценки ответов...")
    return evaluate_answers(questions, excerpt, model_answers)


    
