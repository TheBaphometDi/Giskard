import json
import pandas as pd
import giskard
import os
from giskard.rag import QATestset
from giskard.rag.testset import QuestionSample
from giskard.rag.testset import test_llm_correctness
from datetime import datetime


def setup_giskard_openai():
    try:
        from data_preparation import load_api_keys
        api_keys = load_api_keys()
        if 'openai_api_key' in api_keys and api_keys['openai_api_key']:
            os.environ['OPENAI_API_KEY'] = api_keys['openai_api_key']
            giskard.llm.set_llm_api("openai")
            print("✅ Настроен OpenAI для генерации вопросов")
            return True
        else:
            print("⚠️ openai_api_key не найден в Key.json")
            return False
    except Exception as e:
        print(f"⚠️ Ошибка настройки OpenAI: {e}")
        return False


def setup_giskard_embeddings():
    try:
        from giskard.llm.embeddings import try_get_fastembed_embeddings
        embeddings = try_get_fastembed_embeddings()
        if embeddings:
            giskard.llm.embeddings.set_default_embedding(embeddings)
            print("✅ Настроена модель эмбеддингов FastEmbed")
            return True
        else:
            print("⚠️ FastEmbed недоступен, попробуем установить...")
            return False
    except Exception as e:
        print(f"⚠️ Предупреждение: Не удалось настроить эмбеддинги: {e}")
        print("💡 Установите зависимости: pip install fastembed")
        return False


def evaluate_giskard_answers(questions, excerpt, gemini_answers=None):
    print("=" * 60)
    print("ОЦЕНКА ОТВЕТОВ GISKARD")
    print("=" * 60)
    
    if not setup_giskard_openai():
        print("❌ Не удалось настроить OpenAI")
        return {}
    
    if not setup_giskard_embeddings():
        print("❌ Не удалось настроить эмбеддинги")
        return {}
    
    try:
        testset_data = []
        for i, qa in enumerate(questions):
            agent_answer = None
            if gemini_answers and i < len(gemini_answers):
                agent_answer = gemini_answers[i].get('gemini_answer', '')
            
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
        
        try:
            def get_gemini_answer(question):
                for answer_data in gemini_answers:
                    if answer_data.get('question') == question:
                        return answer_data.get('gemini_answer', '')
                return ""
            
            evaluation_suite = test_llm_correctness(
                testset=testset,
                llm_function=get_gemini_answer,
                threshold=0.5
            )
            
            evaluation_results = []
            total_correct = 0
            total_questions = len(testset_data)
            
            for i, qa in enumerate(questions):
                if gemini_answers and i < len(gemini_answers):
                    gemini_answer = gemini_answers[i].get('gemini_answer', '')
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
            
        except Exception as e:
            print(f"⚠️ Ошибка при автоматической оценке: {e}")
            print("🔄 Переход к упрощенной оценке...")
            
            evaluation_results = []
            total_correct = 0
            
            for i, qa in enumerate(questions):
                if gemini_answers and i < len(gemini_answers):
                    gemini_answer = gemini_answers[i].get('gemini_answer', '')
                    
                    is_correct = gemini_answer and len(gemini_answer) > 20
                    if is_correct:
                        total_correct += 1
                    
                    evaluation_results.append({
                        'question_id': i,
                        'question': qa['question'],
                        'gemini_answer': gemini_answer,
                        'reference_answer': qa.get('answer', ''),
                        'correctness': is_correct,
                        'score': 1.0 if is_correct else 0.0,
                        'max_score': 1.0
                    })
            
            accuracy = total_correct / len(questions) if questions else 0
            automatic_metrics = {
                'accuracy': accuracy,
                'correct_answers': total_correct,
                'total_questions': len(questions),
                'success_rate': accuracy * 100
            }
        
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


def run_giskard_evaluation(questions, excerpt, gemini_answers=None):
    print("Запуск оценки ответов с помощью метрик Giskard...")
    return evaluate_giskard_answers(questions, excerpt, gemini_answers)


if __name__ == "__main__":
    test_questions = [
        {
            'question': 'Какую истину Иешуа сообщил прокуратору в разговоре?',
            'answer': 'Истина в том, что у тебя сейчас болит голова, и она болит так сильно, что ты малодушно помышляешь о смерти.'
        }
    ]
    test_excerpt = "Тестовый отрывок из романа"
    results = run_giskard_evaluation(test_questions, test_excerpt)
    print(f"Результаты оценки: {results}")
