import json
import pandas as pd
import giskard
import os
from giskard.rag import generate_testset, KnowledgeBase
from giskard.rag.question_generators import (
    simple_questions,
    complex_questions,
    situational_questions,
    double_questions,
    conversational_questions,
    distracting_questions
)
from datetime import datetime


def setup_giskard_openai():
    try:
        from data_preparation import load_api_keys
        api_keys = load_api_keys()
        if 'openai_api_key' in api_keys and api_keys['openai_api_key']:
            os.environ['OPENAI_API_KEY'] = api_keys['openai_api_key']
            giskard.llm.set_llm_api("openai")
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
            return True
        else:
            print("⚠️ FastEmbed недоступен, попробуем установить...")
            return False
    except Exception as e:
        print(f"⚠️ Предупреждение: Не удалось настроить эмбеддинги: {e}")
        print("💡 Установите зависимости: pip install fastembed")
        return False


def create_knowledge_base_from_text(text):
    chunks = []
    words = text.split()
    chunk_size = 80
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    if len(chunks) < 8:
        extended_chunks = chunks * 4
    elif len(chunks) < 15:
        extended_chunks = chunks * 2
    else:
        extended_chunks = chunks
    
    df = pd.DataFrame({
        'content': extended_chunks,
        'source': 'Мастер и Маргарита'
    })
    return KnowledgeBase(df)


def generate_questions_with_giskard(excerpt):
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ВОПРОСОВ ЧЕРЕЗ GISCARD")
    print("=" * 60)
    print("Настройка OpenAI для генерации вопросов...")
    if not setup_giskard_openai():
        print("❌ Не удалось настроить OpenAI. Проверьте наличие openai_api_key в Key.json")
        return []
    print("Настройка модели эмбеддингов...")
    if not setup_giskard_embeddings():
        print("❌ Не удалось настроить эмбеддинги. Попробуйте установить зависимости:")
        print("   pip install fastembed")
        return []
    print("Создание базы знаний...")
    knowledge_base = create_knowledge_base_from_text(excerpt)
    if hasattr(knowledge_base, 'documents'):
        print(f"Создано {len(knowledge_base.documents)} фрагментов текста")
    else:
        print("База знаний успешно создана.")
    
    print("\nГенерация тестового набора вопросов через Giskard...")
    try:
        testset = generate_testset(
            knowledge_base=knowledge_base,
            num_questions=20,
            language='ru',
            question_generators=[simple_questions, complex_questions],
            agent_description="Чат-бот, отвечающий на вопросы по роману 'Мастер и Маргарита' Михаила Булгакова"
        )
        if hasattr(testset, 'questions'):
            samples = testset.questions
        elif hasattr(testset, 'to_pandas'):
            samples = testset.to_pandas().to_dict('records')
        else:
            print("Не удалось получить вопросы из QATestset")
            return []
        print(f"Успешно сгенерировано {len(samples)} вопросов")
        questions_and_answers = []
        for sample in samples:
            q = sample['question'] if isinstance(sample, dict) else getattr(sample, 'question', None)
            a = sample.get('reference_answer') if isinstance(sample, dict) else getattr(sample, 'reference_answer',
                                                                                        None)
            if q:
                questions_and_answers.append({
                    'question': str(q),
                    'answer': str(a) if a else 'Ответ будет сгенерирован позже'
                })
        return questions_and_answers
    except Exception as e:
        print(f"❌ Ошибка генерации вопросов через Giskard: {e}")
        return []


def save_giskard_questions(questions, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"giskard_questions_{timestamp}.json"
    data = {
        "generator": "Giskard",
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(questions),
        "questions": questions
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filename


# Импорт функций для работы с Gemini только для получения отрывка
from data_preparation import load_api_keys, initialize_gemini, get_excerpt_by_gemini


def get_excerpt_from_gemini():
    """Получение отрывка из романа через Gemini - единственная функция Gemini в этом модуле"""
    try:
        api_keys = load_api_keys()
        gemini_model = initialize_gemini(api_keys)
        excerpt = get_excerpt_by_gemini(gemini_model)
        return excerpt
    except Exception as e:
        print(f"Ошибка при получении отрывка через Gemini: {e}")
        return None


def run_giskard_generation(return_data=False):
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ВОПРОСОВ ЧЕРЕЗ GISCARD")
    print("=" * 60)
    print("Получение отрывка из романа через Gemini...")
    excerpt = get_excerpt_from_gemini()
    if not excerpt:
        print("ОШИБКА: Не удалось получить отрывок через Gemini")
        return (None, None) if return_data else None
    print(f"✅ Отрывок получен через Gemini (длина: {len(excerpt)} символов)")
    
    print("\nГенерация вопросов через Giskard...")
    questions = generate_questions_with_giskard(excerpt)
    if not questions:
        print("ОШИБКА: Не удалось сгенерировать вопросы через Giskard")
        return (None, excerpt) if return_data else None
    
    filename = save_giskard_questions(questions)
    print(f"\n✅ Вопросы сгенерированы через Giskard и сохранены в файл: {filename}")
    print("\nПримеры сгенерированных вопросов:")
    print("-" * 40)
    for i, qa in enumerate(questions[:5]):
        print(f"{i + 1}. Вопрос: {qa['question']}")
        print(f"   Ответ: {qa['answer'][:100]}...")
        print()

    if return_data:
        return questions, excerpt
    return filename


if __name__ == "__main__":
    run_giskard_generation()
