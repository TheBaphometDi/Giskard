import json
import os
import pandas as pd
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

#ручное создание бз ибо гискард криво парсирует вопросы и ответы
def create_knowledge_base_from_text(text: str) -> KnowledgeBase:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = paragraphs if len(paragraphs) >= 3 else []

    if not chunks:
        words = text.split()
        chunk_size = 120
        chunks = [' '.join(words[i:i + chunk_size]).strip() for i in range(0, len(words), chunk_size)]
        chunks = [c for c in chunks if c]

    if len(chunks) < 2 and text:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) >= 2:
            mid = len(lines) // 2
            chunks = [' '.join(lines[:mid]).strip(), ' '.join(lines[mid:]).strip()]
        else:
            mid = max(200, len(text) // 2)
            chunks = [text[:mid].strip(), text[mid:].strip()]

    df = pd.DataFrame({
        'id': [f'doc_{i}' for i in range(len(chunks))],
        'content': chunks,
        'source': ['Мастер и Маргарита'] * len(chunks)
    })
    return KnowledgeBase(df)


def generate_questions(excerpt):
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ВОПРОСОВ ЧЕРЕЗ GISCARD")
    print("=" * 60)
    print("Создание базы знаний (ручное разбиение и индексация)...")
    knowledge_base = create_knowledge_base_from_text(excerpt)
    if hasattr(knowledge_base, 'documents'):
        print(f"Создано {len(knowledge_base.documents)} фрагментов текста")
    else:
        print("База знаний успешно создана.")

    print("\nГенерация тестового набора вопросов...")
    try:
        if not os.environ.get('OPENAI_API_KEY'):
            try:
                from data_preparation import load_api_keys
                keys = load_api_keys()
                if keys.get('openai_api_key'):
                    os.environ['OPENAI_API_KEY'] = keys['openai_api_key']
            except Exception:
                pass
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


def save_questions(questions, filename=None):
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
from data_preparation import load_api_keys, initialize_text_model, get_excerpt


def fetch_excerpt():
    try:
        api_keys = load_api_keys()
        model = initialize_text_model(api_keys)
        excerpt = get_excerpt(model)
        return excerpt
    except Exception as e:
        print(f"Ошибка при получении отрывка: {e}")
        return None


def run_question_generation(return_data=False):
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ВОПРОСОВ ЧЕРЕЗ GISCARD")
    print("=" * 60)
    print("Получение отрывка из романа...")
    excerpt = fetch_excerpt()
    if not excerpt:
        print("ОШИБКА: Не удалось получить отрывок через Gemini")
        return (None, None) if return_data else None
    print(f"✅ Отрывок получен (длина: {len(excerpt)} символов)")

    print("\nГенерация вопросов...")
    questions = generate_questions(excerpt)
    if not questions:
        print("ОШИБКА: Не удалось сгенерировать вопросы через Giskard")
        return (None, excerpt) if return_data else None

    filename = save_questions(questions)
    print(f"\n✅ Вопросы сгенерированы и сохранены в файл: {filename}")
    print("\nПримеры сгенерированных вопросов:")
    print("-" * 40)
    for i, qa in enumerate(questions[:5]):
        print(f"{i + 1}. Вопрос: {qa['question']}")
        print(f"   Ответ: {qa['answer'][:100]}...")
        print()

    if return_data:
        return questions, excerpt
    return filename


