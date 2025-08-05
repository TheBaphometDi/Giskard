from testing import validate_data_for_giskard, calculate_accuracy_score

def test_validation():
    print("=" * 50)
    print("ТЕСТ ВАЛИДАЦИИ ДАННЫХ")
    print("=" * 50)
    test_reference_qa = [
        {
            "question": "Кто является главным героем романа?",
            "answer": "Мастер - главный герой романа, писатель, который создал роман о Понтии Пилате."
        },
        {
            "question": "Как зовут возлюбленную Мастера?",
            "answer": "Маргарита - возлюбленная Мастера, которая готова на все ради него."
        },
        {
            "question": "Кто такой Воланд?",
            "answer": "Воланд - это дьявол, который прибывает в Москву со своей свитой."
        }
    ]
    test_model_answers = [
        "Мастер - это главный герой романа, писатель, который написал роман о Понтии Пилате.",
        "Маргарита - это возлюбленная Мастера, которая готова пожертвовать всем ради него.",
        "Воланд - это дьявол, который приезжает в Москву со своей свитой для проведения бала."
    ]
    print("Тестовые данные:")
    for i, qa in enumerate(test_reference_qa):
        print(f"{i+1}. Вопрос: {qa['question']}")
        print(f"   Эталон: {qa['answer']}")
        print(f"   Модель: {test_model_answers[i]}")
        print()
    print("Тестирование валидации данных...")
    valid_data = validate_data_for_giskard(test_reference_qa, test_model_answers)
    print(f"Валидных записей: {len(valid_data)}")
    for i, data in enumerate(valid_data):
        print(f"  {i+1}. Вопрос: {data['question'][:50]}...")
        print(f"     Ответ: {data['model_answer'][:50]}...")
    print("\nТестирование расчета точности...")
    for i, qa in enumerate(test_reference_qa):
        accuracy = calculate_accuracy_score(
            test_model_answers[i], 
            qa["answer"], 
            qa["question"]
        )
        print(f"Точность ответа {i+1}: {accuracy:.3f}")
    print("\nТестирование с невалидными данными...")
    invalid_reference_qa = [
        {
            "question": "Короткий вопрос?",
            "answer": "Короткий ответ."
        },
        {
            "question": "Нормальный вопрос для тестирования?",
            "answer": "Нормальный ответ для тестирования."
        }
    ]
    invalid_model_answers = [
        "Не могу ответить на этот вопрос.",
        "Нормальный ответ от модели для тестирования."
    ]
    invalid_valid_data = validate_data_for_giskard(invalid_reference_qa, invalid_model_answers)
    print(f"Валидных записей из невалидных данных: {len(invalid_valid_data)}")
    print("\n" + "=" * 50)
    print("ТЕСТ ЗАВЕРШЕН")
    print("=" * 50)

if __name__ == "__main__":
    test_validation() 