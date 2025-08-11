import json
import google.generativeai as genai

def load_api_keys():
    """Загрузка API ключей из файла Key.json"""
    with open('Key.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        return {
            'gemini_api_key': data.get('gemini_api_key'),
            'openai_api_key': data.get('openai_api_key')
        }

def initialize_gemini(api_keys):
    """Инициализация модели Gemini"""
    genai.configure(api_key=api_keys['gemini_api_key'])
    return genai.GenerativeModel('gemini-2.5-flash')

def get_excerpt_by_gemini(gemini_model):
    """Получение отрывка из романа через Gemini - единственная функция Gemini в этом модуле"""
    prompt = """Выбери значительный отрывок из романа "Мастер и Маргарита" Михаила Булгакова (примерно 500-800 слов). 
    Отрывок должен быть содержательным и подходящим для создания вопросов. 
    Верни только текст отрывка без дополнительных комментариев."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Ошибка при получении отрывка: {e}")
        return None

if __name__ == "__main__":
    print("=" * 50)
    print("МОДУЛЬ ПОДГОТОВКИ ДАННЫХ")
    print("=" * 50)
    print("Этот модуль содержит только базовые функции:")
    print("1. load_api_keys() - загрузка API ключей")
    print("2. initialize_gemini() - инициализация Gemini")
    print("3. get_excerpt_by_gemini() - получение отрывка через Gemini")
    print("\nДля полного процесса используйте Main.py")
    
    # Тестовая функция
    try:
        api_keys = load_api_keys()
        if api_keys['gemini_api_key']:
            print(f"\n✅ API ключ Gemini загружен")
            gemini_model = initialize_gemini(api_keys)
            print("✅ Модель Gemini инициализирована")
        else:
            print("❌ API ключ Gemini не найден в Key.json")
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}") 