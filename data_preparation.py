import json
import google.generativeai as genai

def load_api_keys():
    with open('Key.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        return {
            'gemini_api_key': data.get('gemini_api_key'),
            'openai_api_key': data.get('openai_api_key')
        }

def initialize_text_model(api_keys):
    genai.configure(api_key=api_keys['gemini_api_key'])
    return genai.GenerativeModel('gemini-2.5-flash')

def get_excerpt(model):
    prompt = """Выбери значительный отрывок из романа "Мастер и Маргарита" Михаила Булгакова (примерно 500-800 слов). 
    Отрывок должен быть содержательным и подходящим для создания вопросов. 
    Верни только текст отрывка без дополнительных комментариев."""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Ошибка при получении отрывка: {e}")
        return None
