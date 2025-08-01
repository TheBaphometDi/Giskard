#!/usr/bin/env python3
"""
Скрипт для установки Giscard альтернативными способами
"""

import subprocess
import sys
import os

def install_giscard():
    print("Попытка установки Giscard...")
    
    # Способ 1: Из GitHub
    try:
        print("Способ 1: Установка из GitHub...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/giscard-ai/giscard.git"
        ])
        print("✅ Giscard успешно установлен из GitHub!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Не удалось установить из GitHub")
    
    # Способ 2: Из PyPI с альтернативным именем
    try:
        print("Способ 2: Поиск альтернативного пакета...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "giscard-ai"
        ])
        print("✅ Giscard успешно установлен как giscard-ai!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Не удалось найти giscard-ai")
    
    # Способ 3: Ручная установка зависимостей
    try:
        print("Способ 3: Установка базовых зависимостей...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "transformers", "torch", "datasets", "evaluate", "numpy", "scikit-learn"
        ])
        print("✅ Базовые зависимости установлены!")
        print("⚠️  Giscard не найден, но можно использовать альтернативы")
        return False
    except subprocess.CalledProcessError:
        print("❌ Не удалось установить зависимости")
        return False

if __name__ == "__main__":
    success = install_giscard()
    if success:
        print("\n🎉 Giscard готов к использованию!")
    else:
        print("\n⚠️  Giscard не установлен, но программа будет работать с Gemini") 