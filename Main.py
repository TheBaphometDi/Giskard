from data_preparation import prepare_test_data
from testing import run_testing

def main():
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ GEMINI С GISCARD - ОБНОВЛЕННАЯ ВЕРСИЯ")
    print("=" * 60)
    print("Использует модульную архитектуру:")
    print("- data_preparation.py: подготовка данных")
    print("- testing.py: тестирование и оценка")
    print("=" * 60)
    run_testing()

if __name__ == "__main__":
    main()
