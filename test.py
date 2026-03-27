import joblib

model = joblib.load('linear_model.joblib')
print("Модель загружена!")

def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Ошибка! Введите число, а не текст.")

print("\nВведите данные для расчета калорий:")

protein = get_float_input("Белок (г): ")
fat = get_float_input("Жиры (г): ")
carbs = get_float_input("Углеводы (г): ")
sugar = get_float_input("Сахар (г): ")

X_new = [[protein, fat, carbs, sugar]]
predicted_calories = model.predict(X_new)[0]

print(f"\nПредсказанные калории: {predicted_calories:.2f} ккал")