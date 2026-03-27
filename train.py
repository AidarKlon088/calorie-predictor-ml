import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

os.makedirs('static', exist_ok=True)

df = pd.read_csv('calorie_dataset.csv')

X = df[['protein_g', 'fat_g', 'carbs_g', 'sugar_g']]
y = df['calories']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

joblib.dump(model, 'linear_model.joblib')
print("Модель сохранена!")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)]
)
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Actual vs Predicted Calories')
plt.grid(True)
plt.savefig('static/plot.png')
plt.close()
with open('metrics.txt', 'w') as f:
    f.write(f"MSE: {mse}\nR2: {r2}")

print("График и метрики сохранены!")