from flask import Flask, render_template, request
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

app = Flask(__name__)

model = joblib.load('linear_model.joblib')
df = pd.read_csv('calorie_dataset.csv')

X = df[['protein_g', 'fat_g', 'carbs_g', 'sugar_g']]
y = df['calories']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = model.predict(X_test)

with open('metrics.txt') as f:
    metrics = f.read()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        protein = float(request.form['protein'])
        fat = float(request.form['fat'])
        carbs = float(request.form['carbs'])
        sugar = float(request.form['sugar'])

     
        prediction = model.predict([[protein, fat, carbs, sugar]])[0]

        
        nutrients = ['Protein', 'Fat', 'Carbs', 'Sugar']
        values = [protein, fat, carbs, sugar]

        plt.figure(figsize=(6,4))
        

        plt.scatter(y_test, y_pred, alpha=0.7, color='red', label='Тестовые данные')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linestyle='--', label='Идеальная линия')

  
        
        plt.ylabel('Калории / Граммы')
        plt.title('Линейная регрессия и состав еды')
        plt.grid(True)
        plt.legend()
        plt.savefig('static/plot.png')
        plt.close()

    samples = df.head(10).to_html(classes='table table-striped')

    return render_template('index.html', prediction=prediction, samples=samples, metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)
