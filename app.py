from flask import Flask, render_template, request
import joblib
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

model = joblib.load('linear_model.joblib')
df = pd.read_csv('calorie_dataset.csv')

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

        X_new = [[protein, fat, carbs, sugar]]
        prediction = model.predict(X_new)[0]

        plt.figure(figsize=(5,4))
        plt.bar(['Protein','Fat','Carbs','Sugar'], [protein, fat, carbs, sugar], color='orange')
        plt.ylabel('Граммы')
        plt.title('Состав введённой еды')
        plt.savefig('static/plot.png')  # сохраняем в static
        plt.close()  # закрываем график, чтобы Flask мог его отобразить

    samples = df.head(10).to_html(classes='table')

    return render_template(
        'index.html',
        prediction=prediction,
        samples=samples,
        metrics=metrics
    )

if __name__ == '__main__':
    app.run(debug=True)