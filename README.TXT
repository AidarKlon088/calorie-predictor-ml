#  Calorie Prediction 

This project is a simple machine learning web application that predicts calories based on nutritional values.

##  Features

* Linear Regression model
* Predict calories from:

  * Protein
  * Fat
  * Carbohydrates
  * Sugar
* Web interface using Flask
* Model evaluation (MSE, R2)
* Data visualization (Actual vs Predicted graph)

##  Model

The model is trained using **Linear Regression** from scikit-learn.

## Project Structure

```
AI-project/
│
├── app.py
├── train.py
├── calorie_dataset.csv
├── linear_model.joblib
├── metrics.txt
├── requirements.txt
│
├── static/
│   └── plot.png
│
├── templates/
│   └── index.html
```

##  Installation

```bash
pip install -r requirements.txt
```

##  Run Project

1. Train model:

```bash
python train.py
```

2. Run web app:

```bash
python app.py
```

3. Open browser:

```
http://127.0.0.1:5000
```

##  Example Output

* Predicted calories
* Graph visualization
* Model metrics

##  Author
group:cs-21
Name:Nurbekov Aidar
