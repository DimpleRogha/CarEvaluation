from flask import Flask, render_template, request
import joblib

model = joblib.load("car_model.joblib")
encoders = joblib.load("encoders.joblib")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = {
            'buying': request.form['buying'],
            'maint': request.form['maint'],
            'doors': request.form['doors'],
            'persons': request.form['persons'],
            'lug_boot': request.form['lug_boot'],
            'safety': request.form['safety']
        }

        input_encoded = [encoders[col].transform([user_input[col]])[0] for col in user_input]

        prediction = model.predict([input_encoded])[0]
        result = encoders['class'].inverse_transform([prediction])[0]

        return render_template('index.html', prediction_text=f'Predicted Car Evaluation: {result}')

if __name__ == "__main__":
    app.run(debug=True)
