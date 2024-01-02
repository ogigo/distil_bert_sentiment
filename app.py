from flask import Flask, render_template, request, jsonify
import torch
from inference import predict_hate_speech

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']

        result=predict_hate_speech(text)

        return render_template('index.html', text=text, sentiment=result)

if __name__ == '__main__':
    app.run(debug=True)
