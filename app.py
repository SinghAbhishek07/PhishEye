from flask import Flask, request, render_template , send_from_directory
from src.inference import InferenceEngine
import pandas as pd
import os
app = Flask(__name__,static_folder='src')


@app.route('/figs/<path:filename>')
def custom_static(filename):
    return send_from_directory(os.path.join(app.static_folder, 'figs'), filename)

@app.route('/')
def index():
    """show a web page for user input"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """make an inference using either LLM or Best baseline model"""

    text = request.form.get('message') 

    # making inference here
    # pred = InferenceEngine(user_input=text).predict_with_llm()
    pred, best_model_name = InferenceEngine(user_input=text).predict_with_baseline()

    outcome = ""
    if pred == 0:
        outcome = " not"

    return render_template('prediction.html', message=text, pred=pred, is_phising=outcome, model_name=best_model_name)

if __name__ == '__main__':
    app.run(debug=True) 
    app.run(host="127.0.0.1", port=5000)