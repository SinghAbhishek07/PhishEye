import argparse
from flask import Flask, request, render_template, send_from_directory
from src.inference import InferenceEngine
import pandas as pd
import os

app = Flask(__name__, static_folder='src')

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

    print(f"Code : {app.config.get('MODEL_TYPE')}")

    # making inference here
    if app.config.get('MODEL_TYPE') == 'LLM':
        pred = InferenceEngine(user_input=text).predict_with_llm()
        best_model_name = 'DistilBERT'
    else:
        pred, best_model_name = InferenceEngine(user_input=text).predict_with_baseline()

    outcome = ""
    if pred == 0:
        outcome = " not"

    return render_template('prediction.html', message=text, pred=pred, is_phising=outcome, model_name=best_model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Flask app with specified model type.")
    parser.add_argument('--model_type', choices=['LLM', 'baseline'], required=True, help="Type of model to use for prediction: 'LLM' or 'baseline'")
    args = parser.parse_args()
    
    if args.model_type not in ['LLM', 'baseline']:
        raise ValueError("model_type argument must be 'LLM' or 'baseline'")

    app.config['MODEL_TYPE'] = args.model_type

    app.run(host="0.0.0.0", port=5000, debug=True)