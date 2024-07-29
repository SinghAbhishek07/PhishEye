import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
)
import evaluate
import numpy as np
import time
import pickle
import copy
from datasets import Dataset
from read_dataset import read_raw_data, train_dataset
import os
import torch
import torch.nn as nn

# Import evaluation and preprocessing tools, and handle import errors if paths differ
try:
    from evaluation_tools import save_scores, SCORING  # Add this line
    from preprocessing import init_nltoken
except ImportError:
    from src.evaluation_tools import save_scores, SCORING  # Add this line
    from src.preprocessing import init_nltoken   

# Defining baseline models with their respective parameters
MODELS = {
    "NaiveBayes": (MultinomialNB(),300),
    "LogisticRegression": (LogisticRegression(),500),
    "RandomForest": (RandomForestClassifier(),500),
    "K-NearestNeighbours": (KNeighborsClassifier(n_neighbors=5),500),
    "SupportVectorMachine": (SVC(kernel="sigmoid", gamma=1.0),300),
    "GradientBoosting": (GradientBoostingClassifier(),400),  
    "XGBoost": (XGBClassifier(learning_rate=0.1, n_estimators=150),300),
    "DecisionTree": (DecisionTreeClassifier(),500),   
}

# Define Large Language Models (DistilBERT)
LLMS = {
    "DistilBERT": (
        AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2),
        AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    ),
}

def compute_metrics(y_pred):
#Computing metrics for evaluation
    logits, labels = y_pred
    predictions = np.argmax(logits, axis=-1)
    return evaluate.load("f1").compute(
        predictions=predictions, references=labels, average="macro"
    )

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
#Computing loss with class weights
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def get_trainer(model, dataset, class_weights_tensor):
#Creating a Trainer object with specified parameters
    training_args = TrainingArguments(
        output_dir="experiments",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        use_cpu=True
    )

    trainer = CustomTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics
    )
    return trainer



def predict(trainer, dataset):
#Predicting using the trained model
    return trainer.predict(dataset).predictions.argmax(axis=-1)

def train_llms(train_dataset, test_dataset, tokenizer, class_weights, label_col_name: str, text_col_name: str, seed=123, train_size=0.8, test_set="test"):
#Training and evaluating Large Language Model
    print("Training LLM")
    scores = pd.DataFrame(index=list(LLMS.keys()), columns=list(SCORING.keys()) + ["training_time", "inference_time"])

    experiment = f"llm_experiment_{train_size}_train_seed_{seed}"

    for model_name, (model, _) in LLMS.items():
        print(f"Training {model_name}")
        tokenized_dataset = tokenize(train_dataset, test_dataset, tokenizer, text_col_name)

        # Ensuring labels are included in the dataset
        tokenized_dataset["train"] = tokenized_dataset["train"].map(lambda x: {"labels": x[label_col_name]})
        tokenized_dataset["test"] = tokenized_dataset["test"].map(lambda x: {"labels": x[label_col_name]})

        class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float)

        trainer = get_trainer(model, tokenized_dataset, class_weights_tensor)

        start = time.time()
        trainer.train()
        end = time.time()

        # Saving the model
        os.makedirs("outputs/model", exist_ok=True)
        trainer.save_model("outputs/model/distilbert-base-uncased")

        scores.loc[model_name]["training_time"] = end - start
        print(f"Training time for {model_name}: {end - start} seconds")

        start = time.time()
        predictions = predict(trainer, tokenized_dataset[test_set])
        end = time.time()

        for score_name, score_fn in SCORING.items():
            scores.loc[model_name][score_name] = score_fn(test_dataset[label_col_name], predictions)

        scores.loc[model_name]["inference_time"] = end - start
        print(f"Inference time for {model_name}: {end - start} seconds")
        save_scores("llm_experiment", model_name, scores.loc[model_name].to_dict())
    
    print("Overall LLM Scores")
    # Displaying the final scores
    print(scores)    

def train_baselines(train_dataset, test_dataset, label_col_name: str, text_col_name: str):
    print("Training baseline models")
    scores = pd.DataFrame(index=list(MODELS.keys()), columns=list(SCORING.keys()) + ["training_time", "inference_time"])

    # Converting datasets to pandas DataFrames
    train_df = train_dataset.to_pandas()
    test_df = test_dataset.to_pandas()

    print(f"Train : {train_df.head()}")
    print(f"Test : {train_df.head()}")
    
    # Initialize tokenization
    init_nltoken()

    for model_name, (model, max_iter) in MODELS.items():
        print(f"Training {model_name}")
        vectorizer = TfidfVectorizer(max_features=max_iter)
        X_train = vectorizer.fit_transform(train_df[text_col_name])
        y_train = train_df[label_col_name]
        X_test = vectorizer.transform(test_df[text_col_name])
        y_test = test_df[label_col_name]

        # Training the model and record the training time
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        scores.loc[model_name]["training_time"] = end - start
        print(f"Training time for {model_name}: {end - start} seconds")

        # Predicting and record the inference time
        start = time.time()
        y_pred = model.predict(X_test)
        end = time.time()
        scores.loc[model_name]["inference_time"] = end - start
        print(f"Inference time for {model_name}: {end - start} seconds")

        # Calculating and record the evaluation scores
        print(f"Print scoring : {SCORING.items()}")
        for score_name, score_fn in SCORING.items():
            scores.loc[model_name][score_name] = score_fn(y_test, y_pred)

        save_pickle_models(model_name, model)

        # Saving scores
        save_scores("baseline_experiment", model_name, scores.loc[model_name].to_dict())

    print(scores)

#Save the trained model as a pickle file
def save_pickle_models(model_name, model):
    os.makedirs("outputs/model", exist_ok=True)
    pickle.dump(model, open(f"outputs/model/{model_name}.pkl", "wb"))
    print(f"Model {model_name} saved")

def tokenize(train_dataset, test_dataset, tokenizer, text_col_name):
    def tokenize_function(examples):
        return tokenizer(examples[text_col_name], padding="max_length", truncation=True)

    # Tokenizing and encoding
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    return {
        "train": tokenized_train_dataset,
        "test": tokenized_test_dataset
    }