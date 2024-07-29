from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle
import datasets

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# Importing the custom preprocessing and data reading functions,and handle import errors if paths differ
try:
    from preprocessing import preprocess_data
    from read_dataset import read_raw_data, train_dataset
except ImportError:
    from src.preprocessing import preprocess_data
    from src.read_dataset import read_raw_data, train_dataset


# Defining the best baseline models with their respective parameters and paths
BEST_BASELINE_MODEL = {
    "NaiveBayes": (MultinomialNB(), 300, "src/outputs/model/NaiveBayes.pkl"),
    "LogisticRegression": (LogisticRegression(), 500, "src/outputs/model/LogisticRegression.pkl"),
    "RandomForest": (RandomForestClassifier(), 500, "src/outputs/model/RandomForest.pkl"),
    "K-NearestNeighbours": (KNeighborsClassifier(n_neighbors=5), 500, "src/outputs/model/K-NearestNeighbours.pkl"),
    "SupportVectorMachine": (SVC(kernel="sigmoid", gamma=1.0), 300, "src/outputs/model/SupportVectorMachine.pkl"),
    "GradientBoosting": (GradientBoostingClassifier(), 400, "src/outputs/model/GradientBoosting.pkl"),
    "XGBoost": (
        XGBClassifier(learning_rate=0.1, n_estimators=150), 300,
        "src/outputs/model/XGBoost.pkl",
    ),
    "DecisionTree": (DecisionTreeClassifier(), 500, "src/outputs/model/DecisionTree.pkl"),
}

# Defining the pretrained LLM used from hugging face
LLMS = {
    "DistilBERT": (
        AutoModelForSequenceClassification.from_pretrained(
            "src/outputs/model/distilbert-base-uncased"
        ),
        AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    ),
}

class InferenceEngine:
    def __init__(self, user_input: str, label_col_name='spam', text_col_name='text', dataset_name="dataset/emails.csv"):
        self.user_input = user_input
        self.dataset_name = dataset_name
        self.label_col_name = label_col_name
        self.text_col_name = text_col_name

    def process_input(self):
        """Convert text to matrix for baseline model only"""
        df_infer = pd.DataFrame(data={self.text_col_name: [self.user_input]})
        print(f"***Process input infer : {df_infer} and {self.text_col_name} and {self.label_col_name}")

        # Adding the 'spam' column with a placeholder value 
        df_infer[self.label_col_name] = 0  

        # Preprocess data
        processed_data, _, _ = preprocess_data(df_infer, self.text_col_name, self.label_col_name)
        return processed_data
    
    

    def predict_with_baseline(self):
        """Train (or load saved model) and predict"""
        # Reading and preprocess data
        raw_file_path = 'src/dataset/emails.csv'
        raw_df = read_raw_data(raw_file_path)
        (trained_dataset, test_dataset), tokenizer, class_weights = train_dataset(raw_df, label_col_name=self.label_col_name, text_col_name=self.text_col_name, train_size=0.8)
        train_df = trained_dataset.to_pandas()

        # Loading the best model based on previous scores("f1" taken into consideration to predict the best model)
        experiment = "baseline_experiment"
        df_score = pd.read_csv(f"src/outputs/scores/{experiment}.csv")
        best_model_name = df_score.iloc[df_score['f1'].idxmax()][0]
        print(f"Best line model : {best_model_name}")
        print(f"text_col_name : {self.text_col_name} and label_col_name : {self.label_col_name}")

        # Vectorizing the data
        vectorizer = TfidfVectorizer(max_features=BEST_BASELINE_MODEL[best_model_name][1])
        vectorizer.fit(train_df[self.text_col_name])

        # Loading the model
        model = pickle.load(open(BEST_BASELINE_MODEL[best_model_name][2], "rb"))

        # Processing user input
        df_infer = self.process_input()
        encoded_input = vectorizer.transform(df_infer[self.text_col_name])

        # returning the prediction and best_model_name
        prediction = model.predict(encoded_input)
        print(f"Prediction : {prediction}")
        return prediction[0] , best_model_name
    

    def predict_with_llm(self, model_name="DistilBERT"):
        """Load trained LLM model and make inference"""
        model, tokenizer = LLMS[model_name]

        # Tokenizing the input
        tokenized_input = tokenizer(
            self.user_input,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        # Returning the prediction
        output = model(tokenized_input["input_ids"])
        prediction = output.logits.argmax(axis=-1)

        return prediction.item()