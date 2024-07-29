import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# Dictionary to map scoring metrics to their corresponding functions
SCORING = {
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
    "accuracy": accuracy_score,
}

def save_scores(experiment: str, index: str, values: dict) -> None:
    print(f"Saving score for experiment : {experiment}")
    """Log scores are saved in the corresponding csv file for individual model"""

# List of Large Language Model
    llms = [
        "DistilBERT",
    ]

# List of traditional machine learning models    
    models = ["NaiveBayes", "LogisticRegression", "RandomForest", "K-NearestNeighbours",
               "SupportVectorMachine", "GradientBoosting", "XGBoost", "DecisionTree"]

# Making sure that the directory exists
    Path(f"outputs/scores/").mkdir(parents=True, exist_ok=True)

# Creating the file path for the experiment's scores
    file = Path(f"outputs/scores/{experiment}.csv")

    if file.is_file():      # If the file exists, read it and update the scores for the given model
        scores = pd.read_csv(f"outputs/scores/{experiment}.csv", index_col=0)    
        scores.loc[index] = values
    else:                    # If the file does not exist, creating a new DataFrame with appropriate columns
        if index in llms:
            scores = pd.DataFrame(
                index=llms,
                columns=list(SCORING.keys()) + ["training_time", "inference_time"],
            )
        else:
            scores = pd.DataFrame(
                index=models,
                columns=list(SCORING.keys()) + ["training_time", "inference_time"],
            )
        scores.loc[index] = values

# Finally saving the updated scores to the CSV file
    scores.to_csv(f"outputs/scores/{experiment}.csv")