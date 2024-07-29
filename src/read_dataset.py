import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from datasets import Dataset, DatasetDict

try:
    from preprocessing import preprocess_data, get_class_weights
except ImportError:
    from src.preprocessing import preprocess_data, get_class_weights

def read_raw_data(file_path):
    print("Read RAW Data")
    emails_df = pd.read_csv(file_path)
    return emails_df

def has_no_val_dict(train_df, test_df):
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df),
        }
    )

def has_value_dict(train_df, test_df, val_df):
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "val": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df),
        }
    )


def train_dataset(raw_df, label_col_name="spam", text_col_name="text", train_size=0.8, max_length=128):
    print("Preprocessing and tokenizing data")
    trained_dataset, test_dataset, tokenizer = preprocess_data(
        raw_df,
        text_column=text_col_name,
        label_column=label_col_name,
        test_size=1-train_size,
        max_length=max_length
    )
    
    class_weights = get_class_weights(trained_dataset, label_column=label_col_name)
    
    print(f"Train size: {len(trained_dataset)}, Test size: {len(test_dataset)}")
    return (trained_dataset, test_dataset), tokenizer, class_weights