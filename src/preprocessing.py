import re
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
from datasets import Dataset
from bs4 import BeautifulSoup
import unicodedata
import nltk

def init_nltoken():
    nltk.download("punkt")
    nltk.download('stopwords')

def clean_text(text):
    """Clean and normalize the text"""
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def preprocess_data(df, text_column, label_column, test_size=0.2, max_length=128):
    """Load, clean, and preprocess the dataset"""
    
    print(f"Inside preprocess dataset for {text_column} and {label_column}")
    df = df[[text_column, label_column]].dropna()
    
    # Clean text
    df[text_column] = df[text_column].apply(clean_text)
    
    # Convert labels to integers
    df[label_column] = df[label_column].astype(int)
    
    # Handle case where DataFrame has only one sample
    if len(df) == 1:
        train_df = df
        test_df = df
    else:
        # Split data
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[label_column], random_state=42)
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize_and_encode(examples):
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        )
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize and encode
    train_dataset = train_dataset.map(tokenize_and_encode, batched=True)
    test_dataset = test_dataset.map(tokenize_and_encode, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", label_column])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", label_column])
    
    return train_dataset, test_dataset, tokenizer

def get_class_weights(dataset, label_column="label"):
    """Calculate class weights for imbalanced datasets"""
    labels = dataset[label_column]
    # Convert tensor to pandas Series
    labels = pd.Series(labels.numpy())
    class_counts = labels.value_counts().sort_index()
    total = len(labels)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}

    print(class_weights)
    return class_weights