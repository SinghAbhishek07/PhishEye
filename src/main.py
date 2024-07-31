import argparse

try:
    from read_dataset import read_raw_data, train_dataset
    from training import train_baselines, train_llms
except ImportError:
    from src.read_dataset import read_raw_data, train_dataset
    from src.training import train_baselines, train_llms

def main(training_type):
    # Reading raw data
    raw_file_path = 'dataset/emails.csv'
    raw_df = read_raw_data(raw_file_path)
    print(raw_df.head())

    # Preprocessing and splitting the dataset
    (trained_dataset, test_dataset), tokenizer, class_weights = train_dataset(raw_df, label_col_name="spam", text_col_name="text", train_size=0.8)

    # Printing datasets for verification
    print("Training Dataset:")
    print(trained_dataset)
    print("Test Dataset:")
    print(test_dataset)

    if training_type == "LLM":
        # Training LLM (Used: DistilBERT)
        train_llms(trained_dataset, test_dataset, tokenizer, class_weights, label_col_name="spam", text_col_name="text")
    else:
        # Training baseline models
        train_baselines(trained_dataset, test_dataset, label_col_name="spam", text_col_name="text")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models based on the specified type.")
    parser.add_argument('--model_type', choices=['LLM', 'baseline'], required=True, help="Type of model to train: 'LLM' or 'baseline'")
    args = parser.parse_args()

    main(args.model_type)