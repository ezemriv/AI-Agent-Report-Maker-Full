# src/data/data_processing.py

import os
import json
import polars as pl

# Set up root paths
script_dir = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Define data paths
DATA_PATH = os.path.join(ROOT, 'data', 'raw', 'transactions_data.csv')
CLIENT_DATA_PATH = os.path.join(ROOT, 'data', 'processed', 'users_data.csv')
LABELS_PATH = os.path.join(ROOT, 'data', 'raw', 'train_fraud_labels.json')
PREDICTION_IDS_PATH = os.path.join(ROOT, 'predictions', 'predictions_3.json')

# Define output paths
TRAIN_PATH = os.path.join(ROOT, 'data', 'processed', 'train.parquet')
TEST_PATH = os.path.join(ROOT, 'data', 'processed', 'test.parquet')

class FraudPreProcessor:
    def __init__(self):
        pass

    def pl_load_transactions_data(self, file_path, nrows=None):
        df = pl.read_csv(file_path, n_rows=nrows, try_parse_dates=True)
        print("Processing transactions data...")

        # Clean up 'amount' by stripping '$' and converting to float
        df = df.with_columns(
            pl.col('amount').str.strip_chars("$").cast(pl.Float64).alias('amount_clean')
        )
        
        # # Drop unnecessary columns
        # to_drop = ['errors', 'amount']
        # df = df.drop(to_drop)

        # For forecasting task I only need this
        selected = ['client_id', 'amount_clean','date']
        df = df.select(selected)

        # # Encode 'use_chip' as 1 if 'Swipe Transaction' else 0
        # df = df.with_columns(
        #     (pl.col('use_chip') == 'Swipe Transaction').cast(pl.Int8).alias('use_chip')
        # )

        # # Cast specified columns to categorical
        cat_cols = ['client_id', 'card_id', 'use_chip', 'merchant_city', 'merchant_state', 'mcc', 'merchant_id', 'zip']
        # df = df.with_columns([pl.col(col).cast(pl.String).cast(pl.Categorical) for col in cat_cols])

        # # Fill nulls with "unknown"
        # df = df.with_columns([pl.col(col).fill_null("unknown") for col in ['merchant_state', 'zip']])

        return df, cat_cols

    def load_fraud_labels(self, path):
        with open(path, 'r') as f:
            fraud_labels = json.load(f)

        fraud_dict = fraud_labels.get('target', {})

        fraud_labels_df = pl.DataFrame({
            'id': list(fraud_dict.keys()),
            'fraud': [1 if v == 'Yes' else 0 for v in fraud_dict.values()]
        })

        fraud_labels_df = fraud_labels_df.with_columns(
            pl.col('id').cast(pl.Int64)
        )

        return fraud_labels_df

    def load_prediction_labels(self, path):
        with open(path, 'r') as f:
            pred_labels = json.load(f)

        fraud_dict = pred_labels.get('target', {})

        pred_labels_df = pl.DataFrame({
            'id': list(fraud_dict.keys()),
            'fraud': [1 if v == 'Yes' else 0 for v in fraud_dict.values()]
        })

        pred_labels_df = pred_labels_df.with_columns(
            pl.col('id').cast(pl.Int64)
        )

        return pred_labels_df

    def train_test_split_data(self, data, fraud_labels_df, pred_labels_df):
        test = pred_labels_df.join(data, on='id', how='left')
        train = fraud_labels_df.join(data, on='id', how='left')

        print(f"Train data shape: {train.shape}")
        print(f"Test data shape: {test.shape}")
        print(f"% target distribution in train data:\n{train['fraud'].value_counts(normalize=True)}")

        print("*" * 50)
        print("DATASET IS HIGHLY IMBALANCED")
        print("*" * 50)
        
        return train, test
    
    def save_as_parquet(self, df, file_path):
        df.write_parquet(file_path)
        print(f"Data saved to {file_path}")


def process_fraud_data():
    # Initialize the processor
    processor = FraudPreProcessor()

    # Load transactions data
    transactions_data, cat_cols = processor.pl_load_transactions_data(DATA_PATH)

    # Load fraud and prediction labels
    fraud_labels_df = processor.load_fraud_labels(LABELS_PATH)
    pred_labels_df = processor.load_prediction_labels(PREDICTION_IDS_PATH)

    # Split data into train and test
    train, test = processor.train_test_split_data(transactions_data, fraud_labels_df, pred_labels_df)

    # Save as Parquet files
    processor.save_as_parquet(train, TRAIN_PATH)
    processor.save_as_parquet(test, TEST_PATH)

    print(f"Train data saved to {TRAIN_PATH}")
    print(f"Test data saved to {TEST_PATH}")

    return train, test