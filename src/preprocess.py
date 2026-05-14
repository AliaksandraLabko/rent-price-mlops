import pandas as pd
import numpy as np
import joblib
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from google.cloud import storage

def preprocess_data(project_id, bucket_name, input_file):
    # 1. Load data from GCS
    print(f"Reading data from gs://{bucket_name}/{input_file}")
    path = f"gs://{bucket_name}/{input_file}"
    df = pd.read_csv(path)

    # 2. Define Features
    # Identify column types automatically
    target = 'price' # Assuming 'price' is the target column
    numeric_features = df.select_dtypes(include=['int64', 'float64']).drop([target], axis=1).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # 3. Create Transformers
    # Numeric: Fill missing with Median + Scale (important for some models)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical: Fill missing with Most Frequent + OneHot Encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 4. Bundle transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 5. Split Data
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Fit and Transform
    print("Fitting preprocessor...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 7. Save locally first (then we upload)
    os.makedirs("data/processed", exist_ok=True)
    joblib.dump(preprocessor, "data/processed/preprocessor.joblib")
    
    # Save transformed data as CSVs for the Training Step
    # We combine X and y back together for the trainer
    train_df = pd.DataFrame(X_train_transformed)
    train_df['target'] = y_train.values
    test_df = pd.DataFrame(X_test_transformed)
    test_df['target'] = y_test.values

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    # 8. Upload to GCS
    upload_to_gcs(project_id, bucket_name, "data/processed/")

def upload_to_gcs(project_id, bucket_name, local_path):
    client = storage.Client(project=project_id)
    # We want to save to the 'processed-data' and 'model-artifacts' buckets
    processed_bucket = client.bucket(f"{project_id}-rent-mlops-processed-data")
    artifact_bucket = client.bucket(f"{project_id}-rent-mlops-model-artifacts")

    # Upload CSVs
    for file in ['train.csv', 'test.csv']:
        blob = processed_bucket.blob(file)
        blob.upload_from_filename(os.path.join(local_path, file))
        print(f"Uploaded {file} to {processed_bucket.name}")

    # Upload Preprocessor object
    blob_p = artifact_bucket.blob("preprocessor.joblib")
    blob_p.upload_from_filename(os.path.join(local_path, "preprocessor.joblib"))
    print(f"Uploaded preprocessor.joblib to {artifact_bucket.name}")

if __name__ == "__main__":
    # This allows us to run the script from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, default='arboreal-totem-495915-v1')
    parser.add_argument('--bucket_name', type=str, default='arboreal-totem-495915-v1-rent-mlops-raw-data')
    parser.add_argument('--input_file', type=str, default='apartments_rent_pl_2024_04-2024_06.csv')
    
    args = parser.parse_args()
    preprocess_data(args.project_id, args.bucket_name, args.input_file)