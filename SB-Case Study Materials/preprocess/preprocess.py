import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import json
def drop_ar_column(df):
    columns_to_drop = [col for col in df.columns if col.endswith('_ar')]
    df.drop(columns=columns_to_drop, inplace=True)
    return df

def get_column_types(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    return categorical_columns, numerical_columns


def encode_categorical_variables(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    encoder = OrdinalEncoder()

    encoded_df = df.copy()
    encoded_df[categorical_columns] = encoder.fit_transform(df[categorical_columns])

    return encoded_df, encoder
def convert_date_column(df, columns_to_convert):
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    return df

from sklearn.impute import SimpleImputer

def handle_missing_data(df):
    # Identify numeric columns that have missing values
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_columns = [col for col in numeric_columns if df[col].isnull().any() and not col.endswith('_id')]

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_columns = [col for col in categorical_columns if df[col].isnull().any() and not col.endswith('_id')]

    numeric_imputer = SimpleImputer(strategy='median')
    if numeric_columns:
        df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    if categorical_columns:
        df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns]).astype('object')
    return df


def feature_engineering(df):
    df['room_density'] = df['property_size_sqm'] / df['rooms']
    if 'contract_start_date' in df.columns and 'contract_end_date' in df.columns:
        df['total_agreement_period'] = (df['contract_end_date'] - df['contract_start_date']).dt.days
    df.drop(columns=['contract_start_date', 'contract_end_date','property_size_sqm', 'rooms'], inplace=True)   
    return df

def scale_numerical_features(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def preprocess_data(df, columns_to_convert):
    print("Dropping arabic column...")
    df = drop_ar_column(df)
    print("converting date column to date formate...")
    df = convert_date_column(df, columns_to_convert)
    print("Handling missing values...")
    df = handle_missing_data(df)
    print("Encoding catagorical values...")
    df, encoder = encode_categorical_variables(df)
    print("Scaling....")
    df = scale_numerical_features(df)
    print("Performing feature engineering...")
    df = feature_engineering(df)
    print("data processing is done moving to next stage...")
    return df, encoder

if __name__ == "__main__":

    config = json.load(open("./../config/config.json"))
    path = config['path_for_rent_data']
    columns_to_convert = ['registration_date', 'contract_start_date', 'contract_end_date','req_from','req_to','meta_ts']
    df = pd.read_csv(path)
    df.drop(columns=['project_name_en','project_name_ar','master_project_en','master_project_ar'], inplace=True)
    df = df[:1000]
    df, encoder = preprocess_data(df, columns_to_convert)
    df.to_csv("./../data/pro.csv")
    
    
