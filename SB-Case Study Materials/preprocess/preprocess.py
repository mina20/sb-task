import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import json

class DataPreprocessor:
    def __init__(self, config_path):
        self.config = json.load(open(config_path))
        self.path = self.config['path_for_rent_data']
        self.columns_to_convert = ['registration_date', 'contract_start_date', 'contract_end_date', 'req_from', 'req_to', 'meta_ts']

    def load_data(self):
        df = pd.read_csv(self.path)
        df.drop(columns=['project_name_en', 'project_name_ar', 'master_project_en', 'master_project_ar'], inplace=True)
        df = df[:1000]  
        return df

    @staticmethod
    def drop_ar_column(df):
        columns_to_drop = [col for col in df.columns if col.endswith('_ar')]
        df.drop(columns=columns_to_drop, inplace=True)
        return df

    @staticmethod
    def get_column_types(df):
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        return categorical_columns, numerical_columns

    @staticmethod
    def encode_categorical_variables(df):
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        encoder = OrdinalEncoder()
        encoded_df = df.copy()
        encoded_df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
        return encoded_df, encoder

    @staticmethod
    def convert_date_column(df, columns_to_convert):
        for col in columns_to_convert:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                print(f"Warning: Column '{col}' not found in DataFrame.")
        return df

    @staticmethod
    def handle_missing_data(df):
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

    @staticmethod
    def feature_engineering(df):
        df['room_density'] = df['property_size_sqm'] / df['rooms']
        if 'contract_start_date' in df.columns and 'contract_end_date' in df.columns:
            df['total_agreement_period'] = (df['contract_end_date'] - df['contract_start_date']).dt.days
        df.drop(columns=['contract_start_date', 'contract_end_date', 'property_size_sqm', 'rooms'], inplace=True)   
        return df

    @staticmethod
    def scale_numerical_features(df):
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        return df

    @staticmethod
    def preprocess_datetime_features(df):
        datetime_columns = ['req_to', 'meta_ts', 'registration_date', 'contract_start_date', 'contract_end_date']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        default_date = pd.to_datetime('2024-01-01')
        for col in datetime_columns:
            if col in df.columns:
                df[col].fillna(default_date, inplace=True)

        if 'req_to' in df.columns:
            df['req_to_year'] = df['req_to'].dt.year
            df['req_to_month'] = df['req_to'].dt.month
            df['req_to_day'] = df['req_to'].dt.day
            df['req_to_hour'] = df['req_to'].dt.hour
            df['req_to_weekday'] = df['req_to'].dt.weekday

        if 'meta_ts' in df.columns:
            df['meta_ts_year'] = df['meta_ts'].dt.year
            df['meta_ts_month'] = df['meta_ts'].dt.month
            df['meta_ts_day'] = df['meta_ts'].dt.day
            df['meta_ts_hour'] = df['meta_ts'].dt.hour
            df['meta_ts_weekday'] = df['meta_ts'].dt.weekday

        if 'registration_date' in df.columns:
            df['registration_date_year'] = df['registration_date'].dt.year
            df['registration_date_month'] = df['registration_date'].dt.month
            df['registration_date_day'] = df['registration_date'].dt.day
            df['registration_date_hour'] = df['registration_date'].dt.hour
            df['registration_date_weekday'] = df['registration_date'].dt.weekday

        if 'contract_start_date' in df.columns:
            df['contract_start_year'] = df['contract_start_date'].dt.year
            df['contract_start_month'] = df['contract_start_date'].dt.month
            df['contract_start_day'] = df['contract_start_date'].dt.day
            df['contract_start_hour'] = df['contract_start_date'].dt.hour
            df['contract_start_weekday'] = df['contract_start_date'].dt.weekday

        if 'contract_end_date' in df.columns:
            df['contract_end_year'] = df['contract_end_date'].dt.year
            df['contract_end_month'] = df['contract_end_date'].dt.month
            df['contract_end_day'] = df['contract_end_date'].dt.day
            df['contract_end_hour'] = df['contract_end_date'].dt.hour
            df['contract_end_weekday'] = df['contract_end_date'].dt.weekday

        if 'req_to' in df.columns and 'registration_date' in df.columns:
            df['contract_duration_days'] = (df['req_to'] - df['registration_date']).dt.days

        df.drop(columns=['req_to', 'req_from', 'meta_ts', 'registration_date'], inplace=True, errors='ignore')

        return df

    def preprocess_data(self):
        df = self.load_data()
        print("Dropping arabic columns...")
        df = self.drop_ar_column(df)
        print("Converting date columns to datetime format...")
        df = self.preprocess_datetime_features(df)
        print("Handling missing values...")
        df = self.handle_missing_data(df)
        print("Encoding categorical values...")
        df, encoder = self.encode_categorical_variables(df)
        print("Feature engineering...")
        df = self.feature_engineering(df)
        print("Handling missing values again after feature engineering...")
        df = self.handle_missing_data(df)
        print("Data processing is done, moving to next stage...")
        return df, encoder

if __name__ == "__main__":
    preprocessor = DataPreprocessor(config_path="./../config/config.json")
    df, encoder = preprocessor.preprocess_data()
    df.to_csv("./../data/pro1.csv")
