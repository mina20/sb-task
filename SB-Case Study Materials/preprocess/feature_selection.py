import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
import seaborn as sns
import matplotlib.pyplot as plt

class FeatureSelector:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column

    def get_highly_correlated_pairs(self, threshold=0.9):
        corr_matrix = self.df.corr()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        correlated_pairs = [
            (col, row) 
            for col in upper_tri.columns 
            for row in upper_tri.index 
            if abs(upper_tri.at[row, col]) > threshold
        ]
        return correlated_pairs

    def feature_importance(self, k):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        model = RandomForestRegressor()
        model.fit(X, y)

        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        })

        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        top_features = feature_importance.head(k)['Feature'].tolist()
        print(f"Top {k} important features: {top_features}")
        return top_features

    def rfe_selection(self, num_features=10):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        model = RandomForestRegressor()
        rfe = RFE(estimator=model, n_features_to_select=num_features)
        rfe.fit(X, y)

        selected_features = X.columns[rfe.support_].tolist()
        print(f"Features selected by RFE: {selected_features}")
        return selected_features

    def univariate_selection(self, k=10):
        X = self.df.drop(columns=[self.target_column]) 
        y = self.df[self.target_column]  
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        selected_features = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        })
        selected_features.sort_values(by='Score', ascending=False)
        sorted_feature_names = selected_features['Feature'].tolist()
        return sorted_feature_names

    def combine_selected_features(self, correlation_threshold=0.9, num_feature_rf=10, num_features_rfe=10, k_univariate=10):
        important_features = self.feature_importance(num_feature_rf)
        rfe_features = self.rfe_selection(num_features_rfe)
        univariate_features = self.univariate_selection(k_univariate)
        selected_features = list(set(important_features + rfe_features + univariate_features))
        if self.target_column in selected_features:
            selected_features.remove(self.target_column)

        print(f"Combined selected features: {selected_features}")
        return self.df[selected_features + [self.target_column]]

if __name__ == "__main__":
    df = pd.read_csv("./../data/pro1.csv")
    target_column = "annual_amount"
    df.drop(columns=['parcel_id'], inplace=True)
    selector = FeatureSelector(df, target_column)
    correlated_columns = selector.get_highly_correlated_pairs(threshold=0.8)
    print(f"Correlated pairs: {correlated_columns}")
    columns_to_drop = ['contract_amount', 'is_freehold_text', 'total_properties']
    df.drop(columns=columns_to_drop, inplace=True)
    important_features = selector.feature_importance(10)
    rfe_features = selector.rfe_selection(num_features=10)
    uni_features = selector.univariate_selection(k=10)
    print(f"RFE features: {rfe_features}")
    print(f"Univariate features: {uni_features}")
    print(f"Important features: {important_features}")
    selected_df = selector.combine_selected_features(correlation_threshold=0.9, num_features_rfe=10, k_univariate=10)
    selected_df.to_csv("./../data/selected_features.csv", index=False)
    print(selected_df.head())
