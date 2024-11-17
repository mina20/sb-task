import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

class ModelEvaluator:
    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred
        self.rmse_value = self.calculate_rmse()
        self.r2_value = self.calculate_r2()
        self.mae_value = self.calculate_mae()

    def calculate_rmse(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        return np.sqrt(mse)

    def calculate_r2(self):
        return r2_score(self.y_test, self.y_pred)

    def calculate_mae(self):
        return mean_absolute_error(self.y_test, self.y_pred)

    def plot_metrics_bar(self):
        """Plot a bar chart for RMSE, R², and MAE."""
        metrics = {
            'RMSE': self.rmse_value,
            'R²': self.r2_value,
            'MAE': self.mae_value
        }

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'orange', 'green'], edgecolor='black')
        plt.title('Comparison of RMSE, R², and MAE')
        plt.xlabel('Metrics')
        plt.ylabel('Metric Values')

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 4, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8, rotation=45)
        plt.show()

if __name__ == "__main__":
    # Generate random data for features and target
    np.random.seed(42)  # For reproducibility
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = X[:, 0] * 3 + X[:, 1] * 2 + np.random.randn(100) * 0.1  # Linear combination + noise

    # Convert to DataFrame for consistency with the example
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
    df['target'] = y

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2, random_state=42)

    # Train a sample model (RandomForestRegressor in this case)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Initialize the evaluator class
    evaluator = ModelEvaluator(y_test, y_pred)

    # Plot combined histogram for RMSE, R², and MAE
    evaluator.plot_metrics_bar()
