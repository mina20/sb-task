import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split

class MetaLearner(nn.Module):
    def __init__(self, input_dim):
        super(MetaLearner, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)

def train_meta_learner(X_meta, y_meta, input_dim, lr=0.001, epochs=100, batch_size=32):
    model = MetaLearner(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_meta.values, dtype=torch.float32),
                                              torch.tensor(y_meta.values, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

    return model

def evaluate_meta_learner(model, X_meta, y_meta):
    with torch.no_grad():
        X_tensor = torch.tensor(X_meta.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_meta.values, dtype=torch.float32)
        predictions = model(X_tensor).squeeze().numpy()
        # mse = mean_squared_error(y_meta, predictions)
    return predictions

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    print(f"Model loaded from {model_path}")
    return model

# df = pd.DataFrame({
#     'y_actual': [72000.00, 76867.00, 63525.00, 30000.00, 30800.00, 60000.00, 363000.00, 3000000.00, 63635.76, 667704.00],
#     'XGBoost_pred': [67755.05, 78223.24, 70823.31, 29309.81, 33652.46, 59167.86, 363000.00, 3000000.00, 113259.80, 667703.90],
#     'RandomForest_pred': [67500.25, 78000.30, 70500.11, 29000.00, 33000.22, 59000.00, 360000.00, 2990000.00, 113000.55, 666500.00],
#     'SVR_pred': [67000.30, 78050.00, 70550.60, 29200.00, 33200.50, 59050.30, 362000.00, 3001000.00, 113400.25, 667100.50]
# })

# X_meta = df[['XGBoost_pred', 'RandomForest_pred', 'SVR_pred']]

# y_meta = df['y_actual']

# input_dim = X_meta.shape[1]

# model = train_meta_learner(X_meta, y_meta, input_dim)

# save_model(model, "model_output/meta_learner.pth")

# loaded_model = MetaLearner(input_dim)
# load_model(loaded_model, "meta_learner.pth")

# predictions = evaluate_meta_learner(loaded_model, X_meta, y_meta)
# print(f"Mean Squared Error of Meta Learner: {mse}")
