import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class TrafficForecastingModel(nn.Module):
    def __init__(self, input_seq_len, lstm_units, num_nodes):
        super(TrafficForecastingModel, self).__init__()
        self.lstm = nn.LSTM(input_seq_len, lstm_units, num_layers=1, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_units, num_nodes),
            nn.ReLU(),
            nn.Linear(num_nodes, num_nodes),
            nn.ReLU(),
            nn.Linear(num_nodes, 1),
        )

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.lstm.input_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)

        # Zero the initial cell state
        if self.lstm.bias is not None:
            c0[:, :, 0] = torch.tensor(1.0, device=x.device).unsqueeze(1)  # Set initial cell state to 1

        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc_layers(out[:, -1, :])
        return out

    def train(self, train_dataloader, criterion, optimizer, num_epochs, device):
        self.train()  # Set the model to training mode
        for epoch in range(num_epochs):
            total_loss = 0.0
            for inputs, targets in train_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, targets)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Accumulate loss
                total_loss += loss.item() * inputs.size(0)

            average_loss = total_loss / len(train_dataloader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")

    def predict(self, test_dataloader, device):
        self.eval()  # Set the model to evaluation mode
        predictions = []
        with torch.no_grad():
            for inputs in test_dataloader:
                inputs = inputs[0].to(device)
                outputs = self(inputs)
                predictions.extend(outputs.cpu().numpy())

        return predictions


# Read data from CSV
data_path = "/root/data.csv"
df = pd.read_csv(data_path)

# Drop any non-numeric columns
df_numeric = df.select_dtypes(include=[np.number])

# Extract features and labels
features = df_numeric.drop(columns=["Survived"])  # Assuming 'Survived' is the target column
labels = df_numeric["Survived"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Handle missing values by imputing with mean
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # Reshape for single output
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)  # Reshape for single output

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model, loss, and optimizer
input_seq_len = X_train.shape[1]  # Number of features
num_nodes = X_train.shape[0]  # Number of data points
lstm_units = 64
model = TrafficForecastingModel(input_seq_len, lstm_units, num_nodes)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
model.train(train_dataloader, criterion, optimizer, num_epochs, device)

# Make predictions
predictions = model.predict(test_dataloader, device)

# Evaluate the model
mse = mean_squared_error(y_test.values, predictions)
print(f"Test MSE: {mse:.4f}")
