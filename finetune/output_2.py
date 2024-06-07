import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformerRegressor, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
            	d_model=input_dim,
            	nhead=8
            	dim_feedforward=2048),
            num_layers=6,
        )
            	
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.transformer(x)
        x = torch.mean(x, dim=0)  # Global average pooling
        return self.fc_out(x)

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
data_path = "./root/data.csv"
df = pd.read_csv(data_path)

# Drop any non-numeric columns
df_numeric = df.select_dtypes(include=[np.number])

# Extract features and target
features = df_numeric.drop(columns=["target_col"])  # Assuming 'target_col' is the target column
target = df_numeric["target_col"]

# Handle missing values by imputing with mean
features.fillna(features.mean(), inplace=True)
target.fillna(target.mean(), inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # Reshape for single output
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)  # Reshape for single output

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model, loss, and optimizer
input_dim = X_train.shape[1]  # Number of features
output_dim = 1  # Single output for regression
model = TransformerRegressor(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
model.train(X_train_tensor.tensor(dtype=torch.float32).unsqueeze(1), y_train_tensor, criterion, optimizer, num_epochs=10, device=device)

# Make predictions
predictions = model.predict(torch.tensor(X_test_tensor.values, dtype=torch.float32).unsqueeze(1), device)

# Evaluate the model
mse = mean_squared_error(y_test.values, predictions)
print(f"Test MSE: {mse:.4f}")
