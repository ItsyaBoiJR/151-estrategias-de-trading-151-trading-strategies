import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple trading strategy using a neural network
class TradingStrategyNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TradingStrategyNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Generate dummy financial data
def generate_dummy_data(num_samples, input_size):
    np.random.seed(42)
    X = np.random.rand(num_samples, input_size)  # Random features
    y = (np.sum(X, axis=1) > input_size / 2).astype(np.float32)  # Binary labels
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Train the trading strategy model
def train_model(model, criterion, optimizer, X_train, y_train, epochs=100):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Test the trading strategy model
def test_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze()
        predictions = (predictions > 0.5).float()
        accuracy = (predictions == y_test).float().mean().item()
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    # Hyperparameters
    input_size = 10  # Number of features
    hidden_size = 16
    output_size = 1  # Binary classification
    learning_rate = 0.01
    epochs = 100
    num_samples = 1000

    # Generate dummy data
    X, y = generate_dummy_data(num_samples, input_size)
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Initialize model, loss function, and optimizer
    model = TradingStrategyNN(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train and test the model
    train_model(model, criterion, optimizer, X_train, y_train, epochs)
    test_model(model, X_test, y_test)