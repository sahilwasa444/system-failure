from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from data_preprocessing import X_train, y_train, X_test, y_test


MODEL_PATH = Path(__file__).with_name("lstm_model.pth")
VALIDATION_SPLIT = 0.1
PATIENCE = 15
MIN_DELTA = 1e-4


# Convert to tensor
X_train = torch.tensor(X_train, dtype=torch.float32 )
X_test  = torch.tensor(X_test, dtype=torch.float32 )
y_train = torch.tensor(y_train, dtype=torch.float32 )
y_test  = torch.tensor(y_test, dtype=torch.float32 )


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

validation_size = max(1, int(len(X_train) * VALIDATION_SPLIT))

if len(X_train) <= validation_size:
    raise ValueError("Not enough training samples to create a validation split for early stopping.")

train_inputs = X_train[:-validation_size]
train_targets = y_train[:-validation_size]
val_inputs = X_train[-validation_size:]
val_targets = y_train[-validation_size:]

print(train_inputs.shape, train_targets.shape)
print(val_inputs.shape, val_targets.shape)


# LSTM Model
class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]
        
        out = torch.relu(out)
        
        out = self.fc(out)

        return out



# Model parameters
input_size = X_train.shape[2]
hidden_size = 64
output_size = y_train.shape[1]


# Initialize model
model = LSTMModel(input_size, hidden_size, output_size)


# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)


# Training
num_epochs = 150
best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
best_model_state = deepcopy(model.state_dict())

for epoch in range(num_epochs):

    model.train()

    outputs = model(train_inputs)

    loss = criterion(outputs, train_targets)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model.eval()

    with torch.no_grad():
        val_predictions = model(val_inputs)
        val_loss = criterion(val_predictions, val_targets)

    if val_loss.item() < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss.item()
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        best_model_state = deepcopy(model.state_dict())
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {loss.item():.4f}, "
            f"Val Loss: {val_loss.item():.4f}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping triggered at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f}")
        break

model.load_state_dict(best_model_state)
print(f"Restored best model from epoch {best_epoch} with validation loss {best_val_loss:.4f}")
        
model.eval()

with torch.no_grad():

    predictions = model(X_test)

    test_loss = criterion(predictions, y_test)

    print("Test Loss:", test_loss.item())
    
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model Saved: {MODEL_PATH}")
