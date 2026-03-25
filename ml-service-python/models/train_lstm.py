import torch 
import torch.nn as nn
import numpy as np

from data_preprocessing import X_train, y_train, X_test, y_test


# Convert to tensor
X_train = torch.tensor(X_train, dtype=torch.float32 )
X_test  = torch.tensor(X_test, dtype=torch.float32 )
y_train = torch.tensor(y_train, dtype=torch.float32 )
y_test  = torch.tensor(y_test, dtype=torch.float32 )


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# LSTM Model
class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
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
hidden_size = 32
output_size = y_train.shape[1]


# Initialize model
model = LSTMModel(input_size, hidden_size, output_size)


# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training
num_epochs = 100

for epoch in range(num_epochs):

    model.train()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
model.eval()

with torch.no_grad():

    predictions = model(X_test)

    test_loss = criterion(predictions, y_test)

    print("Test Loss:", test_loss.item())
    
torch.save(model.state_dict(), "lstm_model.pth")
print("Model Saved")