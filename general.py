import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

# Generate synthetic data with missing values
data = torch.randn(1000, 5)
data_with_nans = data.clone()
data_with_nans[torch.randperm(data.shape[0])[:200], 2] = torch.nan  # Introduce missing values

# Standardize the data
scaler = StandardScaler()
data_with_nans_scaled = scaler.fit_transform(data_with_nans)

# Create PyTorch DataLoader
dataset = TensorDataset(torch.Tensor(data_with_nans_scaled))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define a simple Neural ODE model for data imputation
class Imputer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Imputer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, t, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model and define optimizer
input_dim = data.shape[1]
hidden_dim = 32
model = Imputer(input_dim, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        t = torch.linspace(0, 1, len(batch[0]))
        imputed_batch = odeint(model, batch[0], t)
        loss = nn.MSELoss()(imputed_batch[-1], batch[0])  # Use the final imputed values for loss calculation
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Impute missing values in the test data
data_test = torch.randn(200, 5)
data_test_with_nans = data_test.clone()
data_test_with_nans[torch.randperm(data_test.shape[0])[:50], 2] = torch.nan  # Introduce missing values
data_test_with_nans_scaled = scaler.transform(data_test_with_nans)
imputed_data_test = odeint(model, torch.Tensor(data_test_with_nans_scaled), t)[-1].detach().numpy()

# Compare the imputed values with the ground truth
print('Imputed data:')
print(imputed_data_test[:10])  # Display the first 10 imputed samples
print('Ground truth:')
print(data_test_with_nans[:10])  # Display the first 10 ground truth samples
