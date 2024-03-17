import torch
import torch.nn as nn
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

# Define a simple Boltzmann machine model for data imputation
class BoltzmannMachine(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BoltzmannMachine, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.b = nn.Parameter(torch.randn(hidden_dim))
        self.c = nn.Parameter(torch.randn(input_dim))
    
    def forward(self, x):
        v = torch.matmul(x, self.W) + self.b
        h = torch.sigmoid(v)
        p = torch.matmul(h, self.W.t()) + self.c
        return p

# Instantiate the model and define optimizer
input_dim = data.shape[1]
hidden_dim = 32
model = BoltzmannMachine(input_dim, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        p = model(batch[0])
        loss = nn.MSELoss()(p, batch[0])
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Impute missing values in the test data
data_test = torch.randn(200, 5)
data_test_with_nans = data_test.clone()
data_test_with_nans[torch.randperm(data_test.shape[0])[:50], 2] = torch.nan  # Introduce missing values
data_test_with_nans_scaled = scaler.transform(data_test_with_nans)
imputed_data_test = model(torch.Tensor(data_test_with_nans_scaled)).detach().numpy()

# Compare the imputed values with the ground truth
print('Imputed data:')
print(imputed_data_test[:10])  # Display the first 10 imputed samples
print('Ground truth:')
print(data_test_with_nans[:10])  # Display the first 10 ground truth samples
