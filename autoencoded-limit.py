import torch
import torch.nn as nn
from torchdiffeq import odeint

#dtdx​=−y+x(1−x2−y2)
#dydt=x+y(1−x2−y2)dtdy​=x+y(1−x2−y2)

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),  # Input dimension is 2 (x, y)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Output dimension is 2 (x, y)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define the neural ODE equation
class NeuralODE(nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__()
        self.encoder = Autoencoder()
        self.ode_func = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
        )

    def forward(self, x, t):
        encoded = self.encoder.encoder(x)
        dydt = self.ode_func(encoded)
        return dydt

# Initialize the Neural ODE model
model = NeuralODE()

# Define the time points for solving the ODE
t = torch.linspace(0, 10, 100)  # Time from 0 to 10 with 100 points

# Define the initial condition for the ODE system
x0 = torch.tensor([1.0, 1.0], requires_grad=True)

# Solve the ODE using Neural ODE
solution = odeint(model, x0, t)

# Print the solution trajectory
print(solution)
