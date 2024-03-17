# Define the generator network
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, t, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Instantiate the generator and discriminator
generator = Generator(input_dim, hidden_dim)
discriminator = Discriminator(input_dim)

# Define optimizers for the generator and discriminator
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=1e-3)
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

# Adversarial training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch in dataloader:
        # Train the generator
        optimizer_generator.zero_grad()
        t = torch.linspace(0, 1, len(batch[0]))
        imputed_batch = odeint(generator, batch[0], t)
        loss_generator = nn.BCELoss()(discriminator(imputed_batch[-1]), torch.ones_like(batch[0]))
        loss_generator.backward()
        optimizer_generator.step()

        # Train the discriminator
        optimizer_discriminator.zero_grad()
        loss_discriminator_real = nn.BCELoss()(discriminator(batch[0]), torch.ones_like(batch[0]))
        loss_discriminator_fake = nn.BCELoss()(discriminator(imputed_batch[-1]), torch.zeros_like(batch[0]))
        loss_discriminator = loss_discriminator_real + loss_discriminator_fake
        loss_discriminator.backward()
        optimizer_discriminator.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss Generator: {loss_generator.item()}, Loss Discriminator: {loss_discriminator.item()}')

# Impute missing values in the test data with the adversarial generator
data_test_imputed = odeint(generator, torch.Tensor(data_test_with_nans_scaled), t)[-1].detach().numpy()

# Compare the imputed values with the ground truth
print('Imputed data:')
print(data_test_imputed[:10])  # Display the first 10 imputed samples
print('Ground truth:')
print(data_test_with_nans[:10])  # Display the first 10 ground truth samples
