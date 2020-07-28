########################################################################################################################
#
# Two environments simple GAN
#
# Goal: small proof of concept. See a simple GAN structure. The generator network is a simple 1 dimensional function
#
########################################################################################################################

########################################################################################################################
# Preamble
########################################################################################################################
import torch
import matplotlib.pyplot as plt
import numpy as np

########################################################################################################################
# Parameter Initialization
########################################################################################################################
# Do you want information of the process being printed?
print_info = False
# N is sample size
N = 1000
# D_in is input dimension
D_in = 1
# H is hidden dimension
H = 2
# D_out is output dimension
D_out = 1
# Optimization parameters
steps_d = 1
steps_init_d = 200
steps_g = 1
steps_both = 500
learning_rate_d = 1e-2
learning_rate_g = 1e-2
# Initialize parameter to optimize
t = torch.tensor([2.])
t.requires_grad_()

########################################################################################################################
# Dataset Generation
########################################################################################################################
mean = [0, 2]
sigma = [1, 1]
sample_1 = torch.randn(N, D_in)
sample_2 = torch.randn(N, D_in)
# Build initial training data
x_0 = mean[0] + sigma[0] * sample_1
x_1 = t + sigma[1] * sample_2
y_0 = torch.tensor([[1.]*N]).transpose(0,1)
y_1 = torch.tensor([[0.]*N]).transpose(0,1)
# Plotting parameters: coverage serves as a margin, ran sets x axis limits
margin = 3
ran = [min(mean) - margin, max(mean) + margin]

########################################################################################################################
# Setup for training
########################################################################################################################
discriminator = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ELU(),
    torch.nn.Linear(H, H),
    torch.nn.ELU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Sigmoid()
)
# Optimizers
loss_fn_d = torch.nn.MSELoss(reduction='sum')
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d)
optimizer_g = torch.optim.SGD([t], learning_rate_g)

########################################################################################################################
# Training
########################################################################################################################
# Give the discriminator a head start
for i in range(steps_init_d + 1):
    # x_0, x_1 inherited form generator loop
    # build inputs and outputs that the discriminator should look at
    y_pred_0 = discriminator(x_0)
    y_pred_1 = discriminator(x_1)
    # Compute and print discriminator loss.
    loss_d = loss_fn_d(y_pred_0, y_0) + loss_fn_d(y_pred_1, y_1)
    if i % 100 == 0 and i != 0 and print_info:
        print('Mean loss of discriminator after ', i, 'steps: ', loss_d.item() / (2 * N))
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()

for j in range(steps_both + 1):
    # Build discriminator
    for i in range(steps_d + 1):
        # x_0, x_1 inherited form generator loop
        # build inputs and outputs that the discriminator should look at
        y_pred_0 = discriminator(x_0)
        y_pred_1 = discriminator(x_1)
        # Compute and print discriminator loss.
        loss_d = loss_fn_d(y_pred_0, y_0) + loss_fn_d(y_pred_1,y_1)
        if i % 100 == 0 and i != 0 and print_info:
            print('Mean loss of discriminator after ', i, 'steps: ', loss_d.item()/(2*N))
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()
    # Only for the first step, plot the data with the discriminator
    if j == 0:
        x_fun = torch.zeros(N, 1)
        x_fun[:, 0] = torch.tensor(np.linspace(ran[0], ran[1], N))
        y_fun = discriminator(x_fun).detach().numpy()[:, 0]
        x_fun = x_fun.numpy()[:, 0]
        plt.plot(x_0, np.linspace(0, 1, N), marker='o', markersize=4, linestyle='', color=(0.5, 0.5, 1, 0.3), label='e=0')
        plt.plot(x_1.detach(), np.linspace(0, 1, N), marker='o', markersize=4, linestyle='', color=(1, 0.5, 0.5, 0.3), label='e=1')
        plt.plot(x_fun, y_fun, marker='', linewidth=6, linestyle='-', color=(1, 1, 1, 1))
        plt.plot(x_fun, y_fun, marker='', linewidth=2, linestyle='-', color=(0.75, 0.5, 0.75, 1), label='classification')
        plt.ylabel('discriminator output')
        plt.xlabel('realizations of $X^0\sim N$(' + str(mean[0]) + ', ' + str(sigma[0]) + '$^2$) and $X^1\sim N$(t, ' + str(sigma[1]) + '$^2$)')
        plt.title('Initial situation with t = ' + str(t.detach()[0].numpy()))
        plt.legend()
        plt.xlim(ran)
        plt.show()
    # Optimize generator with respect to the newly trained discriminator
    for i in range(steps_g + 1):
        # Compute the discrepancy that has been recognized, can extend to f-GAN
        mean_true = torch.mean(torch.log(discriminator(x_0)))
        mean_generated = torch.mean(torch.log(1-discriminator(x_1)))
        loss_g = (mean_generated + mean_true)
        # Print progress
        if j % 100 == 0 and j != 0 and print_info:
            print('Step ', j, ' loss = ', loss_g.detach().numpy(), ', t = ', t.detach()[0].numpy())
        # Optimize the parameter t
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()
        # Update the sample x_1 with the new version of t
        x_1 = t + sigma[1] * sample_2

########################################################################################################################
# Plotting
########################################################################################################################
x_fun = torch.zeros(N, 1)
x_fun[:, 0] = torch.tensor(np.linspace(ran[0], ran[1], N))
y_fun = discriminator(x_fun).detach().numpy()[:, 0]
x_fun = x_fun.numpy()[:, 0]
plt.plot(x_0, np.linspace(0,1,N), marker='o', markersize=4, linestyle='', color=(0.5, 0.5, 1, 0.3), label='e=0')
plt.plot(x_1.detach(), np.linspace(0,1,N), marker='o', markersize=4, linestyle='', color=(1, 0.5, 0.5, 0.3), label='e=1')
plt.plot(x_fun, y_fun, marker='', linewidth=6, linestyle='-', color=(1, 1, 1, 1))
plt.plot(x_fun, y_fun, marker='', linewidth=2, linestyle='-', color=(0.75, 0.5, 0.75, 1), label='classification')
plt.ylabel('discriminator output')
plt.xlabel('realizations of $X^0\sim N$('+str(mean[0])+', '+str(sigma[0])+'$^2$) and $X^1\sim N$(t, '+str(sigma[1])+'$^2$)')
plt.title('Final situation with t = '+str(t.detach()[0].numpy()))
plt.xlim(ran)
plt.show()
