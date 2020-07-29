########################################################################################################################
#
# Multi-class GAN
#
# Goal: test the multi-class algorithm on the synthetic dataset
#
########################################################################################################################

########################################################################################################################
# Preamble
########################################################################################################################
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import copy
from sklearn.linear_model import LinearRegression

########################################################################################################################
# Initialize parameters
########################################################################################################################
# Do you want information of the process being printed and/or plotted?
plot_discriminator_progress = True
plot_all_environments_true = True
plot_all_environments_reg = True
plot_all_environments_GAN = True
B_completely_random = False
# Choose: deterministic, old_random, new_random
dataset_generation = 'new_random'
# Choose: exp, mixture, normal
noise_type = 'exp'
# N: sample size per environment
N = 3000
# E: number of environments
E = 6
# Dimension of X
d = 5
# Number of causal parameters
m = 3
# Discriminator network parameters
D_in = 1
H = 100
D_out = E
# Optimization parameters
passes = 1000
steps_both = 1
steps_d = 1
steps_d_warmstart = 10
steps_g = 1
regression_kickstart = True
learning_rate_d = 1e-4
learning_rate_g = 1e-4
# For reproducibility
torch.manual_seed(1)
random.seed(1)
# Plotting parameters: coverage serves as a margin, ran: x axis limits
margin = 0.5  # 0.5
ran = [-margin, margin]
# Standard color palette
colors_transparent = [(1, 0.5, 0.5, 0.3), (0.5, 0.5, 1, 0.3)]
colors = [(1, 0.5, 0.5, 1), (0.5, 0.5, 1, 1)]
# Print model parameters
print('Model parameters: \nSample size: \t\t\t\t', N, '\nDimension of X: \t\t\t', d, '\nNumber of environments: \t', E,
      '\nChosen type of noise:\t\t', noise_type)

########################################################################################################################
# Generate dataset
########################################################################################################################
if dataset_generation == 'deterministic':
    # # Generate dataset
    # True causal parameter
    true_t = torch.tensor([[1.]] * m + [[0.]] * (d - m))
    X = torch.randn(E, N, d)
    # Perturb data in other environments
    X[1:E, :, :] = X[1:E, :, :] * 2
    Y = torch.matmul(X, true_t) + 0.1 * torch.randn(E, N, 1)
    # Perturb last column s.t. structural graph has Y -> X_d, but vary perturbation in at least one environment, e = E
    X[:, :, d - 1] = Y[:, :, 0] + 0.1 * torch.cat([torch.randn(E - 1, N), 3 * torch.randn(1, N)], 0)
elif dataset_generation == 'old_random':
    # True causal parameter, \theta^\star
    true_t = torch.tensor([[1.]]*m + [[0.]]*(d-m))
    b_2 = true_t.view(1,-1)
    # X_i that are children of Y, hence Y -> X_i
    b_1 = torch.tensor([[0.]]*(d-1) + [[1.]])
    # Matrix B_x responsible for X_i -> X_j interactions
    B_x = torch.randn(d, d)  # can also opt for torch.zeros(d,d)
    # Combine together in one single matrix B
    B = torch.cat([torch.cat([B_x, b_1], 1), torch.cat([b_2, torch.zeros(1, 1)], 1)], 0)
    # Generate noise vector
    noise = torch.randn(E, N, d+1, 1)
    #noise intervention in all environments
    #for i in range(1, min(E, d+1)):
    #    noise[i, :, i-1, 0] = 2<- random number *torch.randn(N)
    noise[1:E, :, 0:d-1, 0] += 3 #* torch.randn(E-1, N, d-1)
    noise[:, :, d-1, 0] = 0.1 * torch.randn(E, N)
    noise[E-1, :, d-1, 0] = 0.3 * torch.randn(N)
    # Noise on Y
    exponential_distr = torch.distributions.exponential.Exponential(1)
    if noise_type == 'mixture':
        mixture_label = torch.bernoulli(torch.tensor([[0.5]*N]*E))
        noise[:, :, d, 0] = 0.1*(torch.mul(mixture_label, 0.5*torch.randn(E, N)-1) + torch.mul(1-mixture_label, 0.5*torch.randn(E, N)+1))
    elif noise_type == 'exp':
        noise[:, :, d, 0] = 0.1 * (exponential_distr.sample([E, N])-1)
    elif noise_type == 'normal':
        noise[:, :, d, 0] = 0.1 * torch.randn(E, N)
    else:
        print('Error: insert a valid noise distribution type')
        quit()
    # Alternative: totally random matrix B (less easy to verify visually)
    if B_completely_random:
        B = torch.randn(d+1,d+1)
        B[-1, -1] = 0
        true_t[:, 0] = B[-1, 0:-1]
    # check invertibility
    if torch.det(torch.eye(d+1) - B) == 0:
        print('Error: the give configuration does not yield a directed acyclic graph.\nTry again')
        quit()
    print(B)
    # Generate data
    Z = torch.matmul(torch.inverse(torch.eye(d+1) - B), noise)
    X = Z[:, :, 0:d, 0]
    Y = Z[:, :, d, :]
elif dataset_generation == 'new_random':
    # True causal parameter, \theta^\star
    true_t = torch.tensor([[1.]] * m + [[0.]] * (d - m))
    b_2 = true_t.view(1, -1)
    # X_i that are children of Y, hence Y -> X_i
    b_1 = torch.tensor([[0.]] * (d - 1) + [[1.]])
    # Generate upper triangular matrix
    B = torch.zeros(d + 1, d + 1)
    for i in range(d + 1):
        for j in range(i):
            B[i, j] = torch.randint(-1, 1, [1])
    if not B_completely_random:
        B[-1, 0:-1] = b_2
        B[-2, :] = torch.tensor([[0] * d + [1]])
    else:
        B[-1, -2] = 0
        B[-2, :] = torch.zeros(1, d + 1)
        B[-2, -1] = torch.randn(1)
        true_t[:, 0] = B[-1, 0:-1]
    print(B)
    # Generate noise vector
    noise = torch.randn(E, N, d + 1, 1)
    # noise intervention in all environments X_1, ..., X_{d-1}
    for i in range(1, min(E, d-1)):
        noise[i, :, i-1, 0] = 2*torch.randn(N) #2<- random number *torch.randn(N)
    #noise[1:E, :, 0:d - 1, 0] = 2 * torch.randn(E - 1, N, d - 1)
    # noise intervention on X_d
    noise[:, :, d - 1, 0] = 0.1 * torch.randn(E, N)
    noise[E - 1, :, d - 1, 0] = 0.3 * torch.randn(N)
    # Noise on Y
    exponential_distr = torch.distributions.exponential.Exponential(1)
    if noise_type == 'mixture':
        mixture_label = torch.bernoulli(torch.tensor([[0.5] * N] * E))
        noise[:, :, d, 0] = 0.1 * (torch.mul(mixture_label, 0.5 * torch.randn(E, N) - 1) + torch.mul(1 - mixture_label, 0.5 * torch.randn(E, N) + 1))
    elif noise_type == 'exp':
        noise[:, :, d, 0] = 0.1 * (exponential_distr.sample([E, N]) - 1)
    elif noise_type == 'normal':
        noise[:, :, d, 0] = 0.1 * torch.randn(E, N)
    else:
        print('Error: insert a valid noise distribution type')
        quit()
    # check invertibility
    if torch.det(torch.eye(d + 1) - B) == 0:
        print('Error: the give configuration does not yield a directed acyclic graph.\nTry again')
        quit()
    # Generate data
    Z = torch.matmul(torch.inverse(torch.eye(d + 1) - B), noise)
    X = Z[:, :, 0:d, 0]
    Y = Z[:, :, d, :]
else:
    print('Error: choose proper dataset generation technique')
    quit()
# Initialize labels for each environment
y_0 = torch.tensor([0]*N)
y_1 = torch.tensor([1]*N)
# Label environments in E x N tensor
labels = torch.tensor([list(range(E))]*N).transpose(0, 1)

########################################################################################################################
# Perform Linear Regression
########################################################################################################################
# Gather Design Matrix by pooling the data over the environments
Y_reg = Y.view(-1, N*E, 1).squeeze()
X_reg = X.view(-1, N*E, d).squeeze()
# Record start time
start = time.time()
# Perform regression
model = LinearRegression(fit_intercept=False)
model.fit(X_reg, Y_reg)
# Record elapsed time
time_reg = time.time() - start
# Gather parameter t
t_reg = torch.tensor([model.coef_]).view(-1,1)
# Gather residuals
res_reg = Y_reg.view(-1, 1) - torch.mm(X_reg, t_reg)

########################################################################################################################
# Setup for training
########################################################################################################################
# Initialize parameter to optimize, warmstart t by setting it to the regression one
if regression_kickstart:
    t = copy.deepcopy(t_reg)
else:
    t = torch.tensor([[0.]]*d)
t.requires_grad_()
# Define a function generating the data


def generator(param, env):
    data = Y[env, :, :] - torch.mm(X[env, :, :], param)
    return data


# Definition of the discriminator NN
discriminator = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ELU(),
    torch.nn.Linear(H, H),
    torch.nn.ELU(),
    torch.nn.Linear(H, D_out),
    torch.nn.LogSoftmax(dim=1)
)
# Setup optimizers
loss_fn_d = torch.nn.NLLLoss(reduction='sum')
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d)
optimizer_g = torch.optim.SGD([t], learning_rate_g)
# Every pairing of environments should need a different neural network.
# For each pairing store the weights of a different neural network
# Initialize MSE_est
MSE_est = []
t_est = torch.zeros(d, passes)

########################################################################################################################
# Training
########################################################################################################################
# Record start time
start = time.time()
# Warmstart
for i in range(steps_d_warmstart):
    loss_d = 0
    for l in range(E):
        loss_d += (loss_fn_d(discriminator(generator(t.detach(), l)), labels[l, :]))

    # Optimization step
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()
for k in range(passes):
    # # Train both
    for j in range(steps_both):
        # Optimize discriminator
        for i in range(steps_d):
            loss_d = 0
            for l in range(E):
                loss_d += (loss_fn_d(discriminator(generator(t.detach(), l)), labels[l, :]))
            # Optimization step
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            # Weight clipping
            for p in discriminator.parameters():
                p.data.clamp_(-1, 1)
        # Optimize generator with respect to the newly trained discriminator
        for i in range(steps_g):
            # Compute the discrepancy that has been recognized, can extend to f-GAN
            loss_g = 0
            for l in range(E):
                loss_g += -(loss_fn_d(discriminator(generator(t, l)), labels[l, :]))
            # Optimization step
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
    # Record achieved MSE/t_est after a pass
    res_est = Y - torch.matmul(X, t.detach())
    res_est = res_est.view(-1, N * E, 1).squeeze()
    MSE_est.append(torch.mean(res_est.pow(2)).numpy())
    t_est[:, k] = t.squeeze().detach()
    # Plot all discriminators
    if plot_discriminator_progress and k % 100 == 0:
        x_fun = torch.zeros(N, 1)
        x_fun[:, 0] = torch.tensor(np.linspace(ran[0], ran[1], N))
        y_fun = torch.exp(discriminator(x_fun).detach())
        for l in range(E):
            plt.plot(generator(t.detach(), l), np.linspace(0, 1, N), marker='o', linestyle='', linewidth=2, color=(0.5+l/(2*(E-1)), 0.5, 1-l/(2*(E-1)), 0.1), label='classification')
        for l in range(E):
            plt.plot(x_fun, y_fun[:, l], marker='', linewidth=6, linestyle='-', color=(1, 1, 1, 1))
            plt.plot(x_fun, y_fun[:, l], marker='', linewidth=2, linestyle='-', color=(0.5 + l / (2 * (E - 1)), 0.5, 1 - l / (2 * (E - 1)), 1), label='classification')
        plt.ylabel('discriminator output')
        plt.title('Discriminator Status after '+str(k)+' steps')
        # plt.legend()
        plt.xlim(ran)
        plt.show()
    if k*20 % passes == 0:
        progress = int(k/passes*20)
        loading = '[' + '='*progress + ' '*(20-progress) + ']\t' + str(int(k/passes*100)) + '%'
        print('\r', 'Training in progress: ', loading, sep='', end='')
time_gan = time.time() - start
print('\rTraining in progress: ', '[', '='*20, ']\t100%', sep='', end='\n')

########################################################################################################################
# Printing necessary information
########################################################################################################################
# Gather residuals (res_reg already computed in loop above)
res_true = Y - torch.matmul(X, true_t)
res_true = res_true.view(-1, N*E, 1).squeeze()
# Compute MSEs (MSE_est already computed in loop above)
MSE_true = torch.mean(res_true.pow(2)).numpy()
MSE_reg = torch.mean(res_reg.pow(2)).numpy()
# Average out last 100 computations
t_avg = torch.mean(t_est[:, -100:], 1)
# Print info
with np.printoptions(formatter={'float': '{: 0.3f}'.format}):
    print('True parameter: \t\t\t', true_t.squeeze().numpy())
    print('Estimated parameter: \t\t', t.detach().squeeze().numpy())
    print('Estimated avg. parameter: \t', t_avg.detach().squeeze().numpy())
    print('Regression parameter: \t\t', t_reg.squeeze().numpy())
print('True MSE: \t\t\t\t\t', '{:.5f}'.format(MSE_true))
print('Regression MSE: \t\t\t', '{:.5f}'.format(MSE_reg))
print('Estimated MSE: \t\t\t\t', '{:.5f}'.format(MSE_est[-1]))
# Print time taken
print('Time of regression: \t\t', '{:.5f}'.format(time_reg))
print('Time of GAN approach: \t\t', '{:.5f}'.format(time_gan))

########################################################################################################################
# Plotting the results
########################################################################################################################
# Choose the first and last environment
e_0 = generator(t.detach(), 0)
e_1 = generator(t.detach(), E-1)
# Plot all environments together in true approach
if plot_all_environments_true and E > 2:
    fig, axs = plt.subplots(E, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    axs[0].set_yticks([])
    axs[0].set(title='All environments with true/causal approach')
    for i in range(E):
        axs[i].plot(res_true[N*i:N*(i+1)].squeeze(), np.linspace(0, 1, N), marker='o', markersize=3, linestyle='', color=(0.5+i/(2*(E-1)), 0.5, 1-i/(2*(E-1)), 0.3), label='e = '+str(i+1))
        axs[i].set(xlim=ran)
        axs[i].set(xlabel='residuals across environments')
        axs[i].legend(loc=1)
    plt.savefig('plots/multi-class_all_true.pdf')
    fig.show()
# Plot all environments together in regression approach
if plot_all_environments_reg and E > 2:
    fig, axs = plt.subplots(E, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    axs[0].set_yticks([])
    axs[0].set(title='All environments with regression approach')
    for i in range(E):
        axs[i].plot(res_reg[N*i:N*(i+1)].squeeze(), np.linspace(0, 1, N), marker='o', markersize=3, linestyle='', color=(0.5+i/(2*(E-1)), 0.5, 1-i/(2*(E-1)), 0.3), label='e = '+str(i+1))
        axs[i].set(xlim=ran)
        axs[i].set(xlabel='residuals across environments')
        axs[i].legend(loc=1)
    plt.savefig('plots/multi-class_all_reg.pdf')
    fig.show()
# Plot all environments together in GAN approach
if plot_all_environments_GAN and E > 2:
    fig, axs = plt.subplots(E, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    axs[0].set_yticks([])
    axs[0].set(title='All environments with GAN approach')
    for i in range(E):
        axs[i].plot(generator(t.detach(), i), np.linspace(0, 1, N), marker='o', markersize=3, linestyle='', color=(0.5+i/(2*(E-1)), 0.5, 1-i/(2*(E-1)), 0.3), label='e = '+str(i+1))
        axs[i].plot(x_fun, y_fun[:, i], marker='', linewidth=6, linestyle='-', color=(1, 1, 1, 1))
        axs[i].plot(x_fun, y_fun[:, i], marker='', linewidth=2, linestyle='-', color=(0.5+i/(2*(E-1)), 0.5, 1-i/(2*(E-1)), 1))
        axs[i].set(xlim=ran)
        axs[i].set(xlabel='residuals across environments')
        axs[i].legend(loc=1)
    plt.savefig('plots/multi-class_all_gan.pdf')
    fig.show()
# Plot MSE_est progress
plt.plot(range(passes), MSE_est, marker='', linestyle='-', color='black', linewidth=0.5, label='MSE_est')
plt.plot(range(passes), [MSE_reg]*passes, marker='', linestyle='--', color=colors[1], label='MSE_reg')
plt.plot(range(passes), [MSE_true]*passes, marker='', linestyle='--', color=colors[0], label='MSE_true')
plt.yscale('log')
plt.xlabel('number of passes')
plt.ylabel('MSE achieved')
plt.ylim([MSE_reg/1.1, MSE_true*1.1])
plt.title('Advancement of MSE after each pass')
plt.legend(loc=1)
plt.show()
# Plot t_est estimation progress
plt.plot(range(passes), [1.]*passes, marker='', linestyle=':', color=(0.5, 0.5, 0.5, 1))
plt.plot(range(passes), [0.]*passes, marker='', linestyle=':', color=(0.5, 0.5, 0.5, 1))
for i in range(d):
    plt.plot(range(passes), t_est[i, :], marker='', linestyle='-', color=(0, 0.5+i/(2*(d-1)), 1-i/(2*(d-1)), 0.8), zorder=i)
    dot_x_position = int(0.2*passes*i/(d-1) + 0.8*passes - 1)
    dot_y_position = t_est[i, dot_x_position]
    plt.plot(dot_x_position, dot_y_position, marker='o', markersize=20, linestyle='', color=(0, 0.5+i/(2*(d-1)), 1-i/(2*(d-1))), zorder=i)
    plt.plot(dot_x_position, dot_y_position, marker='o', markersize=18, linestyle='', color=(1, 1, 1, 1), zorder=i)
    plt.text(dot_x_position, dot_y_position, r'$\hat{\theta}_' + str(i) + '$', ha='center', va='center', zorder=i)
plt.xlabel('number of passes')
plt.ylabel('Estimated coefficient')
plt.title(r'Advancement of the estimated $\theta$ parameter after each pass')
plt.savefig('plots/multi-class_progress.pdf')
plt.show()
