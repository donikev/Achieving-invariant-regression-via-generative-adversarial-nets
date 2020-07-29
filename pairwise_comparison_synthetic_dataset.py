
########################################################################################################################
#
# Pairwise Comparison Algorithm on the Synthetic Dataset
#
# Goal: show that the pairwise comparison algorithm works on an artificial dataset
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
from time import sleep
from sklearn.linear_model import LinearRegression

########################################################################################################################
# Initialize parameters
########################################################################################################################
# Do you want information of the process being printed and/or plotted?
plot_discriminator_progress = True
plot_all_environments_true = True
plot_all_environments_reg = True
plot_all_environments_GAN = True
# Check whether the interventions on the variables are sensible
plot_variables = True
plot_variables_variable = 4
# Choose: deterministic, random
dataset_generation = 'random'
# Choose: exp, mixture, normal
noise_type = 'exp'
# Choose: TV, JS, W
GAN_type = 'W'
# N: sample size per environment
N = 500
# E: number of environments
E = 7
# Dimension of X
d = 5
# Number of causal parameters
m = 3
# Discriminator network parameters
D_in = 1
H = 2
D_out = 1
# Generate possible environment combinations, without duplicates and index them in one dimension
combinations_number = int(E*(E-1)/2)
combinations_index = range(combinations_number)
combinations = []
for i in range(0, E):
    for j in range(i+1, E):
        combinations.append([i, j])
# Optimization parameters
passes = 300 * combinations_number
steps_both = 1
steps_d = 0
steps_g = 0
regression_kickstart = True
learning_rate_d = 1e-1
learning_rate_g = 1e-2
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
      '\nChosen type of GAN:\t\t\t', GAN_type,
      '\nChosen type of noise:\t\t', noise_type)

########################################################################################################################
# Generate dataset
########################################################################################################################
if dataset_generation == 'deterministic':
    # True causal parameter
    true_t = torch.tensor([[1.]] * m + [[0.]] * (d - m))
    X = torch.randn(E, N, d)
    # Perturb data in other environments
    X[1:E, :, :] = X[1:E, :, :] * 2
    Y = torch.matmul(X, true_t) + 0.1 * torch.randn(E, N, 1)
    # Perturb last column s.t. structural graph has Y -> X_d, but vary perturbation in at least one environment, e = E
    X[:, :, d - 1] = Y[:, :, 0] + 0.1 * torch.cat([torch.randn(E - 1, N), 3 * torch.randn(1, N)], 0)
elif dataset_generation == 'random':
    # True causal parameter, \theta^\star
    true_t = torch.tensor([[1.]] * m + [[0.]] * (d - m))
    b_2 = true_t.view(1, -1)
    # X_i that are children of Y, hence Y -> X_i
    b_1 = torch.tensor([[0.]] * (d - 1) + [[1.]])
    # Generate upper triangular matrix
    B = torch.zeros(d + 1, d + 1)
    for i in range(d + 1):
        for j in range(i):
            B[i, j] = 0.1*torch.randn(1) #torch.randint(-2, 2, [1])
    B[-1, 0:-1] = b_2
    B[-2, :] = torch.tensor([[0] * d + [1]])
    # Generate noise vector
    noise = 0.1 * torch.randn(E, N, d + 1, 1)
    noise[:, :, 0, 0] = torch.randn(E, N)
    #noise[:, :, 0:(d-2), 0] *= 2
    # noise intervention in all environments X_1, ..., X_{d-1}
    for i in range(0, min(E-1, d)):
        noise[i + 1, :, i, 0] *= torch.randint(2, 3, [1])
    # noise intervention on X_d
    #noise[:, :, d - 1, 0] = 0.1 * torch.randn(E, N)
    #noise[E - 1, :, d - 1, 0] = 0.3 * torch.randn(N)
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
    # Plot variables
    if plot_variables:
        fig, axs = plt.subplots(E, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
        axs[0].set_yticks([])
        for i in range(E):
            axs[i].plot(noise[i, :, plot_variables_variable, 0], np.linspace(0, 1, N), marker='o', markersize=3,
                        linestyle='',
                        color=(0.5 + i / (2 * (E - 1)), 0.5, 1 - i / (2 * (E - 1)), 0.3), label='e = ' + str(i + 1))
            axs[i].set(xlim=ran)
            axs[i].set(xlabel='noise')
            axs[i].legend(loc=1)
        fig.show()
else:
    print('Error: choose proper dataset generation technique')
    quit()
# Initialize labels for each environment
y_0 = torch.tensor([[0.]*N]).transpose(0, 1)
y_1 = torch.tensor([[1.]*N]).transpose(0, 1)
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
res_reg = Y_reg.view(-1,1) - torch.mm(X_reg, t_reg)

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
if GAN_type == 'W':
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ELU(),
        torch.nn.Linear(H, H),
        torch.nn.ELU(),
        torch.nn.Linear(H, D_out),
    )
else:
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ELU(),
        torch.nn.Linear(H, H),
        torch.nn.ELU(),
        torch.nn.Linear(H, D_out),
        torch.nn.Sigmoid()
    )
# Setup optimizers
loss_fn_d = torch.nn.BCELoss(reduction='sum')
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d)
optimizer_g = torch.optim.SGD([t], learning_rate_g)
# Every pairing of environments should need a different neural network.
# For each pairing store the weights of a different neural network
params = [{} for i in combinations_index]
for i in combinations_index:
    params[i] = copy.deepcopy(discriminator.state_dict())
# Initialize MSE_est
MSE_est = []
t_est = torch.zeros(d, passes)

########################################################################################################################
# Training
########################################################################################################################
# Record start time
start = time.time()
for k in range(passes):
    # Choose a pairing of environments by sampling from the index
    index = random.sample(combinations_index, 1)[0]
    envs = combinations[index]
    # Load the network parameters corresponding to the chosen pairing
    discriminator.load_state_dict(params[index])
    # # Train both
    for j in range(steps_both):
        # Generate data according to the parameter t
        e_0 = generator(t.detach(), envs[0])
        e_1 = generator(t.detach(), envs[1])
        # Optimize discriminator
        for i in range(steps_d + 1):
            # e_0, e_1 inherited form generator loop
            # build inputs and outputs that the discriminator should look at
            y_pred_0 = discriminator(e_0)
            y_pred_1 = discriminator(e_1)
            # Compute discriminator loss
            if GAN_type == 'TV':
                loss_d = -(torch.mean(y_pred_1) - torch.mean(y_pred_0))
            elif GAN_type == 'JS':
                loss_d = -(torch.log(y_pred_1).mean() + torch.log(1 - y_pred_0).mean())
            elif GAN_type == 'W':
                loss_d = -(torch.mean(y_pred_1) - torch.mean(y_pred_0))
            else:
                print('Error: choose a proper GAN type')
                quit()
            # Optimization step
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            # Do weight clipping for WGAN
            if GAN_type == 'W':
                for p in discriminator.parameters():
                    p.data.clamp_(-1, 1)
            # Update parameters of this specific network to the list
            params[index] = copy.deepcopy(discriminator.state_dict())
        # Optimize generator with respect to the newly trained discriminator
        for i in range(steps_g + 1):
            # Compute the discrepancy that has been recognized, can extend to f-GAN
            if GAN_type == 'TV':
                loss_g = (torch.mean(discriminator(generator(t, envs[1]))) - torch.mean(discriminator(generator(t, envs[0]))))
            elif GAN_type == 'JS':
                loss_g = (torch.log(discriminator(generator(t, envs[1]))).mean() + torch.log(1 - discriminator(generator(t, envs[0]))).mean())
            elif GAN_type == 'W':
                loss_g = (torch.mean(discriminator(generator(t, envs[1]))) - torch.mean(discriminator(generator(t, envs[0]))))
            else:
                print('Error: choose a proper GAN type')
                quit()
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
    if plot_discriminator_progress and k % (100*combinations_number) == 0 and E > 2:
        fig, axs = plt.subplots(E-1, E-1, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})
        #fig.suptitle('Discriminator output after $' + str(int(k / combinations_number)) + r'\cdot|C|$ steps')
        #fig.subplots_adjust(top=2)
        axs[0, 0].set_yticks([])
        axs[0, 0].set_xticks([])
        #axs[0,0].set(title='All discriminators after '+str(k)+' passes')
        counter = 0
        for i in range(E-1):
            for j in range(i):
                axs[i, j].set_facecolor((0.5, 0.5, 0.5, 0.1))
            for j in range(i, E-1):
                discriminator.load_state_dict(params[counter])
                x_fun = torch.zeros(N, 1)
                x_fun[:, 0] = torch.tensor(np.linspace(ran[0], ran[1], N))
                y_fun = discriminator(x_fun).detach().squeeze()
                x_fun = x_fun.squeeze()
                axs[i, j].plot(x_fun, y_fun, marker='', linestyle=':', color='black', label='classification')
                if GAN_type == 'W':
                    axs[i, j].set(xlim=ran)
                else:
                    axs[i, j].set(xlim=ran, ylim=[0, 1])
                    axs[i, j].text(0, 0.8, str(combinations[counter][0]+1)+' vs '+str(combinations[counter][1]+1), ha='center', va='center', bbox=dict(boxstyle="round", ec=(0.5, 0.5, 0.5, 0.5), fc=(0.8, 0.8, 0.8, 0.5)))
                counter = counter + 1
        #fig.savefig('plots/steps'+str(int(k / combinations_number))+'.pdf')
        fig.show()
    if k*20%passes == 0:
        progress = int(k/passes*20)
        loading = '[' + '='*progress + ' '*(20-progress) + ']\t' + str(int(k/passes*100)) + '%'
        print('\r', 'Training in progress: ', loading, sep='', end='')
time_gan = time.time() - start
print('\rTraining in progress: ', '[','='*20,']\t100%', sep='', end='\n')

########################################################################################################################
# Printing necessary infromation
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
index = E-2
envs = combinations[index]
discriminator.load_state_dict(params[index])
e_0 = generator(t.detach(), envs[0])
e_1 = generator(t.detach(), envs[1])
# Plot all environments together in true approach
if plot_all_environments_true and E > 2:
    fig, axs = plt.subplots(E, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    axs[0].set_yticks([])
    axs[0].set(title=r'All environments with $\theta^\star$')
    for i in range(E):
        axs[i].plot(res_true[N*i:N*(i+1)].squeeze(), np.linspace(0, 1, N), marker='o', markersize=3, linestyle='', color=(0.5+i/(2*(E-1)), 0.5, 1-i/(2*(E-1)), 0.3), label='e = '+str(i+1))
        axs[i].set(xlim=ran)
        axs[i].set(xlabel='residuals across environments')
        axs[i].legend(loc=1)
    plt.savefig('plots/pairwise_all_true.pdf')
    fig.show()
# Plot all environments together in regression approach
if plot_all_environments_reg and E > 2:
    fig, axs = plt.subplots(E, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    axs[0].set_yticks([])
    axs[0].set(title=r'All environments with $\hat{\theta}_{reg}$')
    for i in range(E):
        axs[i].plot(res_reg[N*i:N*(i+1)].squeeze(), np.linspace(0, 1, N), marker='o', markersize=3, linestyle='', color=(0.5+i/(2*(E-1)), 0.5, 1-i/(2*(E-1)), 0.3), label='e = '+str(i+1))
        axs[i].set(xlim=ran)
        axs[i].set(xlabel='residuals across environments')
        axs[i].legend(loc=1)
    plt.savefig('plots/pairwise_all_reg.pdf')
    fig.show()
# Plot all environments together in GAN approach
if plot_all_environments_GAN and E > 2:
    fig, axs = plt.subplots(E, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    axs[0].set_yticks([])
    axs[0].set(title=r'All environments with $\hat{\theta}_{'+GAN_type+'}$')
    for i in range(E):
        axs[i].plot(generator(t.detach(), i), np.linspace(0, 1, N), marker='o', markersize=3, linestyle='', color=(0.5+i/(2*(E-1)), 0.5, 1-i/(2*(E-1)), 0.3), label='e = '+str(i+1))
        axs[i].set(xlim=ran)
        axs[i].set(xlabel='residuals across environments')
        axs[i].legend(loc=1)
    plt.savefig('plots/pairwise_all_gan.pdf')
    fig.show()
# Plot MSE_est progress
plt.plot(range(passes), MSE_est, marker='', linestyle='-', color='black', linewidth=0.5, label='MSE_est')
plt.plot(range(passes), [MSE_reg]*passes, marker='', linestyle='--', color=colors[1], label='MSE_reg')
plt.plot(range(passes), [MSE_true]*passes, marker='', linestyle='--', color=colors[0], label='MSE_true')
plt.yscale('log')
plt.xlabel('number of passes')
plt.ylabel('MSE achieved')
plt.ylim([MSE_reg/1.1,MSE_true*1.1])
plt.title('Advancement of MSE after each pass')
plt.legend(loc=1)
plt.show()
# Plot t_est estimation progress
plt.plot(range(passes), [1.]*passes, marker='', linestyle=':', color=(0.5, 0.5, 0.5, 1))
plt.plot(range(passes), [0.]*passes, marker='', linestyle=':', color=(0.5, 0.5, 0.5, 1))
for i in range(d):
    plt.plot(range(passes), t_est[i, :], marker='', linestyle='-', color=(0, 0.5+i/(2*(d-1)), 1-i/(2*(d-1)), 0.8), zorder=i)
    dot_x_position = int(0.2 * passes * i / (d - 1) + 0.8 * passes - 1)
    dot_y_position = t_est[i, dot_x_position]
    plt.plot(dot_x_position, dot_y_position, marker='o', markersize=20, linestyle='', color=(0, 0.5 + i / (2 * (d - 1)), 1 - i / (2 * (d - 1))), zorder=i)
    plt.plot(dot_x_position, dot_y_position, marker='o', markersize=18, linestyle='', color=(1, 1, 1, 1), zorder=i)
    plt.text(dot_x_position, dot_y_position, r'$\hat{\theta}_' + str(i) + '$', ha='center', va='center', zorder=i)
plt.xlabel('number of steps')
plt.ylabel('Estimated coefficient')
plt.title(r'Advancement of the estimated $\theta$ parameter after each pass')
plt.savefig('plots/progress.pdf')
plt.show()
