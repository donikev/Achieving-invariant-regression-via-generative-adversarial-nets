########################################################################################################################
#
# Multi-class GAN
#
# Goal: test the multi-class algorithm on the flow cytometry dataset
#
########################################################################################################################

########################################################################################################################
# Preamble
########################################################################################################################
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random
import time
import copy
from tabulate import tabulate
#import progressbar
from time import sleep
from sklearn.linear_model import LinearRegression

########################################################################################################################
# Parameter Initialization
########################################################################################################################
# Do you want information of the process being printed and/or plotted?
plot_discriminator_progress = False
info_step = 50
plot_all_environments_true = True
plot_all_environments_reg = True
plot_all_environments_GAN = True
# Plotting parameters
mpl.rcParams["figure.figsize"] = [6.4, 6.4] # Default [6.4, 4.8] option [9.6, 7.2]
mpl.rcParams["figure.dpi"] = 100 # Default 100
# Choose: deterministic, old_random, new_random
dataset_generation = 'new_random'
# Choose: exp, mixture, normal
noise_type = 'normal'
# Choose: TV, JS
GAN_type = 'TV'
# N: sample size per environment
N = 707
# E: number of environments
E = 9
# Dimension of X
d = 11-1
# Number of causal parameters
m = 3
# Discriminator network parameters
D_in = 1
H = 50
D_out = E
# Optimization parameters
passes = 3000
steps_both = 1
steps_d = 1
steps_d_warmstart = 20
steps_g = 1
regression_kickstart = True
lambda_1 = 0.001
lambda_2 = 0.15
learning_rate_d = 1e-3
learning_rate_g = 1e-2
# Choose response variable
response_index = 5
# For reproducibility
torch.manual_seed(1)
random.seed(1)
# Plotting parameters: coverage serves as a margin, ran: x axis limits
margin = 5  # 10 is also a good choice
ran = [-margin, margin]
# Standard color palette
colors_transparent = [(1, 0.5, 0.5, 0.3), (0.5, 0.5, 1, 0.3)]
colors = [(1, 0.5, 0.5, 1), (0.5, 0.5, 1, 1)]
# Print model parameters
print('Model parameters: \nSample size: \t\t\t\t', N, '\nDimension of X: \t\t\t', d, '\nNumber of environments: \t', E)

########################################################################################################################
# Dataset Generation
########################################################################################################################
covariates_index = [i for i in range(d+1) if i != response_index]
true_t = torch.tensor([[0.]] * d)
# File names
file_names = ["1. cd3cd28",
              "2. cd3cd28icam2",
              "3. cd3cd28+aktinhib",
              "4. cd3cd28+g0076",
              "5. cd3cd28+psitect",
              "6. cd3cd28+u0126",
              "7. cd3cd28+ly",
              "8. pma",
              "9. b2camp",
              "10. cd3cd28icam2+aktinhib",
              "11. cd3cd28icam2+g0076",
              "12. cd3cd28icam2+psit",
              "13. cd3cd28icam2+u0126",
              "14. cd3cd28icam2+ly"]
# Get observational Data Frame
data = pd.read_excel("data\\Sachs_Nolan\\" + file_names[0] + ".xls")
# Get headers
var_names = list(data.columns)
Z = torch.empty(E, N, d+1, 1)
for i in range(E):
    # Get Data Frame
    data = pd.read_excel("data\\Sachs_Nolan\\" + file_names[i] + ".xls").to_numpy()
    Z[i, :, :, 0] = torch.tensor(data[0:N, :])
X = torch.log(Z[:, :, covariates_index, 0])
Y = torch.log(Z[:, :, response_index, :])
# Center the variables
X_mean = torch.zeros(E, 1, d)
Y_mean = torch.zeros(E, 1, 1)
for i in range(E):
    X_mean[i, 0, :] = X[i, :, :].mean(0)
    Y_mean[i, 0, 0] = Y[i, :, 0].mean()
X = X - X_mean
Y = Y - Y_mean

# Initialize labels for each environment
y_0 = torch.tensor([[0.]*N]).transpose(0, 1)
y_1 = torch.tensor([[1.]*N]).transpose(0, 1)
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
res_reg = torch.zeros(E, N)
for e in range(E):
    res_reg[e, :] = torch.squeeze(Y[e, :, :] - torch.mm(X[e, :, :], t_reg))

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
MSE_est_process = []
loss_g_l1_progress = torch.zeros(passes)
loss_g_sq_progress = torch.zeros(passes)
loss_g_score_progress = torch.zeros(passes)
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
                loss_g_l1 = t.abs().sum()/d
                loss_g_sq = generator(t, l).pow(2).sum()/(N*E)
                loss_g_score = -(loss_fn_d(discriminator(generator(t, l)), labels[l, :]))/(N*E)
                loss_g += lambda_1*loss_g_l1 + lambda_2*loss_g_sq + loss_g_score
                loss_g_l1_progress[k] += loss_g_l1.detach()
                loss_g_sq_progress[k] += loss_g_sq.detach()
                loss_g_score_progress[k] += loss_g_score.detach()
            # Optimization step
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
    # Record achieved MSE/t_est after a pass
    res_est = Y - torch.matmul(X, t.detach())
    res_est = res_est.view(-1, N * E, 1).squeeze()
    MSE_est_process.append(torch.mean(res_est.pow(2)).numpy())
    t_est[:, k] = t.squeeze().detach()
    # Plot all discriminators
    if plot_discriminator_progress and k % info_step == 0:
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
# Gather true residuals (res_reg already computed in loop above)
res_true = torch.squeeze(Y - torch.matmul(X, true_t))
# Gather gan residuals
res_est = torch.zeros(E, N)
for e in range(E):
    res_est[e, :] = generator(t, e).detach().squeeze()
# Compute MSEs (MSE_est_process already computed in loop above)
MSE_true_env = torch.mean(res_true.pow(2), 1)
MSE_true = torch.mean(MSE_true_env).item()
MSE_reg_env = torch.mean(res_reg.pow(2), 1)
MSE_reg = torch.mean(MSE_reg_env).item()
MSE_est_env = torch.mean(res_est.pow(2), 1)
MSE_est = torch.mean(MSE_est_env).item()
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
print('Estimated MSE: \t\t\t\t', '{:.5f}'.format(MSE_est))
# Print time taken
print('Time of regression: \t\t', '{:.5f}'.format(time_reg))
print('Time of GAN approach: \t\t', '{:.5f}'.format(time_gan))
# Print in table form
table = [['Estimated parameter'] + t.detach().view(1, -1).squeeze().tolist(),
         ['Regression parameter'] + t_reg.view(1, -1).squeeze().tolist()]
print(tabulate(table, headers = ["Covariates"] + [var_names[i] for i in covariates_index]))

########################################################################################################################
# Plotting the results
########################################################################################################################
# Choose the first and last environment
e_0 = generator(t.detach(), 0)
e_1 = generator(t.detach(), E-1)
# Plot regression results
fig, axs = plt.subplots(2, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
axs[0].plot(res_reg[0, :].squeeze(), np.linspace(0, 1, N), marker='o', markersize=3, linestyle='', color=(0.5, 0.5, 1, 0.3), label='e = 0')
axs[1].plot(res_reg[-1, :].squeeze(), np.linspace(0, 1, N), marker='o', markersize=3, linestyle='', color=(1, 0.5, 0.5, 0.3), label='e = '+str(E-1))
axs[0].set_yticks([])
axs[0].set(title='Regression approach')
axs[1].set(xlim=ran)
axs[1].set(xlabel='residuals across environments')
fig.legend()
fig.show()
# Plot gan results
x_fun = torch.zeros(N, 1)
x_fun[:, 0] = torch.tensor(np.linspace(ran[0], ran[1], N))
y_fun = discriminator(x_fun).detach().squeeze().exp()
x_fun = x_fun.squeeze()
fig, axs = plt.subplots(3, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
axs[0].plot(res_est[0, :], np.linspace(0, 1, N), marker='o', markersize=3, linestyle='', color=(0.5, 0.5, 1, 0.3), label='e = 0')
axs[0].set_yticks([])
axs[0].set(title='GAN approach')
axs[1].plot(res_est[-1, :], np.linspace(0, 1, N), marker='o', markersize=3, linestyle='', color=(1, 0.5, 0.5, 0.3), label='e = '+str(E-1))
axs[2].plot(x_fun, y_fun, marker='', linestyle='-', color=(0.5, 0.5, 0.5, 0.5))
axs[2].set(xlim=ran)
axs[2].set(xlabel='residuals across environments')
fig.legend()
fig.show()
# Plot all environments together in true approach
if plot_all_environments_true and E > 2:
    fig, axs = plt.subplots(E, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    axs[0].set_yticks([])
    axs[0].set(title=r'All environments with $\theta = 0$' + '\nMSE = ' + '{:.3f}'.format(MSE_true) + ', Y: ' + var_names[response_index])
    for i in range(E):
        axs[i].plot(res_true[i, :], np.linspace(0, 1, N), marker='o', markersize=3, linestyle='', color=(0.5+i/(2*(E-1)), 0.5, 1-i/(2*(E-1)), 0.3), label='e = '+str(i+1))
        axs[i].text(0.9*ran[0], 0.5, '{:.3f}'.format(MSE_true_env[i]), ha='left', va='center')
        axs[i].set(xlim=ran)
        axs[i].set(xlabel='residuals across environments')
        axs[i].legend(loc=1)
    fig.show()
# Plot all environments together in regression approach
if plot_all_environments_reg and E > 2:
    fig, axs = plt.subplots(E, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    axs[0].set_yticks([])
    axs[0].set(title=r'All environments with $\hat{\theta}_{reg}$' + '\nMSE = ' + '{:.3f}'.format(MSE_reg) )# + ', Y: ' + var_names[response_index])
    for i in range(E):
        axs[i].plot(res_reg[i, :], np.linspace(0, 1, N), marker='o', markersize=3, linestyle='', color=(0.5+i/(2*(E-1)), 0.5, 1-i/(2*(E-1)), 0.3), label='e = '+str(i+1))
        axs[i].text(0.9*ran[0], 0.5, '{:.3f}'.format(MSE_reg_env[i]), ha='left', va='center')
        axs[i].set(xlim=ran)
        axs[i].set(xlabel='residuals across environments')
        axs[i].legend(loc=1)
    fig.savefig('Plots/multi-class_SN_reg.pdf')
    fig.show()
# Plot all environments together in GAN approach
if plot_all_environments_GAN and E > 2:
    fig, axs = plt.subplots(E, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    axs[0].set_yticks([])
    axs[0].set(title=r'All environments with $\hat{\theta}_{GAN}$' + '\nMSE = ' + '{:.3f}'.format(MSE_est_process[-1]) )# + ', Y: ' + var_names[response_index])
    for i in range(E):
        axs[i].plot(res_est[i, :], np.linspace(0, 1, N), marker='o', markersize=3, linestyle='', color=(0.5+i/(2*(E-1)), 0.5, 1-i/(2*(E-1)), 0.3), label='e = '+str(i+1))
        axs[i].plot(x_fun, y_fun[:, i], marker='', linewidth=6, linestyle='-', color=(1, 1, 1, 1))
        axs[i].plot(x_fun, y_fun[:, i], marker='', linewidth=2, linestyle='-', color=(0.5+i/(2*(E-1)), 0.5, 1-i/(2*(E-1)), 1))
        axs[i].text(0.9*ran[0], 0.5, '{:.3f}'.format(MSE_est_env[i]), ha='left', va='center')
        axs[i].set(xlim=ran)
        axs[i].set(xlabel='residuals across environments')
        axs[i].legend(loc=1)
    fig.savefig('Plots/multi-class_SN_gan.pdf')
    fig.show()
# Plot MSE_est progress
plt.plot(range(passes), MSE_est_process, marker='', linestyle='-', color='black', linewidth=0.5, label='MSE_est')
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
for i in range(d):
    plt.plot(range(passes), t_est[i, :], marker='', linestyle='-', color=(0, 0.5+i/(2*(d-1)), 1-i/(2*(d-1)), 0.8), zorder=i)
    plt.plot(range(passes), [t_reg[i, 0]]*passes, marker='', linestyle=':', color=(0, 0.5+i/(2*(d-1)), 1-i/(2*(d-1)), 0.8), zorder=i)
    dot_x_position = int(0.2*passes*i/(d-1) + 0.8*passes - 1)
    dot_y_position = t_est[i, dot_x_position]
    plt.plot(dot_x_position, dot_y_position, marker='o', markersize=20, linestyle='', color=(0, 0.5+i/(2*(d-1)), 1-i/(2*(d-1))), zorder=i)
    plt.plot(dot_x_position, dot_y_position, marker='o', markersize=18, linestyle='', color=(1, 1, 1, 1), zorder=i)
    plt.text(dot_x_position, dot_y_position, r'$\hat{\theta}_' + str(i) + '$', ha='center', va='center', zorder=i)
plt.xlabel('number of passes')
plt.ylabel('Estimated coefficient')
plt.title(r'Advancement of the estimated $\theta$ parameter after each pass')
plt.show()
# Plot loss process
plt.plot(range(passes), loss_g_l1_progress, marker='', linestyle='-', color=(0.5, 0.5, 1, 0.8), zorder=0, label=r'$\ell_1$ loss')
plt.plot(range(passes), loss_g_sq_progress, marker='', linestyle='-', color=(0.75, 0.5, 0.75, 0.8), zorder=0, label='Squared deviation')
plt.plot(range(passes), -loss_g_score_progress, marker='', linestyle='-', color=(1, 0.5, 0.5, 0.8), zorder=0, label='Discriminator loss')
plt.xlabel('number of passes')
plt.ylabel('Loss incurred')
plt.title('Losses incurred')
plt.legend(loc=6)
plt.show()