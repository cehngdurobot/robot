% Revision History:
% v1: Feb. 5th, 2020, by Shuai Li
% v2: Feb. 11th, 2020, by Shuai Li
% Modified: April 10th, 2025 ¨C Using PSO for optimization

clear all
close all
tic

%%------------------------------------------
% Calibration Part
% This part calibrates the robot based on the dataset.
% It uses the following functions:
%   pre_processing.m   (to compute forward kinematics and the Jacobian matrix)
%   my_forward.m       (forward kinematics evaluation)
%   my_Jacobian.m      (Jacobian computation)
%%------------------------------------------

% System parameters (real and nominal values)
% Parameters order: a1,a2,...,a6, d1,...,d6, alpha1,...,alpha6, theta0_1,...,theta0_6
a_d_alpha_theta0_norm0 = [0 0.27 0.07 0 0 0 ... 
    0.29 0 0 0.302 0 0.072 ... 
   -1.571 0 -1.571 1.571 -1.571 0 ... 
    0 -1.57 0 0 0 0]';

% Calibration point (P0) chosen experimentally
P0 = [0.24500 -0.45600 0.00665]';

% Nominal parameters initialization
a_d_alpha_theta0_norm = a_d_alpha_theta0_norm0;

%%------------------------------------------
% Input Data
%%------------------------------------------
% Data1: training/calibration dataset; Data2: testing dataset
Data1 = xlsread('Data.xlsx',1,'D2:J112');
Data2 = xlsread('Data.xlsx',2,'D2:J22');

% U: joint angles in degrees (convert to radians)
U = Data1(:, 1:6) * pi/180;
% MM1: measured line lengths (convert mm to m)
MM1 = Data1(:, 7) / 1000;

%%------------------------------------------
% Robot Configuration from the Dataset
%%------------------------------------------
q1_batch = U(:, 1);
q2_batch = U(:, 2);
q3_batch = U(:, 3);
q4_batch = U(:, 4);
q5_batch = U(:, 5);
q6_batch = U(:, 6);

%%------------------------------------------
% PSO Optimization for Calibration Parameters
%%------------------------------------------
% Optimize the 24 calibration parameters (a_d_alpha_theta0_norm) using a PSO
% algorithm that minimizes the cost returned by objective function fh.
% (Ensure that fh, my_forward, and my_Jacobian are implemented and on the MATLAB path.)

% PSO Parameters
num_particles = 30;    % Number of particles in the swarm
max_iter      = 50;    % Maximum number of iterations
w  = 0.7;              % Inertia weight
c1 = 1.5;              % Cognitive (personal) acceleration coefficient
c2 = 1.5;              % Social (global) acceleration coefficient

dim = length(a_d_alpha_theta0_norm);  % 24 parameters

% Define search bounds based on the nominal value ¡À10%
lb = a_d_alpha_theta0_norm - 0.1 * abs(a_d_alpha_theta0_norm);
ub = a_d_alpha_theta0_norm + 0.1 * abs(a_d_alpha_theta0_norm);
% For any parameter that is zero, assign a small range:
for i = 1:dim
    if a_d_alpha_theta0_norm(i) == 0
        lb(i) = -0.1;
        ub(i) = 0.1;
    end
end

% Initialize particle positions and velocities
positions  = zeros(dim, num_particles);
velocities = zeros(dim, num_particles);
pbest      = zeros(dim, num_particles);
pbest_fitness = inf(1, num_particles);

for j = 1:num_particles
    positions(:, j) = lb + (ub - lb) .* rand(dim, 1);  % Random initialization within bounds
    velocities(:, j) = zeros(dim, 1);
    % Evaluate fitness of the initial particle position
    pbest_fitness(j) = fh(positions(:, j), Data1, P0);
    pbest(:, j) = positions(:, j);
end

% Identify the global best particle:
[global_best_fitness, best_idx] = min(pbest_fitness);
global_best = pbest(:, best_idx);

% Record fitness history for visualization
fitness_history = zeros(max_iter, 1);

disp('Starting PSO optimization...');
for iter = 1:max_iter
    for j = 1:num_particles
        % Generate random numbers for the stochastic updates:
        r1 = rand(dim, 1);
        r2 = rand(dim, 1);
        % Update velocity for particle j:
        velocities(:, j) = w * velocities(:, j) ...
                           + c1 * r1 .* (pbest(:, j) - positions(:, j)) ...
                           + c2 * r2 .* (global_best - positions(:, j));
        % Update the particle position:
        positions(:, j) = positions(:, j) + velocities(:, j);
        % Enforce boundaries:
        positions(:, j) = max(positions(:, j), lb);
        positions(:, j) = min(positions(:, j), ub);
        
        % Evaluate fitness at the new position:
        current_fitness = fh(positions(:, j), Data1, P0);
        % Update the personal best if current fitness is better:
        if current_fitness < pbest_fitness(j)
            pbest(:, j) = positions(:, j);
            pbest_fitness(j) = current_fitness;
        end
        % Update the global best if needed:
        if current_fitness < global_best_fitness
            global_best = positions(:, j);
            global_best_fitness = current_fitness;
        end
    end
    fitness_history(iter) = global_best_fitness;
    disp(['Iteration ' num2str(iter) ', Best Fitness = ' num2str(global_best_fitness)]);
end

% Update the nominal parameters with the best solution from PSO:
a_d_alpha_theta0_norm = global_best;

disp('PSO optimization completed.');
disp('Calibrated Parameters:');
disp(a_d_alpha_theta0_norm);

%%------------------------------------------
% Data Visualization: Plot fitness history
%%------------------------------------------
figure(1), clf,
semilogy(1:max_iter, fitness_history, 'rx-')
xlabel('Iteration number');
ylabel('Error (mm)');
title('PSO Optimization Convergence');

%%------------------------------------------
% Testing Phase: Evaluate the Calibration Results
%%------------------------------------------
% Single test case
q_test = [-67.7 24.7 -14.5 -14.9 75.1 -54.4]' * pi/180;
L_ex11_test = 0.4845;  % Measured value for the test case (in meters)
L_norm1_test = my_forward(a_d_alpha_theta0_norm, q_test, P0);
J_norm1_test = my_Jacobian(a_d_alpha_theta0_norm, q_test, P0);
Error_test = L_norm1_test - L_ex11_test;
disp(['Position error for single test case: ', num2str(Error_test')]);

% Testing on the entire training set (Data1)
MSE = 0;
E_max2 = 0;
for i = 1:length(q1_batch)
    q = [q1_batch(i); q2_batch(i); q3_batch(i); q4_batch(i); q5_batch(i); q6_batch(i)];
    L_ex11 = MM1(i);
    L_norm1 = my_forward(a_d_alpha_theta0_norm, q, P0);
    Error = L_norm1 - L_ex11;
    MSE = MSE + (norm(Error))^2;
    E_max2 = max((norm(Error))^2, E_max2);
end
MSE = MSE / length(q1_batch);
RMSE = sqrt(MSE);
E_max2 = sqrt(E_max2);
disp(['Testing RMSE: ', num2str(RMSE)]);
disp(['Max Error: ', num2str(E_max2)]);

toc


%%======================================================================================
% IMPORTANT:
% Ensure that the following functions are implemented and accessible:
%
% 1. my_forward(params, q, P0)
%    - Computes the forward kinematics (line length, pose, etc.) using the calibration
%      parameters, joint angles vector q, and calibration point P0.
%
% 2. my_Jacobian(params, q, P0)
%    - Computes the Jacobian matrix for the calibration process.
%
% 3. fh(params, Data, P0)
%    - Computes a cost (error) metric over the calibration dataset.
%      (For example, see the implementation given in fh.m.)
%%======================================================================================