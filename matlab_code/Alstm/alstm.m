

tic;
rng(2023, 'twister');  
clear; clc;


a_d_alpha_theta0_norm0 = [...
     0,     0.27, 0.07,  0,     0,     0,  ...
     0.29,  0,    0,     0.302, 0,     0.072, ...
    -1.571,  0,   -1.571, 1.571,-1.571,  0,   ...
     0,    -1.57, 0,     0,     0,     0]';
P0 = [0.24500; -0.45600; 0.00665];   % 标定点（用于正向运动学）
baseDH = a_d_alpha_theta0_norm0;      % 基础 DH 参数


Data1 = xlsread('Data.xlsx', 1, 'D2:J112');  % 训练数据
Data2 = xlsread('Data.xlsx', 2, 'D2:J22');     % 测试数据

%% ------数据预处理----------------------------------------------------------
% 训练数据：前6列为关节角（单位：度），转换为弧度；第7列为测量值（单位：mm -> m）
U_train = Data1(:, 1:6) * pi/180;   % 训练集，尺寸 m_train x 6
U_test  = Data2(:, 1:6) * pi/180;    % 测试集
m_train = size(U_train,1);
m_test  = size(U_test,1);

% 计算训练集 nominal 输出（利用 DH 正向运动学函数）
L_nom_train = zeros(m_train,1);
for i = 1:m_train
    q = U_train(i,:)';
    L_nom_train(i) = DH(baseDH, q, P0);
end

% 将测量值转换为 m（假设单位为 mm）
e_true_train = Data1(:,7) / 1000;
% 训练目标：真实误差 = 测量值 - nominal 输出
e_train = e_true_train - L_nom_train;

%% ------设置 LSTM 网络结构与参数维数-------------------------------
% 将每个样本的 6 个关节角作为序列（T=6，输入维度 =1），
% LSTM 隐藏层维数设为 H = 10.
H = 1; T = 6;  
% 网络参数打包后总维数 D。经计算 D = 491.
D = 491;

%% ------PSO 参数设定--------------------------------------------
pop_size = 50;     % 粒子数
max_iter = 50;     % 最大迭代次数
w_inertia = 0.76;   % 惯性权重
c1 = 1.5;          % 个体学习因子
c2 = 1.5;          % 全局学习因子

% 初始化粒子位置（每个粒子为 1×D 向量）与速度
% 采用较小随机初始化（如均值0、方差0.01）
X = 0.01* randn(pop_size, D);
V = 0.01 * randn(pop_size, D);

% 初始化个体最优及适应度值
pbest = X;
pbest_fit = zeros(pop_size,1);
for i = 1:pop_size
    pbest_fit(i) = psoObjLSTM(X(i,:)', U_train, e_train, H);
end

% 初始化全局最优
[global_fit, idx] = min(pbest_fit);
gbest = X(idx,:)';

% 保存每次迭代全局最佳适应度
fit_history = zeros(max_iter,1);

%% ------PSO 主循环--------------------------------------------
fprintf('\n开始使用 PSO 优化 LSTM 网络权重：\n');
for iter = 1:max_iter
    for i = 1:pop_size
        % 更新速度与位置
        r1 = rand(D,1); r2 = rand(D,1);
        V(i,:) = w_inertia * V(i,:)' + c1 * r1 .* (pbest(i,:)' - X(i,:)') + c2 * r2 .* (gbest - X(i,:)');
        X(i,:) = X(i,:) + V(i,:);
        
        % 计算当前适应度（目标：平均损失）
        current_fit = psoObjLSTM(X(i,:)', U_train, e_train, H);
        % 更新个体最优
        if current_fit < pbest_fit(i)
            pbest(i,:) = X(i,:);
            pbest_fit(i) = current_fit;
        end
        % 更新全局最优
        if current_fit < global_fit
            global_fit = current_fit;
            gbest = X(i,:)';
        end
    end
    fit_history(iter) = global_fit;
    fprintf('Iteration %d, Best Fit = %f\n', iter, global_fit);
end

%% ------利用优化得到的 LSTM 网络预测补偿误差--------------------------
% 将 gbest 解向量解包为网络参数结构
opt_params = unpackLSTMParams(gbest, H);

% 补偿后训练集结果： nominal 输出 + LSTM 预测误差
L_comp_train = zeros(m_train, 1);
for i = 1:m_train
    q = U_train(i,:)';
    L_nom_train(i) = DH(baseDH, q, P0);
    [pred_error, ~] = lstmForward(q, opt_params);
    L_comp_train(i) = L_nom_train(i) + pred_error;
end
RMSE_train = sqrt(mean((Data1(:,7)/1000 - L_comp_train).^2));
fprintf('\n训练集补偿后 RMSE = %f m\n', RMSE_train);

% 测试集补偿
L_comp_test = zeros(m_test,1);
L_nom_test = zeros(m_test,1);
for i = 1:m_test
    q = U_test(i,:)';
    L_nom_test(i) = DH(baseDH, q, P0);
    [pred_error, ~] = lstmForward(q, opt_params);
    L_comp_test(i) = L_nom_test(i) + pred_error;
end
RMSE_test = sqrt(mean((Data2(:,7)/1000 - L_comp_test).^2));
fprintf('测试集补偿后 RMSE = %f m\n', RMSE_test);

%% ------绘制 PSO 迭代过程中适应度收敛曲线-------------------------
figure;
plot(1:max_iter, fit_history, 'b-o','LineWidth',1.5);
xlabel('Iteration');
ylabel('Best Fitness');
title('PSO 迭代过程中适应度收敛曲线');
grid on;

toc;

%% ===== 辅助函数部分 =====

%% PSO目标函数：给定候选解theta，计算LSTM在训练集上的平均损失
function fitness = psoObjLSTM(theta, U_train, e_train, H)
    % 将theta向量解包为 LSTM 参数
    params = unpackLSTMParams(theta, H);
    m = size(U_train,1);
    loss_sum = 0;
    for i = 1:m
        x_seq = U_train(i,:)';  % 序列长度T=6，1维输入
        [y_pred, ~] = lstmForward(x_seq, params);
        err = y_pred - e_train(i);
        loss_sum = loss_sum + 0.5*err^2;
    end
    fitness = loss_sum / m;  % 平均损失
end

%% 将LSTM权重向量theta解包为结构体params
function params = unpackLSTMParams(theta, H)
    % theta：一个列向量，维数为 D = 491.
    idx = 1;
    params.Wf = reshape(theta(idx: idx+H-1), [H, 1]); idx = idx + H;
    params.Uf = reshape(theta(idx: idx+H*H-1), [H, H]); idx = idx + H*H;
    params.bf = reshape(theta(idx: idx+H-1), [H, 1]); idx = idx + H;
    
    params.Wi = reshape(theta(idx: idx+H-1), [H, 1]); idx = idx + H;
    params.Ui = reshape(theta(idx: idx+H*H-1), [H, H]); idx = idx + H*H;
    params.bi = reshape(theta(idx: idx+H-1), [H, 1]); idx = idx + H;
    
    params.Wo = reshape(theta(idx: idx+H-1), [H, 1]); idx = idx + H;
    params.Uo = reshape(theta(idx: idx+H*H-1), [H, H]); idx = idx + H*H;
    params.bo = reshape(theta(idx: idx+H-1), [H, 1]); idx = idx + H;
    
    params.Wc = reshape(theta(idx: idx+H-1), [H, 1]); idx = idx + H;
    params.Uc = reshape(theta(idx: idx+H*H-1), [H, H]); idx = idx + H*H;
    params.bc = reshape(theta(idx: idx+H-1), [H, 1]); idx = idx + H;
    
    params.Wy = reshape(theta(idx: idx+H-1), [1, H]); idx = idx + H;
    params.by = theta(idx);
end

%% LSTM 前向传播函数
function [y, cache] = lstmForward(x, params)
% 输入:
%   x      : 输入序列，T×1（本例中 T=6，为关节角序列）
%   params : 网络参数结构体
% 输出:
%   y      : 网络输出（预测补偿误差），标量
%   cache  : 存储前向传播中的中间变量（仅用于调试，此处非必须）

T = length(x);         % 序列长度
H = size(params.Wf,1);   % 隐藏层维数

% 初始化隐藏状态与细胞状态
h{1} = zeros(H,1);  % h0
c{1} = zeros(H,1);  % c0

cache.x = cell(T,1);
cache.f = cell(T,1);
cache.i = cell(T,1);
cache.o = cell(T,1);
cache.g = cell(T,1);
cache.h = cell(T+1,1);
cache.c = cell(T+1,1);
cache.h{1} = h{1};
cache.c{1} = c{1};

for t = 1:T
    cache.x{t} = x(t);
    % 计算各门激活
    a_f = params.Wf * x(t) + params.Uf * cache.h{t} + params.bf;
    f_t = sigmoid(a_f);
    
    a_i = params.Wi * x(t) + params.Ui * cache.h{t} + params.bi;
    i_t = sigmoid(a_i);
    
    a_o = params.Wo * x(t) + params.Uo * cache.h{t} + params.bo;
    o_t = sigmoid(a_o);
    
    a_c = params.Wc * x(t) + params.Uc * cache.h{t} + params.bc;
    g_t = tanh(a_c);
    
    % 更新细胞状态与隐藏状态
    c_t = f_t .* cache.c{t} + i_t .* g_t;
    h_t = o_t .* tanh(c_t);
    
    cache.f{t} = f_t;
    cache.i{t} = i_t;
    cache.o{t} = o_t;
    cache.g{t} = g_t;
    cache.c{t+1} = c_t;
    cache.h{t+1} = h_t;
end

h_T = cache.h{T+1};   % 最后一时刻隐藏状态
y = params.Wy * h_T + params.by;
cache.T = T;
cache.h_final = h_T;
end

%% Sigmoid 函数
function y = sigmoid(x)
    y = 1./(1+exp(-x));
end

