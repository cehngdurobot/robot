%% 基于 LSTM 的位置误差补偿方法（使用 SGD 优化）
% 参考文献:
% Chen et al. (2019), Meas. Sci. Technol. 30 125010
%
% 说明：
% 本示例将每个机器人关节角构成一个时间序列（共6步，每步1维），
% 利用 LSTM 模型对实际测量误差进行拟合，从而实现误差补偿。
% 网络结构：1层 LSTM（隐藏层维数 H=10） + 1层全连接层输出（标量输出）。
% 使用 SGD 对所有参数进行更新。

tic;
rng(20, 'twister');  % 固定随机数种子
clear; clc;

%% ------基本参数设定----------------------------------------------------------
% 原始 DH 参数（示例，24×1 列向量；请根据实际情况修改）
a_d_alpha_theta0_norm0 = [...
     0,     0.27, 0.07,  0,     0,     0,  ...
     0.29,  0,    0,     0.302, 0,     0.072, ...
    -1.571,  0,   -1.571, 1.571,-1.571,  0,   ...
     0,    -1.57, 0,     0,     0,     0]';
P0 = [0.24500; -0.45600; 0.00665];   % 标定点（用于正向运动学）
baseDH = a_d_alpha_theta0_norm0;      % 基础 DH 参数

% 从 Excel 中读取数据
Data1 = xlsread('Data.xlsx', 1, 'D2:J112');  % 训练数据
Data2 = xlsread('Data.xlsx', 2, 'D2:J22');     % 测试数据

%% ------数据预处理----------------------------------------------------------
% 训练数据：前6列关节角（单位：度），转换为弧度后视为一个序列，每个样本长度 T=6，
% 第7列为实际测量距离（单位：mm，转换为 m）
U_train = Data1(:, 1:6) * pi/180;   % 尺度： m_train x 6
U_test  = Data2(:, 1:6) * pi/180;    
m_train = size(U_train,1);
m_test  = size(U_test,1);

% 计算 nominal 输出（利用 DH 正向运动学函数）
L_nom_train = zeros(m_train,1);
for i = 1:m_train
    q = U_train(i,:)';
    L_nom_train(i) = DH(baseDH, q, P0);
end

% 转换测量值（单位：mm 转 m）
e_true_train = Data1(:,7) / 1000;
% 训练目标：真实误差 = 测量值 - nominal 输出
e_train = e_true_train - L_nom_train;

%% ------初始化 LSTM 网络参数-----------------------------------------------
% 输入：1维；时间步 T=6；隐藏层维数 H=10
H = 10; T = 6;  % T 为序列长度（关节数）
% 初始化 LSTM 权重（均较小随机值）
params.Wf = randn(H, 1)*0.01;  % 尺寸 H x 1
params.Uf = randn(H, H)*0.01;  % 尺寸 H x H
params.bf = zeros(H,1);        % 尺寸 H x 1

params.Wi = randn(H, 1)*0.01;
params.Ui = randn(H, H)*0.01;
params.bi = zeros(H,1);

params.Wo = randn(H, 1)*0.01;
params.Uo = randn(H, H)*0.01;
params.bo = zeros(H,1);

params.Wc = randn(H, 1)*0.01;
params.Uc = randn(H, H)*0.01;
params.bc = zeros(H,1);

% 输出层参数：将隐藏层最后时刻输出 h_T (H×1) 映射为标量
params.Wy = randn(1, H)*0.01;   % 尺寸 1 x H
params.by = 0;

%% ------SGD 优化 LSTM 网络参数 ---------------------------------------------
epochs = 50;  
lr = 0.01;  % 学习率（所有参数统一使用，可以根据需要分开设定）
loss_history = zeros(epochs,1);

fprintf('\n开始使用 SGD 优化 LSTM 参数：\n');
for epoch = 1:epochs
    idxs = randperm(m_train);  % 随机打乱训练样本顺序
    total_loss = 0;
    for idx = idxs
        % 对于每个样本，将关节角序列视为 T×1 向量
        x_seq = U_train(idx, :)';  % 6×1
        target = e_train(idx);     % 标量目标
        
        % 前向传播：计算 LSTM 输出（预测补偿误差）
        [y_pred, cache] = lstmForward(x_seq, params);
        
        % 损失与梯度（损失：0.5*(y_pred - target)^2）
        loss = 0.5 * (y_pred - target)^2;
        total_loss = total_loss + loss;
        dy = (y_pred - target);  % d(loss)/dy_pred
        
        % 反向传播：计算梯度（使用 BPTT）
        grads = lstmBackward(dy, cache, params);
        
        % SGD 更新所有参数：
        params.Wf = params.Wf - lr * grads.Wf;
        params.Uf = params.Uf - lr * grads.Uf;
        params.bf = params.bf - lr * grads.bf;
        
        params.Wi = params.Wi - lr * grads.Wi;
        params.Ui = params.Ui - lr * grads.Ui;
        params.bi = params.bi - lr * grads.bi;
        
        params.Wo = params.Wo - lr * grads.Wo;
        params.Uo = params.Uo - lr * grads.Uo;
        params.bo = params.bo - lr * grads.bo;
        
        params.Wc = params.Wc - lr * grads.Wc;
        params.Uc = params.Uc - lr * grads.Uc;
        params.bc = params.bc - lr * grads.bc;
        
        params.Wy = params.Wy - lr * grads.Wy;
        params.by = params.by - lr * grads.by;
    end
    loss_history(epoch) = total_loss / m_train;
    fprintf('Epoch %d, Loss = %f\n', epoch, loss_history(epoch));
end

%% ------利用训练好的 LSTM 计算补偿后的结果------------------------------
% 补偿后训练集输出： nominal 输出 + LSTM 预测的误差补偿
L_comp_train = zeros(m_train, 1);
for i = 1:m_train
    q = U_train(i, :)';
    L_nom_train(i) = DH(baseDH, q, P0);
    % LSTM 预测：将关节角序列作为输入
    [pred_error, ~] = lstmForward(U_train(i, :)', params);
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
    [pred_error, ~] = lstmForward(U_test(i,:)', params);
    L_comp_test(i) = L_nom_test(i) + pred_error;
end
RMSE_test = sqrt(mean((Data2(:,7)/1000 - L_comp_test).^2));
fprintf('测试集补偿后 RMSE = %f m\n', RMSE_test);

%% ------绘制训练过程中的损失变化曲线-----------------------------------------
figure;
plot(1:epochs, loss_history, 'b-o','LineWidth',1.5);
xlabel('Epoch');
ylabel('Training Loss');
title('LSTM 使用 SGD 优化过程中的训练损失');
grid on;

%% ===== 辅助函数部分 =====

%% LSTM 前向传播函数
function [y, cache] = lstmForward(x, params)
% 输入:
%   x      : 输入序列，T×1 (T=6，本例中为关节角序列)
%   params : 网络参数结构体
% 输出:
%   y      : 网络输出(标量)，预测补偿误差
%   cache  : 存储每个时间步的中间变量，用于反向传播

T = length(x);         % 序列长度
H = size(params.Wf,1);   % 隐藏层维数

% 初始化隐藏状态和细胞状态
h{1} = zeros(H,1);  % h0
c{1} = zeros(H,1);  % c0

% 存储中间变量
cache.x = cell(T,1);
cache.f = cell(T,1);
cache.i = cell(T,1);
cache.o = cell(T,1);
cache.g = cell(T,1);
cache.h = cell(T+1,1);  % h{1}...h{T+1}
cache.c = cell(T+1,1);  % c{1}...c{T+1}
cache.h{1} = h{1};
cache.c{1} = c{1};

for t = 1:T
    cache.x{t} = x(t);
    % 计算各个门
    a_f = params.Wf * x(t) + params.Uf * cache.h{t} + params.bf;
    f_t = sigmoid(a_f);
    
    a_i = params.Wi * x(t) + params.Ui * cache.h{t} + params.bi;
    i_t = sigmoid(a_i);
    
    a_o = params.Wo * x(t) + params.Uo * cache.h{t} + params.bo;
    o_t = sigmoid(a_o);
    
    a_c = params.Wc * x(t) + params.Uc * cache.h{t} + params.bc;
    g_t = tanh(a_c);
    
    % 更新细胞状态
    c_t = f_t .* cache.c{t} + i_t .* g_t;
    % 更新隐藏状态
    h_t = o_t .* tanh(c_t);
    
    % 保存当前时刻的中间变量
    cache.f{t} = f_t;
    cache.i{t} = i_t;
    cache.o{t} = o_t;
    cache.g{t} = g_t;
    cache.c{t+1} = c_t;
    cache.h{t+1} = h_t;
end

% 最后时刻隐藏状态 h_T 为 h{T+1}
h_T = cache.h{T+1};

% 全连接输出层
y = params.Wy * h_T + params.by;
cache.T = T;  % 序列长度
cache.h_final = h_T;
end

%% LSTM 反向传播函数（BPTT）
function grads = lstmBackward(dy, cache, params)
% 输入:
%   dy     : 损失对输出 y 的梯度 (标量)
%   cache  : 前向传播存储的中间变量
%   params : 网络参数结构体
% 输出:
%   grads  : 与 params 结构体对应的梯度

T = cache.T;
H = size(params.Wf,1);

% 初始化梯度累计
dWf = zeros(size(params.Wf)); dUf = zeros(size(params.Uf)); dbf = zeros(size(params.bf));
dWi = zeros(size(params.Wi)); dUi = zeros(size(params.Ui)); dbi = zeros(size(params.bi));
dWo = zeros(size(params.Wo)); dUo = zeros(size(params.Uo)); dbo = zeros(size(params.bo));
dWc = zeros(size(params.Wc)); dUc = zeros(size(params.Uc)); dbc = zeros(size(params.bc));
dWy = zeros(size(params.Wy)); dby = 0;

% 输出层梯度
h_final = cache.h{T+1};
dWy = dy * h_final';
dby = dy;
% 初始梯度传递到最后时间步
dh_next = params.Wy' * dy;
dc_next = zeros(H,1);

% 反向传播遍历每个时间步 t = T...1
for t = T:-1:1
    h_t   = cache.h{t+1};
    c_t   = cache.c{t+1};
    h_prev = cache.h{t};
    c_prev = cache.c{t};
    x_t = cache.x{t};
    
    f_t = cache.f{t};
    i_t = cache.i{t};
    o_t = cache.o{t};
    g_t = cache.g{t};
    
    % 对 h_t 和 c_t 的梯度
    dh = dh_next;  % 来自上层的隐藏状态梯度
    do = dh .* tanh(c_t) .* o_t .* (1 - o_t);  % sigmoid 梯度
    % 梯度通过 tanh(c_t)
    dtanh_c = (1 - tanh(c_t).^2);
    dc = dh .* o_t .* dtanh_c + dc_next;
    
    df = dc .* c_prev .* f_t .* (1 - f_t);
    di = dc .* g_t    .* i_t .* (1 - i_t);
    dg = dc .* i_t    .* (1 - g_t.^2);
    
    % 累计各个门的参数梯度
    dWf = dWf + df * x_t;
    dUf = dUf + df * h_prev';
    dbf = dbf + df;
    
    dWi = dWi + di * x_t;
    dUi = dUi + di * h_prev';
    dbi = dbi + di;
    
    dWo = dWo + do * x_t;
    dUo = dUo + do * h_prev';
    dbo = dbo + do;
    
    dWc = dWc + dg * x_t;
    dUc = dUc + dg * h_prev';
    dbc = dbc + dg;
    
    % 传递梯度到前一隐藏状态
    dh_prev = params.Uf' * df + params.Ui' * di + params.Uo' * do + params.Uc' * dg;
    dc_prev = dc .* f_t;
    
    dh_next = dh_prev;
    dc_next = dc_prev;
end

% 收集梯度
grads.Wf = dWf;
grads.Uf = dUf;
grads.bf = dbf;
grads.Wi = dWi;
grads.Ui = dUi;
grads.bi = dbi;
grads.Wo = dWo;
grads.Uo = dUo;
grads.bo = dbo;
grads.Wc = dWc;
grads.Uc = dUc;
grads.bc = dbc;
grads.Wy = dWy;
grads.by = dby;
end

%% Sigmoid 函数及其导数
function y = sigmoid(x)
    y = 1./(1+exp(-x));
end

function y = dsigmoid(x)
    s = sigmoid(x);
    y = s .* (1 - s);
end
