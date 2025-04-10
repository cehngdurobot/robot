
tic;
rng(20, 'twister');  % 固定随机数种子
clear; clc;

%% ------基本参数设定----------------------------------------------------------
% 原始 DH 参数（示例，24×1 列向量；请根据实际情况修改）
a_d_alpha_theta0_norm0 = [...
    0,    0.27, 0.07, 0,     0,     0,  ...
    0.29, 0,    0,    0.302, 0,     0.072, ...
    -1.571,0,   -1.571,1.571,-1.571,0,   ...
    0,   -1.57, 0,    0,     0,     0]';
P0 = [0.24500; -0.45600; 0.00665];   % 标定点，用于正向运动学

baseDH = a_d_alpha_theta0_norm0;  % 基础 DH 参数

% 从 Excel 中读取数据
Data1 = xlsread('Data.xlsx', 1, 'D2:J112');  % 训练数据
Data2 = xlsread('Data.xlsx', 2, 'D2:J22');     % 测试数据

%% ------数据预处理----------------------------------------------------------
% 关节角（前6列）转换为弧度
U_train = Data1(:, 1:6) * pi/180;   % 训练
U_test  = Data2(:, 1:6) * pi/180;    % 测试

% 计算 nominal 输出（利用 DH 正向运动学函数）  
m_train = size(U_train,1);
L_nom_train = zeros(m_train,1);
for i = 1:m_train
    q = U_train(i,:)';
    L_nom_train(i) = DH(baseDH, q, P0);
end

% 将测量值转换为 m（假设原单位为 mm）
e_true_train = Data1(:,7)/1000;      % 测量值（训练）
% 真实误差：实际测量值与 nominal 输出之差
e_train = e_true_train - L_nom_train;

%% ------SGD优化 RBFNN 参数----------------------------------------------------
% 我们需要同时优化输出层权重 w 和高斯核带宽 sigma
% 网络结构：每个训练样本作为隐藏层神经元中心，总神经元数 m_train

% 初始化：权重向量 w (m_train × 1) 和 sigma (标量)
w_rbf = randn(m_train,1)*0.01;   % 随机初始化较小值
sigma = 0.1;                     % sigma 初始值

% 设置不同的学习率（可根据实际情况调整）
lr_w     = 0.01;   % 权重的学习率
lr_sigma = 0.001;  % sigma 的学习率

epochs = 50;      % 训练迭代轮数
loss_history = zeros(epochs,1);

fprintf('\n开始使用 SGD 优化 RBFNN 参数：\n');
for epoch = 1:epochs
    % 随机打乱训练样本顺序
    indices = randperm(m_train);
    for idx = indices
        % 对样本 idx，计算隐藏层的输出：
        % phi(idx,:) 为 1×m_train 的向量，其 j 元素为：
        % phi_ij = exp( -||U_train(idx,:) - U_train(j,:)||^2/(2*sigma^2) )
        phi_i = zeros(1, m_train);
        for j = 1:m_train
            diff = U_train(idx,:) - U_train(j,:);
            phi_i(j) = exp( - (norm(diff)^2) / (2 * sigma^2) );
        end
        
        % 预测输出
        y_hat = phi_i * w_rbf;
        
        % 计算误差（标量）
        error_i = y_hat - e_train(idx);
        
        % 计算梯度：对输出层权重
        % dL/dw_j = (error_i) * phi_ij
        grad_w = (error_i) * phi_i';
        
        % 更新权重
        w_rbf = w_rbf - lr_w * grad_w;
        
        % 同时计算 sigma 的梯度：
        % 对于每个 j， dphi(i,j)/dsigma = phi(i,j)*(d_ij)/(sigma^3)
        % 其中 d_ij = ||U_train(idx,:) - U_train(j,:)||^2
        grad_sigma = 0;
        for j = 1:m_train
            diff = U_train(idx,:) - U_train(j,:);
            d_ij = norm(diff)^2;
            phi_ij = phi_i(j);
            grad_sigma = grad_sigma + w_rbf(j) * phi_ij * d_ij;
        end
        grad_sigma = (error_i) * grad_sigma / (sigma^3);
        
        % 更新 sigma，并确保 sigma>0
        sigma = sigma - lr_sigma * grad_sigma;
        if sigma <= 0
            sigma = 0.001;
        end
    end
    
    % 计算本 epoch 上的训练集 RMSE（全量遍历计算预测值）
    predictions = zeros(m_train,1);
    for i = 1:m_train
        phi_i = zeros(1, m_train);
        for j = 1:m_train
            diff = U_train(i,:) - U_train(j,:);
            phi_i(j) = exp( - (norm(diff)^2)/(2*sigma^2) );
        end
        predictions(i) = phi_i * w_rbf;
    end
    epoch_RMSE = sqrt(mean((predictions - e_train).^2));
    loss_history(epoch) = epoch_RMSE;
    fprintf('Epoch %d, RMSE = %f, sigma = %f\n', epoch, epoch_RMSE, sigma);
end

%% ------利用训练好的 RBFNN 计算训练集和测试集的补偿结果------------------
% 训练集补偿
L_comp_train = zeros(m_train,1);
for i = 1:m_train
    % nominal 输出已知
    q = U_train(i,:)';
    L_nom_train(i) = DH(baseDH, q, P0);
    
    phi_i = zeros(1, m_train);
    for j = 1:m_train
        diff = U_train(i,:) - U_train(j,:);
        phi_i(j) = exp( - (norm(diff)^2)/(2*sigma^2) );
    end
    e_hat = phi_i * w_rbf;
    L_comp_train(i) = L_nom_train(i) + e_hat;
end
RMSE_train = sqrt(mean((Data1(:,7)/1000 - L_comp_train).^2));
fprintf('\n训练集补偿后 RMSE = %f m\n', RMSE_train);

% 测试集补偿
m_test = size(U_test,1);
L_nom_test = zeros(m_test,1);
e_hat_test = zeros(m_test,1);
for i = 1:m_test
    q = U_test(i,:)';
    L_nom_test(i) = DH(baseDH, q, P0);
    
    sum_val = 0;
    for j = 1:m_train
       diff = U_test(i,:) - U_train(j,:);
       sum_val = sum_val + w_rbf(j) * exp( - (norm(diff)^2)/(2*sigma^2) );
    end
    e_hat_test(i) = sum_val;
end
L_comp_test = L_nom_test + e_hat_test;
RMSE_test = sqrt(mean((Data2(:,7)/1000 - L_comp_test).^2));
fprintf('测试集补偿后 RMSE = %f m\n', RMSE_test);

%% ------绘制训练过程中的 RMSE 变化曲线-----------------------------------------
figure;
plot(1:epochs, loss_history, 'b-o','LineWidth',1.5);
xlabel('Epoch');
ylabel('Training RMSE (m)');
title('RBFNN 使用 SGD 优化过程中的训练 RMSE');
grid on;

