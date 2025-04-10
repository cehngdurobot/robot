
tic;
rng(20, 'twister');  % �̶����������
clear; clc;

%% ------���������趨----------------------------------------------------------
% ԭʼ DH ������ʾ����24��1 �������������ʵ������޸ģ�
a_d_alpha_theta0_norm0 = [...
    0,    0.27, 0.07, 0,     0,     0,  ...
    0.29, 0,    0,    0.302, 0,     0.072, ...
    -1.571,0,   -1.571,1.571,-1.571,0,   ...
    0,   -1.57, 0,    0,     0,     0]';
P0 = [0.24500; -0.45600; 0.00665];   % �궨�㣬���������˶�ѧ

baseDH = a_d_alpha_theta0_norm0;  % ���� DH ����

% �� Excel �ж�ȡ����
Data1 = xlsread('Data.xlsx', 1, 'D2:J112');  % ѵ������
Data2 = xlsread('Data.xlsx', 2, 'D2:J22');     % ��������

%% ------����Ԥ����----------------------------------------------------------
% �ؽڽǣ�ǰ6�У�ת��Ϊ����
U_train = Data1(:, 1:6) * pi/180;   % ѵ��
U_test  = Data2(:, 1:6) * pi/180;    % ����

% ���� nominal ��������� DH �����˶�ѧ������  
m_train = size(U_train,1);
L_nom_train = zeros(m_train,1);
for i = 1:m_train
    q = U_train(i,:)';
    L_nom_train(i) = DH(baseDH, q, P0);
end

% ������ֵת��Ϊ m������ԭ��λΪ mm��
e_true_train = Data1(:,7)/1000;      % ����ֵ��ѵ����
% ��ʵ��ʵ�ʲ���ֵ�� nominal ���֮��
e_train = e_true_train - L_nom_train;

%% ------SGD�Ż� RBFNN ����----------------------------------------------------
% ������Ҫͬʱ�Ż������Ȩ�� w �͸�˹�˴��� sigma
% ����ṹ��ÿ��ѵ��������Ϊ���ز���Ԫ���ģ�����Ԫ�� m_train

% ��ʼ����Ȩ������ w (m_train �� 1) �� sigma (����)
w_rbf = randn(m_train,1)*0.01;   % �����ʼ����Сֵ
sigma = 0.1;                     % sigma ��ʼֵ

% ���ò�ͬ��ѧϰ�ʣ��ɸ���ʵ�����������
lr_w     = 0.01;   % Ȩ�ص�ѧϰ��
lr_sigma = 0.001;  % sigma ��ѧϰ��

epochs = 50;      % ѵ����������
loss_history = zeros(epochs,1);

fprintf('\n��ʼʹ�� SGD �Ż� RBFNN ������\n');
for epoch = 1:epochs
    % �������ѵ������˳��
    indices = randperm(m_train);
    for idx = indices
        % ������ idx���������ز�������
        % phi(idx,:) Ϊ 1��m_train ���������� j Ԫ��Ϊ��
        % phi_ij = exp( -||U_train(idx,:) - U_train(j,:)||^2/(2*sigma^2) )
        phi_i = zeros(1, m_train);
        for j = 1:m_train
            diff = U_train(idx,:) - U_train(j,:);
            phi_i(j) = exp( - (norm(diff)^2) / (2 * sigma^2) );
        end
        
        % Ԥ�����
        y_hat = phi_i * w_rbf;
        
        % ������������
        error_i = y_hat - e_train(idx);
        
        % �����ݶȣ��������Ȩ��
        % dL/dw_j = (error_i) * phi_ij
        grad_w = (error_i) * phi_i';
        
        % ����Ȩ��
        w_rbf = w_rbf - lr_w * grad_w;
        
        % ͬʱ���� sigma ���ݶȣ�
        % ����ÿ�� j�� dphi(i,j)/dsigma = phi(i,j)*(d_ij)/(sigma^3)
        % ���� d_ij = ||U_train(idx,:) - U_train(j,:)||^2
        grad_sigma = 0;
        for j = 1:m_train
            diff = U_train(idx,:) - U_train(j,:);
            d_ij = norm(diff)^2;
            phi_ij = phi_i(j);
            grad_sigma = grad_sigma + w_rbf(j) * phi_ij * d_ij;
        end
        grad_sigma = (error_i) * grad_sigma / (sigma^3);
        
        % ���� sigma����ȷ�� sigma>0
        sigma = sigma - lr_sigma * grad_sigma;
        if sigma <= 0
            sigma = 0.001;
        end
    end
    
    % ���㱾 epoch �ϵ�ѵ���� RMSE��ȫ����������Ԥ��ֵ��
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

%% ------����ѵ���õ� RBFNN ����ѵ�����Ͳ��Լ��Ĳ������------------------
% ѵ��������
L_comp_train = zeros(m_train,1);
for i = 1:m_train
    % nominal �����֪
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
fprintf('\nѵ���������� RMSE = %f m\n', RMSE_train);

% ���Լ�����
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
fprintf('���Լ������� RMSE = %f m\n', RMSE_test);

%% ------����ѵ�������е� RMSE �仯����-----------------------------------------
figure;
plot(1:epochs, loss_history, 'b-o','LineWidth',1.5);
xlabel('Epoch');
ylabel('Training RMSE (m)');
title('RBFNN ʹ�� SGD �Ż������е�ѵ�� RMSE');
grid on;

