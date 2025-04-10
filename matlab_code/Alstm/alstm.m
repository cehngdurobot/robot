

tic;
rng(2023, 'twister');  
clear; clc;


a_d_alpha_theta0_norm0 = [...
     0,     0.27, 0.07,  0,     0,     0,  ...
     0.29,  0,    0,     0.302, 0,     0.072, ...
    -1.571,  0,   -1.571, 1.571,-1.571,  0,   ...
     0,    -1.57, 0,     0,     0,     0]';
P0 = [0.24500; -0.45600; 0.00665];   % �궨�㣨���������˶�ѧ��
baseDH = a_d_alpha_theta0_norm0;      % ���� DH ����


Data1 = xlsread('Data.xlsx', 1, 'D2:J112');  % ѵ������
Data2 = xlsread('Data.xlsx', 2, 'D2:J22');     % ��������

%% ------����Ԥ����----------------------------------------------------------
% ѵ�����ݣ�ǰ6��Ϊ�ؽڽǣ���λ���ȣ���ת��Ϊ���ȣ���7��Ϊ����ֵ����λ��mm -> m��
U_train = Data1(:, 1:6) * pi/180;   % ѵ�������ߴ� m_train x 6
U_test  = Data2(:, 1:6) * pi/180;    % ���Լ�
m_train = size(U_train,1);
m_test  = size(U_test,1);

% ����ѵ���� nominal ��������� DH �����˶�ѧ������
L_nom_train = zeros(m_train,1);
for i = 1:m_train
    q = U_train(i,:)';
    L_nom_train(i) = DH(baseDH, q, P0);
end

% ������ֵת��Ϊ m�����赥λΪ mm��
e_true_train = Data1(:,7) / 1000;
% ѵ��Ŀ�꣺��ʵ��� = ����ֵ - nominal ���
e_train = e_true_train - L_nom_train;

%% ------���� LSTM ����ṹ�����ά��-------------------------------
% ��ÿ�������� 6 ���ؽڽ���Ϊ���У�T=6������ά�� =1����
% LSTM ���ز�ά����Ϊ H = 10.
H = 1; T = 6;  
% ��������������ά�� D�������� D = 491.
D = 491;

%% ------PSO �����趨--------------------------------------------
pop_size = 50;     % ������
max_iter = 50;     % ����������
w_inertia = 0.76;   % ����Ȩ��
c1 = 1.5;          % ����ѧϰ����
c2 = 1.5;          % ȫ��ѧϰ����

% ��ʼ������λ�ã�ÿ������Ϊ 1��D ���������ٶ�
% ���ý�С�����ʼ�������ֵ0������0.01��
X = 0.01* randn(pop_size, D);
V = 0.01 * randn(pop_size, D);

% ��ʼ���������ż���Ӧ��ֵ
pbest = X;
pbest_fit = zeros(pop_size,1);
for i = 1:pop_size
    pbest_fit(i) = psoObjLSTM(X(i,:)', U_train, e_train, H);
end

% ��ʼ��ȫ������
[global_fit, idx] = min(pbest_fit);
gbest = X(idx,:)';

% ����ÿ�ε���ȫ�������Ӧ��
fit_history = zeros(max_iter,1);

%% ------PSO ��ѭ��--------------------------------------------
fprintf('\n��ʼʹ�� PSO �Ż� LSTM ����Ȩ�أ�\n');
for iter = 1:max_iter
    for i = 1:pop_size
        % �����ٶ���λ��
        r1 = rand(D,1); r2 = rand(D,1);
        V(i,:) = w_inertia * V(i,:)' + c1 * r1 .* (pbest(i,:)' - X(i,:)') + c2 * r2 .* (gbest - X(i,:)');
        X(i,:) = X(i,:) + V(i,:);
        
        % ���㵱ǰ��Ӧ�ȣ�Ŀ�꣺ƽ����ʧ��
        current_fit = psoObjLSTM(X(i,:)', U_train, e_train, H);
        % ���¸�������
        if current_fit < pbest_fit(i)
            pbest(i,:) = X(i,:);
            pbest_fit(i) = current_fit;
        end
        % ����ȫ������
        if current_fit < global_fit
            global_fit = current_fit;
            gbest = X(i,:)';
        end
    end
    fit_history(iter) = global_fit;
    fprintf('Iteration %d, Best Fit = %f\n', iter, global_fit);
end

%% ------�����Ż��õ��� LSTM ����Ԥ�ⲹ�����--------------------------
% �� gbest ���������Ϊ��������ṹ
opt_params = unpackLSTMParams(gbest, H);

% ������ѵ��������� nominal ��� + LSTM Ԥ�����
L_comp_train = zeros(m_train, 1);
for i = 1:m_train
    q = U_train(i,:)';
    L_nom_train(i) = DH(baseDH, q, P0);
    [pred_error, ~] = lstmForward(q, opt_params);
    L_comp_train(i) = L_nom_train(i) + pred_error;
end
RMSE_train = sqrt(mean((Data1(:,7)/1000 - L_comp_train).^2));
fprintf('\nѵ���������� RMSE = %f m\n', RMSE_train);

% ���Լ�����
L_comp_test = zeros(m_test,1);
L_nom_test = zeros(m_test,1);
for i = 1:m_test
    q = U_test(i,:)';
    L_nom_test(i) = DH(baseDH, q, P0);
    [pred_error, ~] = lstmForward(q, opt_params);
    L_comp_test(i) = L_nom_test(i) + pred_error;
end
RMSE_test = sqrt(mean((Data2(:,7)/1000 - L_comp_test).^2));
fprintf('���Լ������� RMSE = %f m\n', RMSE_test);

%% ------���� PSO ������������Ӧ����������-------------------------
figure;
plot(1:max_iter, fit_history, 'b-o','LineWidth',1.5);
xlabel('Iteration');
ylabel('Best Fitness');
title('PSO ������������Ӧ����������');
grid on;

toc;

%% ===== ������������ =====

%% PSOĿ�꺯����������ѡ��theta������LSTM��ѵ�����ϵ�ƽ����ʧ
function fitness = psoObjLSTM(theta, U_train, e_train, H)
    % ��theta�������Ϊ LSTM ����
    params = unpackLSTMParams(theta, H);
    m = size(U_train,1);
    loss_sum = 0;
    for i = 1:m
        x_seq = U_train(i,:)';  % ���г���T=6��1ά����
        [y_pred, ~] = lstmForward(x_seq, params);
        err = y_pred - e_train(i);
        loss_sum = loss_sum + 0.5*err^2;
    end
    fitness = loss_sum / m;  % ƽ����ʧ
end

%% ��LSTMȨ������theta���Ϊ�ṹ��params
function params = unpackLSTMParams(theta, H)
    % theta��һ����������ά��Ϊ D = 491.
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

%% LSTM ǰ�򴫲�����
function [y, cache] = lstmForward(x, params)
% ����:
%   x      : �������У�T��1�������� T=6��Ϊ�ؽڽ����У�
%   params : ��������ṹ��
% ���:
%   y      : ���������Ԥ�ⲹ����������
%   cache  : �洢ǰ�򴫲��е��м�����������ڵ��ԣ��˴��Ǳ��룩

T = length(x);         % ���г���
H = size(params.Wf,1);   % ���ز�ά��

% ��ʼ������״̬��ϸ��״̬
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
    % ������ż���
    a_f = params.Wf * x(t) + params.Uf * cache.h{t} + params.bf;
    f_t = sigmoid(a_f);
    
    a_i = params.Wi * x(t) + params.Ui * cache.h{t} + params.bi;
    i_t = sigmoid(a_i);
    
    a_o = params.Wo * x(t) + params.Uo * cache.h{t} + params.bo;
    o_t = sigmoid(a_o);
    
    a_c = params.Wc * x(t) + params.Uc * cache.h{t} + params.bc;
    g_t = tanh(a_c);
    
    % ����ϸ��״̬������״̬
    c_t = f_t .* cache.c{t} + i_t .* g_t;
    h_t = o_t .* tanh(c_t);
    
    cache.f{t} = f_t;
    cache.i{t} = i_t;
    cache.o{t} = o_t;
    cache.g{t} = g_t;
    cache.c{t+1} = c_t;
    cache.h{t+1} = h_t;
end

h_T = cache.h{T+1};   % ���һʱ������״̬
y = params.Wy * h_T + params.by;
cache.T = T;
cache.h_final = h_T;
end

%% Sigmoid ����
function y = sigmoid(x)
    y = 1./(1+exp(-x));
end

