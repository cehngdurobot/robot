%% ���� LSTM ��λ������������ʹ�� SGD �Ż���
% �ο�����:
% Chen et al. (2019), Meas. Sci. Technol. 30 125010
%
% ˵����
% ��ʾ����ÿ�������˹ؽڽǹ���һ��ʱ�����У���6����ÿ��1ά����
% ���� LSTM ģ�Ͷ�ʵ�ʲ�����������ϣ��Ӷ�ʵ��������
% ����ṹ��1�� LSTM�����ز�ά�� H=10�� + 1��ȫ���Ӳ�����������������
% ʹ�� SGD �����в������и��¡�

tic;
rng(20, 'twister');  % �̶����������
clear; clc;

%% ------���������趨----------------------------------------------------------
% ԭʼ DH ������ʾ����24��1 �������������ʵ������޸ģ�
a_d_alpha_theta0_norm0 = [...
     0,     0.27, 0.07,  0,     0,     0,  ...
     0.29,  0,    0,     0.302, 0,     0.072, ...
    -1.571,  0,   -1.571, 1.571,-1.571,  0,   ...
     0,    -1.57, 0,     0,     0,     0]';
P0 = [0.24500; -0.45600; 0.00665];   % �궨�㣨���������˶�ѧ��
baseDH = a_d_alpha_theta0_norm0;      % ���� DH ����

% �� Excel �ж�ȡ����
Data1 = xlsread('Data.xlsx', 1, 'D2:J112');  % ѵ������
Data2 = xlsread('Data.xlsx', 2, 'D2:J22');     % ��������

%% ------����Ԥ����----------------------------------------------------------
% ѵ�����ݣ�ǰ6�йؽڽǣ���λ���ȣ���ת��Ϊ���Ⱥ���Ϊһ�����У�ÿ���������� T=6��
% ��7��Ϊʵ�ʲ������루��λ��mm��ת��Ϊ m��
U_train = Data1(:, 1:6) * pi/180;   % �߶ȣ� m_train x 6
U_test  = Data2(:, 1:6) * pi/180;    
m_train = size(U_train,1);
m_test  = size(U_test,1);

% ���� nominal ��������� DH �����˶�ѧ������
L_nom_train = zeros(m_train,1);
for i = 1:m_train
    q = U_train(i,:)';
    L_nom_train(i) = DH(baseDH, q, P0);
end

% ת������ֵ����λ��mm ת m��
e_true_train = Data1(:,7) / 1000;
% ѵ��Ŀ�꣺��ʵ��� = ����ֵ - nominal ���
e_train = e_true_train - L_nom_train;

%% ------��ʼ�� LSTM �������-----------------------------------------------
% ���룺1ά��ʱ�䲽 T=6�����ز�ά�� H=10
H = 10; T = 6;  % T Ϊ���г��ȣ��ؽ�����
% ��ʼ�� LSTM Ȩ�أ�����С���ֵ��
params.Wf = randn(H, 1)*0.01;  % �ߴ� H x 1
params.Uf = randn(H, H)*0.01;  % �ߴ� H x H
params.bf = zeros(H,1);        % �ߴ� H x 1

params.Wi = randn(H, 1)*0.01;
params.Ui = randn(H, H)*0.01;
params.bi = zeros(H,1);

params.Wo = randn(H, 1)*0.01;
params.Uo = randn(H, H)*0.01;
params.bo = zeros(H,1);

params.Wc = randn(H, 1)*0.01;
params.Uc = randn(H, H)*0.01;
params.bc = zeros(H,1);

% ���������������ز����ʱ����� h_T (H��1) ӳ��Ϊ����
params.Wy = randn(1, H)*0.01;   % �ߴ� 1 x H
params.by = 0;

%% ------SGD �Ż� LSTM ������� ---------------------------------------------
epochs = 50;  
lr = 0.01;  % ѧϰ�ʣ����в���ͳһʹ�ã����Ը�����Ҫ�ֿ��趨��
loss_history = zeros(epochs,1);

fprintf('\n��ʼʹ�� SGD �Ż� LSTM ������\n');
for epoch = 1:epochs
    idxs = randperm(m_train);  % �������ѵ������˳��
    total_loss = 0;
    for idx = idxs
        % ����ÿ�����������ؽڽ�������Ϊ T��1 ����
        x_seq = U_train(idx, :)';  % 6��1
        target = e_train(idx);     % ����Ŀ��
        
        % ǰ�򴫲������� LSTM �����Ԥ�ⲹ����
        [y_pred, cache] = lstmForward(x_seq, params);
        
        % ��ʧ���ݶȣ���ʧ��0.5*(y_pred - target)^2��
        loss = 0.5 * (y_pred - target)^2;
        total_loss = total_loss + loss;
        dy = (y_pred - target);  % d(loss)/dy_pred
        
        % ���򴫲��������ݶȣ�ʹ�� BPTT��
        grads = lstmBackward(dy, cache, params);
        
        % SGD �������в�����
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

%% ------����ѵ���õ� LSTM ���㲹����Ľ��------------------------------
% ������ѵ��������� nominal ��� + LSTM Ԥ�������
L_comp_train = zeros(m_train, 1);
for i = 1:m_train
    q = U_train(i, :)';
    L_nom_train(i) = DH(baseDH, q, P0);
    % LSTM Ԥ�⣺���ؽڽ�������Ϊ����
    [pred_error, ~] = lstmForward(U_train(i, :)', params);
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
    [pred_error, ~] = lstmForward(U_test(i,:)', params);
    L_comp_test(i) = L_nom_test(i) + pred_error;
end
RMSE_test = sqrt(mean((Data2(:,7)/1000 - L_comp_test).^2));
fprintf('���Լ������� RMSE = %f m\n', RMSE_test);

%% ------����ѵ�������е���ʧ�仯����-----------------------------------------
figure;
plot(1:epochs, loss_history, 'b-o','LineWidth',1.5);
xlabel('Epoch');
ylabel('Training Loss');
title('LSTM ʹ�� SGD �Ż������е�ѵ����ʧ');
grid on;

%% ===== ������������ =====

%% LSTM ǰ�򴫲�����
function [y, cache] = lstmForward(x, params)
% ����:
%   x      : �������У�T��1 (T=6��������Ϊ�ؽڽ�����)
%   params : ��������ṹ��
% ���:
%   y      : �������(����)��Ԥ�ⲹ�����
%   cache  : �洢ÿ��ʱ�䲽���м���������ڷ��򴫲�

T = length(x);         % ���г���
H = size(params.Wf,1);   % ���ز�ά��

% ��ʼ������״̬��ϸ��״̬
h{1} = zeros(H,1);  % h0
c{1} = zeros(H,1);  % c0

% �洢�м����
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
    % ���������
    a_f = params.Wf * x(t) + params.Uf * cache.h{t} + params.bf;
    f_t = sigmoid(a_f);
    
    a_i = params.Wi * x(t) + params.Ui * cache.h{t} + params.bi;
    i_t = sigmoid(a_i);
    
    a_o = params.Wo * x(t) + params.Uo * cache.h{t} + params.bo;
    o_t = sigmoid(a_o);
    
    a_c = params.Wc * x(t) + params.Uc * cache.h{t} + params.bc;
    g_t = tanh(a_c);
    
    % ����ϸ��״̬
    c_t = f_t .* cache.c{t} + i_t .* g_t;
    % ��������״̬
    h_t = o_t .* tanh(c_t);
    
    % ���浱ǰʱ�̵��м����
    cache.f{t} = f_t;
    cache.i{t} = i_t;
    cache.o{t} = o_t;
    cache.g{t} = g_t;
    cache.c{t+1} = c_t;
    cache.h{t+1} = h_t;
end

% ���ʱ������״̬ h_T Ϊ h{T+1}
h_T = cache.h{T+1};

% ȫ���������
y = params.Wy * h_T + params.by;
cache.T = T;  % ���г���
cache.h_final = h_T;
end

%% LSTM ���򴫲�������BPTT��
function grads = lstmBackward(dy, cache, params)
% ����:
%   dy     : ��ʧ����� y ���ݶ� (����)
%   cache  : ǰ�򴫲��洢���м����
%   params : ��������ṹ��
% ���:
%   grads  : �� params �ṹ���Ӧ���ݶ�

T = cache.T;
H = size(params.Wf,1);

% ��ʼ���ݶ��ۼ�
dWf = zeros(size(params.Wf)); dUf = zeros(size(params.Uf)); dbf = zeros(size(params.bf));
dWi = zeros(size(params.Wi)); dUi = zeros(size(params.Ui)); dbi = zeros(size(params.bi));
dWo = zeros(size(params.Wo)); dUo = zeros(size(params.Uo)); dbo = zeros(size(params.bo));
dWc = zeros(size(params.Wc)); dUc = zeros(size(params.Uc)); dbc = zeros(size(params.bc));
dWy = zeros(size(params.Wy)); dby = 0;

% ������ݶ�
h_final = cache.h{T+1};
dWy = dy * h_final';
dby = dy;
% ��ʼ�ݶȴ��ݵ����ʱ�䲽
dh_next = params.Wy' * dy;
dc_next = zeros(H,1);

% ���򴫲�����ÿ��ʱ�䲽 t = T...1
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
    
    % �� h_t �� c_t ���ݶ�
    dh = dh_next;  % �����ϲ������״̬�ݶ�
    do = dh .* tanh(c_t) .* o_t .* (1 - o_t);  % sigmoid �ݶ�
    % �ݶ�ͨ�� tanh(c_t)
    dtanh_c = (1 - tanh(c_t).^2);
    dc = dh .* o_t .* dtanh_c + dc_next;
    
    df = dc .* c_prev .* f_t .* (1 - f_t);
    di = dc .* g_t    .* i_t .* (1 - i_t);
    dg = dc .* i_t    .* (1 - g_t.^2);
    
    % �ۼƸ����ŵĲ����ݶ�
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
    
    % �����ݶȵ�ǰһ����״̬
    dh_prev = params.Uf' * df + params.Ui' * di + params.Uo' * do + params.Uc' * dg;
    dc_prev = dc .* f_t;
    
    dh_next = dh_prev;
    dc_next = dc_prev;
end

% �ռ��ݶ�
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

%% Sigmoid �������䵼��
function y = sigmoid(x)
    y = 1./(1+exp(-x));
end

function y = dsigmoid(x)
    s = sigmoid(x);
    y = s .* (1 - s);
end
