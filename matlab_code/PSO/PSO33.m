tic   % 计时开始

%% ------初始格式化--------------------------------------------------
rng(20, 'twister');      % 固定随机数种子
clear all;               % 清空工作区
clc;                     % 清空命令行窗口
% format long;          % 可选：以长格式显示数值
% 基础 DH 参数（原始标定参数）

a_d_alpha_theta0_norm0 = [...
    0,    0.27, 0.07, 0,     0,     0,  ...
    0.29, 0,    0,    0.302, 0,     0.072, ...
    -1.571,0,   -1.571,1.571,-1.571,0,   ...
    0,   -1.57, 0,    0,     0,     0]';
P0 = [0.24500; -0.45600; 0.00665];   % 标定点（用于正向运动学）

a_d_alpha_theta0_norm = a_d_alpha_theta0_norm0;

% 读取 Excel 数据
Data1 = xlsread('Data.xlsx', 1, 'D2:J112');  % 第一工作表
Data2 = xlsread('Data.xlsx', 2, 'D2:J22');     % 第二工作表

% 数据预处理
U   = Data1(:,1:6) * pi/180;  % 角度转换为弧度
MM1 = Data1(:,7)   / 1000;     % 距离单位（例如 mm 转 m）
UU  = Data2(:,1:6) * pi/180;
VV1 = Data2(:,7)   / 1000;

% 提取实验数据中的各个关节角
q1_batch = U(:,1);
q2_batch = U(:,2);
q3_batch = U(:,3);
q4_batch = U(:,4);
q5_batch = U(:,5);
q6_batch = U(:,6);
q = [q1_batch, q2_batch, q3_batch, q4_batch, q5_batch, q6_batch];
L_ex11 = MM1;   % 真实量测值

%% ------给定 PSO 参数与初始条件--------------------------------------
% PSO 参数（用于误差补偿优化）
c1 = 1.4962;      % 学习因子 1
c2 = 1.4962;      % 学习因子 2
w  = 0.2298;      % 惯性权重

MaxDT = 50;       % 最大迭代次数
D     = 6;        % 设计变量维数（直接优化误差补偿 K22，共6维）
N     = 300;       % PSO 粒子数（种群规模）
% ------初始化种群粒子（仅对 K22 进行搜索）--------------------

for i = 1:N
    for j = 1:D
        x(i,j) = 0.001 * randn;  % 位置初值：服从 N(0, 0.001^2)
        v(i,j) = 0.001 * randn;  % 速度初值：服从 N(0, 0.001^2)
    end
end

% KK 保存基础 DH 参数（不变）
KK = a_d_alpha_theta0_norm;

% ------计算各个粒子的适应度，并初始化个体和全局最优------------
for i = 1:N
    % 适应度函数 fh 已经修改为：利用候选误差补偿直接构造 R12 =
    % [zeros(18,1); K22]，代入 DH 计算，并基于 Data1 计算均方误差
    p(i)   = fh(x(i,:), KK, Data1);  
    y(i,:) = x(i,:);
end

Pg = x(1,:);  % 初始化全局最优为第一个粒子
for i = 2:N
    if fh(x(i,:), KK, Data1) < fh(Pg, KK, Data1)
        Pg = x(i,:);
    end
end

%% ------PSO 主循环----------------------------------------------------
for t = 1:MaxDT
    KK = a_d_alpha_theta0_norm;  % 这里 KK 没有变化，但写上以便与 fh 保持一致
    for i = 1:N
        % 粒子速度更新
        v(i,:) = w * v(i,:) + c1 * rand * (y(i,:) - x(i,:)) + c2 * rand * (Pg - x(i,:));
        % 更新位置
        x(i,:) = x(i,:) + v(i,:);
        % 更新个体最优
        if fh(x(i,:), KK, Data1) < p(i)
            p(i)   = fh(x(i,:), KK, Data1);
            y(i,:) = x(i,:);
        end
        % 更新全局最优
        if p(i) < fh(Pg, KK, Data1)
            Pg = y(i,:);
        end
    end
    % 保存每一代的最优适应度值（便于后续收敛曲线绘制）
    Pbest(t) = fh(Pg, KK, Data1);
end

%% ------获取最终优化结果并构造误差补偿向量-------------------------
% 直接将 PSO 搜索到的全局最优解作为误差补偿 K22
K22 = Pg(:);  % 转换为列向量

% 构造完整的补偿向量 R12 (补偿只作用于后 6 项，其余前 18 项为 0)
R12 = [zeros(18,1); K22];

toc   % 计时结束

% ------显示优化结果----------------------------------------------
disp('*************************************************************')    
disp('函数的全局最优误差补偿 K22 为：')
disp(K22)
fprintf('最后得到的优化极值为：%s\n', num2str(fh(Pg, KK, Data1)));
disp('*************************************************************')

%% ------绘制迭代过程中适应度的收敛曲线------------------------
figure(1), clf,
semilogy(Pbest, 'rx-')
xlabel('Iteration');
ylabel('Fitness Value');
title('PSO 迭代过程中适应度的收敛曲线');

%% ------测试部分：计算补偿后的正向运动学误差----------------------
MSE = 0; 
E_max2 = 0;
for i = 1:length(q1_batch)
    % 取当前测量点的关节角（列向量）
    q_test = [q1_batch(i); q2_batch(i); q3_batch(i); q4_batch(i); q5_batch(i); q6_batch(i)];
    L_ex_val = MM1(i);   % 真实测量值
    % 计算补偿后的正向运动学：参数为基础参数加上误差补偿 R12
    L_norm1 = DH(a_d_alpha_theta0_norm + R12, q_test, P0);  
    Error   = L_norm1 - L_ex_val;  % 计算误差
    MSE     = MSE + (norm(Error))^2;
    E_max2  = max((norm(Error))^2, E_max2);
end

MSE   = MSE / length(q1_batch);
RMSE  = sqrt(MSE);
E_max2 = sqrt(E_max2);

disp(['Testing RMSE: ', num2str(RMSE)]);
disp(['Max Error: ', num2str(E_max2)]);

% tic   % 计时开始

% %% ------初始格式化--------------------------------------------------
% rng(20, 'twister');      % 固定随机数种子
% clear all;               % 清空工作区
% clc;                     % 清空命令行窗口
% % format long;          % 可选：以长格式显示数值
% 
% % 基础 DH 参数（原始标定参数）
% a_d_alpha_theta0_norm0 = [...
%     0,    0.27, 0.07, 0,     0,     0,  ...
%     0.29, 0,    0,    0.302, 0,     0.072, ...
%     -1.571,0,   -1.571,1.571,-1.571,0,   ...
%     0,   -1.57, 0,    0,     0,     0]';
% P0 = [0.24500; -0.45600; 0.00665];   % 标定点（用于正向运动学）
% 
% a_d_alpha_theta0_norm = a_d_alpha_theta0_norm0;
% 
% % 读取 Excel 数据
% Data1 = xlsread('Data.xlsx', 1, 'D2:J112');  % 第一工作表
% Data2 = xlsread('Data.xlsx', 2, 'D2:J22');     % 第二工作表
% 
% % 数据预处理
% U   = Data1(:,1:6) * pi/180;  % 角度转换为弧度
% MM1 = Data1(:,7)   / 1000;     % 距离单位转换为m（例如 mm 转 m）
% UU  = Data2(:,1:6) * pi/180;
% VV1 = Data2(:,7)   / 1000;
% 
% % 提取实验数据中的各个关节角
% q1_batch = U(:,1);
% q2_batch = U(:,2);
% q3_batch = U(:,3);
% q4_batch = U(:,4);
% q5_batch = U(:,5);
% q6_batch = U(:,6);
% q = [q1_batch, q2_batch, q3_batch, q4_batch, q5_batch, q6_batch];
% L_ex11 = MM1;   % 真实量测值
% 
% %% ------给定 PSO 参数与初始条件--------------------------------------
% % PSO 参数（用于误差补偿优化）
% c1 = 1.4962;      % 学习因子 1
% c2 = 1.4962;      % 学习因子 2
% w  = 0.2298;      % 惯性权重
% 
% MaxDT = 50;       % 最大迭代次数
% D     = 24;       % 设计变量维数（直接优化24个DH参数的误差补偿）
% N     = 300;      % PSO 粒子数（种群规模）
% 
% % ------初始化种群粒子（对24个误差补偿参数进行搜索）--------------------
% for i = 1:N
%     for j = 1:D
%         x(i,j) = 0.001 * randn;  % 位置初值：服从 N(0, 0.001^2)
%         v(i,j) = 0.001 * randn;  % 速度初值：服从 N(0, 0.001^2)
%     end
% end
% 
% % KK 保存基础 DH 参数（不变）
% KK = a_d_alpha_theta0_norm;
% 
% % ------计算各个粒子的适应度，并初始化个体和全局最优------------
% for i = 1:N
%     % 适应度函数 fh 已修改为：利用候选误差补偿直接构造修正的 DH 参数，
%     % 代入 DH 计算，通过 Data1 测量值计算均方误差。
%     p(i)   = fh(x(i,:), KK, Data1);  
%     y(i,:) = x(i,:);
% end
% 
% Pg = x(1,:);  % 初始化全局最优为第一个粒子
% for i = 2:N
%     if fh(x(i,:), KK, Data1) < fh(Pg, KK, Data1)
%         Pg = x(i,:);
%     end
% end
% 
% %% ------PSO 主循环----------------------------------------------------
% for t = 1:MaxDT
%     KK = a_d_alpha_theta0_norm;  % 固定基础 DH 参数
%     for i = 1:N
%         % 粒子速度更新
%         v(i,:) = w * v(i,:) + c1 * rand * (y(i,:) - x(i,:)) + c2 * rand * (Pg - x(i,:));
%         % 更新位置
%         x(i,:) = x(i,:) + v(i,:);
%         % 更新个体最优
%         if fh(x(i,:), KK, Data1) < p(i)
%             p(i)   = fh(x(i,:), KK, Data1);
%             y(i,:) = x(i,:);
%         end
%         % 更新全局最优
%         if p(i) < fh(Pg, KK, Data1)
%             Pg = y(i,:);
%         end
%     end
%     % 保存每一代的最优适应度值（便于后续收敛曲线绘制）
%     Pbest(t) = fh(Pg, KK, Data1);
% end
% 
% %% ------获取最终优化结果并构造误差补偿向量-------------------------
% % 将 PSO 搜索到的全局最优解直接作为误差补偿向量 R12（24×1向量）
% R12 = Pg(:);
% 
% toc   % 计时结束
% 
% % ------显示优化结果----------------------------------------------
% disp('*************************************************************')    
% disp('函数的全局最优误差补偿 R12 为：')
% disp(R12)
% fprintf('最后得到的优化极值为：%s\n', num2str(fh(Pg, KK, Data1)));
% disp('*************************************************************')
% 
% %% ------绘制迭代过程中适应度的收敛曲线------------------------
% figure(1), clf,
% semilogy(Pbest, 'rx-')
% xlabel('Iteration');
% ylabel('Fitness Value');
% title('PSO 迭代过程中适应度的收敛曲线');
% 
% %% ------测试部分：计算补偿后的正向运动学误差----------------------
% MSE = 0; 
% E_max2 = 0;
% for i = 1:length(q1_batch)
%     % 取当前测量点的关节角（列向量）
%     q_test = [q1_batch(i); q2_batch(i); q3_batch(i); q4_batch(i); q5_batch(i); q6_batch(i)];
%     L_ex_val = MM1(i);   % 真实测量值
%     % 计算补偿后的正向运动学：参数为基础 DH 参数加上补偿 R12（24维全补偿）
%     L_norm1 = DH(a_d_alpha_theta0_norm + R12, q_test, P0);  
%     Error   = L_norm1 - L_ex_val;  % 计算误差
%     MSE     = MSE + (norm(Error))^2;
%     E_max2  = max((norm(Error))^2, E_max2);
% end
% 
% MSE   = MSE / length(q1_batch);
% RMSE  = sqrt(MSE);
% E_max2 = sqrt(E_max2);
% 
% disp(['Testing RMSE: ', num2str(RMSE)]);
% disp(['Max Error: ', num2str(E_max2)]);
% 
% function result = fh(K22, baseDH, Data1)
% % fh - 适应度函数：利用24维误差补偿向量K22修正DH参数后计算误差
% %
% %   result = fh(K22, baseDH, Data1)
% %
% % 输入参数：
% %   K22     : 24维误差补偿量（待优化变量），用于对所有DH参数进行补偿。
% %   baseDH  : 原始DH参数（24×1列向量）。
% %   Data1   : 实验数据，前6列为关节角（单位：度），第7列为测量值（例如距离，单位：mm）。
% %
% % 说明：
% %   误差补偿向量K22直接作用于全部DH参数，从而补偿后的DH参数为：
% %
% %         compDH = baseDH + K22.
% %
% %   使用补偿后的DH参数，通过正向运动学函数DH计算末端输出，
% %   并对数据集中每组测量计算预测输出与实际测量值之间的误差，
% %   此处目标函数采用所有测量数据中最大的平方误差作为适应度输出；
% %   如果需要，也可以取消注释采用均方误差。
% %
% % 输出：
% %   result  : 目标函数值。
% 
% % 标定点（用于正向运动学计算）
% P0 = [0.24500; -0.45600; 0.00665];
% 
% % 数据预处理：将角度转换为弧度，将测量值由mm转换为m
% U   = Data1(:, 1:6) * pi/180;
% MM1 = Data1(:, 7)   / 1000;
% 
% % 误差补偿向量：K22直接为24×1向量（注意K22应为行向量时转换为列向量）
% R12 = K22(:);
% 
% % 计算补偿后的DH参数
% compDH = baseDH + R12;
% 
% % 初始化累积误差与最大误差
% sum_err = 0;
% E_max2  = 0;
% 
% % 遍历所有测量数据
% for k = 1:size(U,1)
%     % 当前测量点的关节角（列向量）
%     q = U(k, :)';
%     % 实测末端输出（例如距离）
%     L_ex_val = MM1(k);
% 
%     % 利用修正后的DH参数计算正向运动学输出
%     L_norm = DH(compDH, q, P0);
% 
%     % 计算当前测量的平方误差
%     err = (norm(L_ex_val - L_norm))^2;
% 
%     % 累加求和（可用于均方误差）
%     sum_err = sum_err + err;
% 
%     % 记录最大平方误差
%     if err > E_max2
%         E_max2 = err;
%     end
% end
% 
% % 这里目标函数可以设计为：
% % ① 最大平方误差：
% result = E_max2;
% % ② 均方误差（则取消下面注释即可）：
% % result = sum_err / size(U,1);
% 
% end