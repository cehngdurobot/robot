function L = DH(a_d_alpha_theta0_xyz,q, P0)
% dh_kinematics_single 
%
% 将下列功能合并在同一个函数中：
%   1. 解析 DH 参数 (parse_dh_params)
%   2. 计算正向运动学 (forward_kinematics_6dof)
%   3. 计算末端与目标点 P0 的距离 (dh_forward_kinematics_distance)
%
% 输入:
%   q                     : 1×6 的关节角数组 [若为弧度制，则 (alpha, theta0) 也应为弧度]
%   a_d_alpha_theta0_xyz  : 1×24 的 DH 参数 (a1..a6, d1..d6, alpha1..alpha6, theta0_1..theta0_6)
%   P0                    : 1×3 的数组，表示实际末端坐标 (x0, y0, z0)
%
% 输出:
%   L  : 末端计算坐标与 P0 的欧氏距离 (标量)

    %---------------------------%
    % 1. 解析 24 个 DH 参数    %
    %---------------------------%
    params = a_d_alpha_theta0_xyz(:);  % 转成列向量


    a1 = params(1);  a2 = params(2);  a3 = params(3);
    a4 = params(4);  a5 = params(5);  a6 = params(6);

    d1 = params(7);  d2 = params(8);  d3 = params(9);
    d4 = params(10); d5 = params(11); d6 = params(12);

    alpha1 = params(13); alpha2 = params(14); alpha3 = params(15);
    alpha4 = params(16); alpha5 = params(17); alpha6 = params(18);

    theta0_1 = params(19); theta0_2 = params(20); theta0_3 = params(21);
    theta0_4 = params(22); theta0_5 = params(23); theta0_6 = params(24);

    %---------------------------%
    % 2. 计算正向运动学        %
    %---------------------------%
    % 将关节的 DH 参数组合为 6×4 的矩阵, 每行 (a_i, alpha_i, d_i, theta0_i)
    dh_list = [ ...
        a1, alpha1, d1, theta0_1;
        a2, alpha2, d2, theta0_2;
        a3, alpha3, d3, theta0_3;
        a4, alpha4, d4, theta0_4;
        a5, alpha5, d5, theta0_5;
        a6, alpha6, d6, theta0_6 ];

    % 累乘得到末端变换矩阵
    T = eye(4);
    for i = 1:6
        a_i     = dh_list(i,1);
        alpha_i = dh_list(i,2);
        d_i     = dh_list(i,3);
        theta0_i= dh_list(i,4);

        % 真实关节角 = q(i) + theta0_i
        theta_i = q(i) + theta0_i;

        % 求第 i 轴的 DH 齐次变换矩阵
        T_i = local_dh_transform(a_i, alpha_i, d_i, theta_i);

        % 累乘
        T = T * T_i;
    end

    % 提取末端坐标
    P_end = T(1:3, 4);

    %-------------------------------------%
    % 3. 与 P0 的距离（欧氏距离）         %
    %-------------------------------------%
    diff_ = P_end - P0(:);
    L = norm(diff_);

end


%%======================================================================
function T = local_dh_transform(a, alpha, d, theta)
% local_dh_transform
%
% 给定单个关节的 DH 参数 (a, alpha, d, theta)，返回其 4×4 齐次变换矩阵。
%
% 注意：若 alpha、theta 为角度(°)，请在此处使用 deg2rad() 转为弧度；
%       若已是弧度制，则可直接使用。

    ca = cos(alpha);
    sa = sin(alpha);
    ct = cos(theta);
    st = sin(theta);

    T = [ ...
        ct,        -st*ca,   st*sa,    a*ct;
        st,         ct*ca,  -ct*sa,    a*st;
        0,          sa,      ca,       d;
        0,          0,       0,        1 ];
end
