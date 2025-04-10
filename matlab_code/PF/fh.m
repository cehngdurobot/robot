 function result=fh(x,Data1)
% 
% y=norm(x);
% 
% end

% function y=f(x,KK,q,P0, L_ex11)
% sum=0;
% 
%     x1=KK+x;
%     L_norm1=my_forward(x1,q,P0);
%     sum = sum+(norm(L_ex11- L_norm1))^2;
% 
% 
% 
% y=sum;
P0=[ 0.24500 -0.45600 0.00665]';%标定点，测试效果
% a_d_alpha_theta0_norm=KK;
%a_d_alpha_theta0_ex=a_d_alpha_theta0_norm0;
% Data1 = xlsread('Data.xlsx',1,'D2:J111');
% Data2 = xlsread('Data.xlsx',2,'D2:J22');
U=Data1(:,1:6)*pi/180;
MM1=Data1(:,7)/1000;
% UU=Data2(:,1:6)*pi/180;
% VV1=Data2(:,7)/1000;

%------------------------------------------
%robot configurations for experiments
q1_batch = U(:,1);
q2_batch = U(:,2);
q3_batch = U(:,3);
q4_batch = U(:,4);
q5_batch = U(:,5);
q6_batch = U(:,6);

sum=0;
for j = 1:length(q1_batch)%each i represent one measurement
    q=[q1_batch(j);q2_batch(j);q3_batch(j);q4_batch(j);q5_batch(j);q6_batch(j)];
L_ex11=MM1(j);
 
% 
% for i=1:D
% 
%     sum=sum+x(i)^2;
% 
% end
x1=x;
%     x1=KK+x;
    L_norm1=my_forward(x1,q,P0);
    sum = sum+(norm(L_ex11- L_norm1))^2;

end

result1=sum/length(q1_batch);
result=(sqrt(result1))*1000;



