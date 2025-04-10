%revision history:
%v1: Feb.5th, 2020,by Shuai Li
%v2: Feb. 11th,2020 by Shuai Li
clear all
close all
tic
%------------------------------------------
%�궨���֣��������ݼ����б궨
%�ⲿ���ǻ�������������pre_processing.m(����matlab���������ȡ���궨��е�۵�foward
%kinematics����Բ����㶯��Jacobian���󣩣�my_forward.m(��pre_processing.m��õ�forward
%kinematics����my_Jacobian.m����pre_processing.m��õ�Jacobian)��
%------------------------------------------
%ϵͳ������������е�۵���ʵ����ֵ���������ֵ��
%����˳��a1,a2,a3,a4,a5,a6,d1,d2,d3,d4,d5,d6,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,...
%theta0_1,theta0_2,theta0_3,theta0_4,theta0_5,theta0_6;
%a_d_alpha_theta0_ex=rand(24,1);%[];%�����˵���ʵ����a d alpha theta0
    a_d_alpha_theta0_norm0=...
    [0 0.27 0.07 0 0 0 ...
    0.29 0 0 0.302 0 0.072 ...
    -1.571 0 -1.571 1.571 -1.571 0 ...
    0 -1.57 0 0 0 0]';
%P0=[0.5,-1,1-0.0627]';%�궨��
 %P0=[0.242654051842270 -0.453838756463155 0.007685659149684]';
% P0=[ 0.242 -0.453 0.006]';%Ч����
 P0=[ 0.24500 -0.45600 0.00665]';%�궨�㣬����Ч��
%a_d_alpha_theta0_norm=rand(24,1);%[];%�����˵��������a d alpha theta0
a_d_alpha_theta0_norm=a_d_alpha_theta0_norm0;
%a_d_alpha_theta0_ex=a_d_alpha_theta0_norm.*(1+0.05*rands(24,1))+0.05*rands(24,1)+0.1*randn(24,1);
% a_d_alpha_theta0_ex=a_d_alpha_theta0_norm.*(1+0.05*rands(27,1))+0.05*rands(27,1);
%  a_d_alpha_theta0_ex=a_d_alpha_theta0_norm.*(1+0.05*rands(24,1))+0.05*rands(24,1);
%a_d_alpha_theta0_ex=a_d_alpha_theta0_norm0;
Data1 = xlsread('Data.xlsx',1,'D2:J112');
Data2 = xlsread('Data.xlsx',2,'D2:J22');
U=Data1(:,1:6)*pi/180;
MM1=Data1(:,7)/1000;
UU=Data2(:,1:6)*pi/180;
VV1=Data2(:,7)/1000;

%------------------------------------------
%robot configurations for experiments
% q1_batch = pi/2*rands(30,1);
% q2_batch = pi/2*rands(30,1);
% q3_batch = pi/2*rands(30,1);
% q4_batch = pi/2*rands(30,1);
% q5_batch = pi/2*rands(30,1);
% q6_batch = pi/2*rands(30,1);
q1_batch = U(:,1);
q2_batch = U(:,2);
q3_batch = U(:,3);
q4_batch = U(:,4);
q5_batch = U(:,5);
q6_batch = U(:,6);
% % P0=[1,-1,0.16-0.0627]';%�궨��
KH1=0.0004;
% 
% %------------------------------------------%------------------------------------------
% %------------------------------------------%------------------------------------------
%Iter=8;
% Iter1=5;
MSE_store=zeros(0,1);
%L_norm1_store=zeros(0,1);
  %L_ex11_store=zeros(0,1);
  %Iter1=10;
  
% %PF11�㷨
 x1 = a_d_alpha_theta0_norm; %��ʼֵ

xbest=x1;

    
  fbest=fh(xbest,Data1);   
 
 


Iter1=50;
  for iteration=1:Iter1
XX=zeros(24,1);

 x = a_d_alpha_theta0_norm; %��ʼֵ



R = 2;
%Q = 0.0001;
%tf = 100; %����ʱ��


N = 20;  %���Ӹ���
 P = 0.3;
xhatPart = x;
xpartminus=zeros(24,N);
xhatPartArr=zeros(24,1);
MSE0=0;
MSE=0;
% for i = 1 : N    
% %     xpart(:,i) = x + 0.001*sqrt(P) * randn(24,1);%��ʼ״̬����0��ֵ������Ϊsqrt(P)�ĸ�˹�ֲ�
%  xpart(:,i) = x+0.001*sqrt(P) * randn(24,1);
%  %xpart(:,i) = x;
% end


for i = 1 : N    
    xpart(:,i) = x + 0.01*sqrt(P) * randn(24,1);%��ʼ״̬����0��ֵ������Ϊsqrt(P)�ĸ�˹�ֲ�
end


% xArr = [x];
% %yArr = [x^2 / 20 + sqrt(R) * randn];
% xhatArr = [x];
% PArr = [P];
% xhatPartArr = [xhatPart];

    
for i = 1 : N 
    
for j = 1:length(q1_batch)%each i represent one measurement
    q=[q1_batch(j);q2_batch(j);q3_batch(j);q4_batch(j);q5_batch(j);q6_batch(j)];
    L_ex11=MM1(j);
%     L_ex12=my_forward(a_d_alpha_theta0_ex,q,P0);  
xx=xpart(:,i);
xpartminus(:,i) = xpart(:,i);

% x = x+ 0.01*sqrt(Q) * randn(24,1);
%0.5 * x + 25 * x / (1 + x^2) + 8 * cos(1.2*(k-1)) + sqrt(Q) * randn;
    %kʱ����ʵֵ
%     y = x^2 / 20 + sqrt(R) * randn;  %kʱ�̹۲�ֵ
y=my_forward(xx,q,P0);  
 %for k = 1 : N
     %xpartminus(:,i) = xpart(:,i)+ 0.01*sqrt(Q) * randn(24,1);
%      xpartminus(:,i) = xpart(:,i);
%      
%      0.5 * xpart(i) + 25 * xpart(i) / (1 + xpart(i)^2) ...
%          + 8 * cos(1.2*(k-1)) + sqrt(Q) * randn;%�������N������
%      ypart = xpartminus(i)^2 / 20;%ÿ�����Ӷ�Ӧ�۲�ֵ
ypart =L_ex11;
     vhat1 = y - ypart;%����ʵ�۲�֮�����Ȼ
     
     MSE0=MSE0+(norm(vhat1))^2;
end
vhat=sqrt(MSE0/length(q1_batch));  
     
     
    
     qq(i) = (1 / sqrt(R) / sqrt(2*pi)) * exp(-vhat^2 / 2 / R); 
     %ÿ�����ӵ���Ȼ�����ƶ�
% end
end
 qsum = sum(qq);

for i1 = 1 : N
    qq(i1) = qq(i1) / qsum;%Ȩֵ��һ��
end  

  for ii = 1 : N %����Ȩֵ���²���
      u = 0.5*rand;
      qtempsum = 0;
      for j1 = 1 : N

 qtempsum = qtempsum + qq(j1);
          if qtempsum >= u
              xpart(:,ii) = xpartminus(:,j1);
              break;
          end
      end
  end
  GetB= xpart(:,any(xpart));
   xpart=GetB;
  
  NN=size(xpart,2);
  
  
  for i2=1:NN
%     kk=xpart(i,:);
   XX=XX+ xpart(:,i2);
    
    
  end
%XX=XX;
 XX=(XX/NN);
x=XX;
% XX=(XX/NN);
% x=XX;
%xhatPart=XX;

%�����
f1=fh(x,Data1);

%%%%%%%%%%%

if f1<fbest

xbest=x;

fbest=f1;

end

% x=xbest;


for i2 = 1:length(q1_batch)%each i represent one measurement
    q=[q1_batch(i2);q2_batch(i2);q3_batch(i2);q4_batch(i2);q5_batch(i2);q6_batch(i2)];
%     L_ex11=my_forward(a_d_alpha_theta0_xyz_ex,q);
L_ex11=MM1(i2);
    L_norm1=my_forward(xbest,q,P0);
    
    Error=L_norm1-L_ex11;
    MSE=MSE+(norm(Error))^2;
  
  
end





MSE=sqrt(MSE/length(q1_batch));
MSE=MSE*1000;
MSE_store=[MSE_store;MSE];



%  MSE=vhat*1000;
% MSE_store=[MSE_store;MSE];
  
%xhatPartArr=xhatPartArr+xhatPart;
% a_d_alpha_theta0_norm=x;  
  
% xhatPart = mean(xpart);
% %����״̬����ֵ��ΪN�����ӵ�ƽ��ֵ�����ﾭ�����²�����������ӵ�Ȩֵ��ͬ
% xArr = [xArr x];   
% yArr = [yArr y];  
% % xhatArr = [xhatArr xhat]; 
% PArr = [PArr P]; 
% xhatPartArr = [xhatPartArr xhatPart];
 end



%xhatPartArr=xhatPartArr/length(q1_batch);

a_d_alpha_theta0_norm=xbest;
% end
toc
a_d_alpha_theta0_norm

figure(1),clf(1),
semilogy(MSE_store,'rx-')
%xlabel('Iteration');
xlabel('Iteration number');
%ylabel('MSE');
ylabel('Error/mm');

% toc
% %------------------------------------------
% %���ݿ��ӻ�data visualization
% %------------------------------------------
%  format long
% figure(1),clf(1),
% semilogy(MSE_store,'rx-')
% %xlabel('Iteration');
% xlabel('Iteration number');
% %ylabel('MSE');
% ylabel('Error/mm');
% % display('Calibration Error:');
% % %a_d_alpha_theta0_norm-a_d_alpha_theta0_ex
% display('Calibrated Parameters:');
% a_d_alpha_theta0_norm
% P0
%------------------------------------------
%���Բ��֣������µ����ݼ����Ա궨Ч��
%------------------------------------------
%the above can be viewed as training from machine learning perspective
%the following is testing
% q=rands(6,1);
q11=[-67.7	24.7	-14.5	-14.9	75.1	-54.4];
q=q11*pi/180;
%L_ex11=my_forward(a_d_alpha_theta0_ex,q);
L_ex11=0.4845;
L_norm1=my_forward(a_d_alpha_theta0_norm,q,P0);
J_norm1=my_Jacobian(a_d_alpha_theta0_norm,q,P0);
Error=L_norm1-L_ex11;
display(['Position error:',num2str(Error')]);
%testing data batch
% q1_batch = pi*rands(50,1);
% q2_batch = pi*rands(50,1);
% q3_batch = pi*rands(50,1);
% q4_batch = pi*rands(50,1);
% q5_batch = pi*rands(50,1);
% q6_batch = pi*rands(50,1);
q1_batch = U(:,1);
q2_batch = U(:,2);
q3_batch = U(:,3);
q4_batch = U(:,4);
q5_batch = U(:,5);
q6_batch = U(:,6);

MSE=0;
E_max2=0;
%--------
%a_d_alpha_theta0_ex=a_d_alpha_theta0_ex+0.4*rands(24,1);
for i = 1:length(q1_batch)%each i represent one measurement
    q=[q1_batch(i);q2_batch(i);q3_batch(i);q4_batch(i);q5_batch(i);q6_batch(i)];
%     L_ex11=my_forward(a_d_alpha_theta0_ex,q);
    L_ex11=MM1(i);
    L_norm1=my_forward(a_d_alpha_theta0_norm,q,P0);
    Error=L_norm1-L_ex11;
    MSE=MSE+(norm(Error))^2;
    E_max2=max((norm(Error))^2,E_max2);
end
%MSE=MSE/length(q1_batch);
%display(['Testing MSE:',num2str(MSE)]);
%display(['Max Error square:',num2str(E_max2)]);
MSE=MSE/length(q1_batch);
RMSE=sqrt(MSE);
E_max2=sqrt(E_max2);
display(['Testing RMSE:',num2str(RMSE)]);
display(['Max Error:',num2str(E_max2)]);
