clear all;
close all;
clc;
zL=0;zU=1;
t0=0;tf=100;
t_sample=0.001;z_sample=0.05;
N_t=round((tf-t0)/t_sample)+1;
N_z=round((zU-zL)/z_sample)+1;

%setlmis([])
D=0.1;
Km=1;
eta=3;
cf=3;
for it=1:N_t
    ttt(it)=(it-1)*t_sample;   
    vv(it)=0;
    yo(it)=0;
    dd1(it)=0;
    y(it,1)=cf;
    ny(it)=0;
end

ydd=0.1; % 出口处期望浓度

for iz=1:N_z
    zzz(iz)=(iz-1)*z_sample;
    y(1,iz)=cf;
end
yo(1)=y(1,N_z);
        
% initial condition
KP=-0.001;
KI=-0.24;
KD=0.01;

for it=1:(N_t-1)
    for iz=2:N_z-1
        y(it+1,iz)=y(it,iz)+t_sample*(D*(y(it,iz+1)-2*y(it,iz)+y(it,iz-1))/z_sample/z_sample-vv(it)*(y(it,iz)-y(it,iz-1))/z_sample-eta*y(it,iz)/(Km+y(it,iz)));                
    end
    y(it+1,N_z)=y(it+1,N_z-1);
    yo(it+1)=y(it+1,N_z);
        
    for i=1:it+1
        dd1(it+1)=dd1(it+1)+t_sample*(yo(i)-ydd);
    end
    %模糊PID控制策略
    vv(it+1)=KP*(yo(it+1)-ydd)+KI*dd1(it+1)...
        +KD*(yo(it+1)-yo(it))/t_sample;
    
    %模糊积分控制策略
%     uu1(it+1)=uu1(it+1)+dd1(it+1);
end

for it=1:N_t
    for iz=1:N_z
        ny(it)=ny(it)+y(it,iz)*y(it,iz)*z_sample;
    end
end
% figure(1)
% plot(ttt,x)
figure(1)
set(1,'position',[449 240 364 269])
mesh(zzz,ttt,y)
view(-37.5+90,30);
xlabel({'$x$'},'Interpreter','latex')
ylabel({'$t$'},'Interpreter','latex')
zlabel({'$C$'},'Interpreter','latex')

figure(2)
set(2,'position',[131 400 344 179])
plot(ttt,sqrt(ny),'k-')
xlabel({'$t$'},'Interpreter','latex')
ylabel({'$\|C(\cdot,t)\|_2$'},'Interpreter','latex')

figure(5)
set(5,'position',[131 400 344 229])
h=plot(ttt,yo,'b--')
%legend('手动控制','积分控制','模糊PID控制','Location','SouthEast')
set(h,'LineWidth',2);
xlabel({'$t$'},'Interpreter','latex')
ylabel({'$y_{out}(t)$'},'Interpreter','latex')
% 
figure(3)
set(3,'position',[131 400 344 179])
plot(ttt,vv,'k-')
xlabel({'$t$'},'Interpreter','latex')
ylabel({'$v(t)$'},'Interpreter','latex')

figure(4)
set(4,'position',[449 240 364 269])
plot(ttt,sqrt(ny),'k-')
%legend({'$\|C(\cdot,t)\|_2$','$\|\hat{C}(\cdot,t)\|_2$'},'Interpreter','latex')
xlabel({'$t$'},'Interpreter','latex')
ylabel({'$\|C(\cdot,t)\|_2$, $\|\hat{C}(\cdot,t)\|_2$'},'Interpreter','latex')

% figure(7)
% set(5,'position',[131 400 344 179])
% plot(ttt,dnorm1,'k-',ttt,dnorm2,'r:')
% legend({'$\|y_1(\cdot,t)\|_2$','$\|y_2(\cdot,t)\|_2$'},'Interpreter','latex')
% xlabel({'$t$'},'Interpreter','latex')
% ylabel({'$\|y_1(\cdot,t)\|_2$, $\|y_2(\cdot,t)\|_2$'},'Interpreter','latex')
% 
% 
% figure(6)
% set(6,'position',[449 240 364 269*1.2])
% plot(y2(:,5),y1(:,5))