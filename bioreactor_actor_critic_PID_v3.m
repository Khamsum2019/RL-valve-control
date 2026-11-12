%% Actor-Critic RBF Incremental PID - Final Clean Version
clear all; close all; clc;

%% PART 1: System Parameters
zL=0; zU=1;
t0=0; tf=100;
t_sample = 0.001;
z_sample = 0.05;
N_t = round((tf-t0)/t_sample)+1;
N_z = round((zU-zL)/z_sample)+1;

D=0.1; Km=1; eta=3; cf=3; ydd=0.1;

ttt = (0:N_t-1)*t_sample;
zzz = (0:N_z-1)*z_sample;

fprintf('Actor-Critic RBF Incremental PID - Final Version\n');
fprintf('N_t=%d, N_z=%d\n\n', N_t, N_z);

%% PART 2: RBF Parameters
n_rbf   = 3;
n_input = 3;

alpha_w  = 0.001;
alpha_v  = 0.005;
alpha_mu = 0.001;
alpha_sg = 0.001;
gamma    = 0.98;

%% Control update interval
%% This prevents chattering in the incremental PID
ctrl_interval = 100;

%% PART 3: Training Setup
num_episodes   = 50;
episode_reward = zeros(1, num_episodes);
episode_ISE    = zeros(1, num_episodes);
best_ISE = inf;
best_yo  = zeros(1, N_t);
best_vv  = zeros(1, N_t);
best_y   = zeros(N_t, N_z);
best_KP  = zeros(1, N_t);
best_KI  = zeros(1, N_t);
best_KD  = zeros(1, N_t);

%% Network persists across episodes
w_actor  = 0.01 * randn(n_rbf, 3);
w_critic = 0.01 * randn(n_rbf, 1);
mu       = 0.1  * randn(n_input, n_rbf);
sg       = ones(1, n_rbf);

fprintf('%-8s %-14s %-12s %-36s\n','Episode','Reward','ISE','KP  KI  KD');
fprintf('%s\n', repmat('-',1,75));

for episode = 1:num_episodes

    y       = zeros(N_t, N_z);
    vv      = zeros(1,   N_t);
    yo      = zeros(1,   N_t);
    KP_hist = zeros(1,   N_t);
    KI_hist = zeros(1,   N_t);
    KD_hist = zeros(1,   N_t);

    for it = 1:N_t; y(it,1) = cf; end
    for iz = 1:N_z; y(1,iz) = cf; end
    yo(1) = y(1, N_z);

    KP      = 0;
    KI      = 0;
    KD      = 0;
    e_prev2 = 0;
    e_prev1 = yo(1) - ydd;
    V_prev  = 0;
    total_reward = 0;

    %% v starts at 0, updated every ctrl_interval steps
    v_current = 0.01;

    for it = 1:(N_t-1)

        %% PDE
        for iz = 2:N_z-1
            y(it+1,iz) = y(it,iz) + t_sample*( ...
                D*(y(it,iz+1)-2*y(it,iz)+y(it,iz-1))/z_sample^2 ...
                - vv(it)*(y(it,iz)-y(it,iz-1))/z_sample ...
                - eta*y(it,iz)/(Km+y(it,iz)));
            if isnan(y(it+1,iz)) || y(it+1,iz)<0
                y(it+1,iz) = y(it,iz);
            end
        end
        y(it+1,N_z) = y(it+1,N_z-1);
        yo(it+1)    = max(y(it+1,N_z), 0);

        %% Error signals
        e_curr = yo(it+1) - ydd;
        De     = e_curr - e_prev1;
        D2e    = e_curr - 2*e_prev1 + e_prev2;

        %% Clamp inputs
        e_curr = max(min(e_curr, 5), -5);
        De     = max(min(De,     5), -5);
        D2e    = max(min(D2e,    5), -5);

        theta = [e_curr; -De; -D2e];

        %% RBF
        phi = zeros(n_rbf, 1);
        for j = 1:n_rbf
            d      = theta - mu(:,j);
            phi(j) = exp(-sum(d.^2) / (2*sg(j)^2 + 1e-8));
        end

        %% Actor gains
        K  = w_actor' * phi;
        KP = max(min(K(1),  0.1), -0.1);
        KI = max(min(K(2),  0.5), -0.5);
        KD = max(min(K(3), 0.05), -0.05);

        %% Update v(t) only every ctrl_interval steps
        if mod(it, ctrl_interval) == 0 || it == 1
            delta_u   = KP*De + KI*e_curr + KD*D2e;
            v_current = v_current + delta_u;
            v_current = max(min(v_current, 0.5), 0);
        end
        vv(it+1) = v_current;

        KP_hist(it+1) = KP;
        KI_hist(it+1) = KI;
        KD_hist(it+1) = KD;

        %% Reward and TD
        R_t      = 0.5 * e_curr^2;
        V_curr   = w_critic' * phi;
        delta_TD = max(min(R_t + gamma*V_curr - V_prev, 10), -10);
        dy_du    = -1;

        %% Update actor
        w_actor(:,1) = w_actor(:,1) - alpha_w * delta_TD * e_curr * De  * phi * dy_du;
        w_actor(:,2) = w_actor(:,2) - alpha_w * (-delta_TD * e_curr^2   * phi * dy_du);
        w_actor(:,3) = w_actor(:,3) - alpha_w * delta_TD * e_curr * D2e * phi * dy_du;

        %% Update critic
        w_critic = w_critic + alpha_v * delta_TD * phi;

        %% Update centers and widths
        for j = 1:n_rbf
            mu(:,j) = mu(:,j) + alpha_mu * delta_TD * w_critic(j) * phi(j) ...
                      * (theta - mu(:,j)) / (sg(j)^2 + 1e-8);
            sg(j)   = max(sg(j) + alpha_sg * delta_TD * w_critic(j) * phi(j) ...
                      * sum((theta-mu(:,j)).^2) / (sg(j)^3+1e-8), 0.01);
        end

        e_prev2      = e_prev1;
        e_prev1      = e_curr;
        V_prev       = V_curr;
        total_reward = total_reward - R_t;

    end

    ISE = sum((yo - ydd).^2) * t_sample;
    episode_reward(episode) = total_reward;
    episode_ISE(episode)    = ISE;

    if ISE < best_ISE
        best_ISE = ISE; best_yo = yo; best_vv = vv;
        best_y = y; best_KP = KP_hist;
        best_KI = KI_hist; best_KD = KD_hist;
        fprintf('  -> Best ISE=%.4f at episode %d\n', best_ISE, episode);
    end

    if mod(episode,5)==0 || episode==1
        fprintf('%-8d %-14.4f %-12.4f  KP=%.5f KI=%.5f KD=%.5f\n', ...
            episode, total_reward, ISE, KP, KI, KD);
    end
end

fprintf('\nDone! Best ISE=%.4f\n', best_ISE);
if all(best_yo==0); best_yo=yo; best_vv=vv; best_y=y; end

best_ny = zeros(1, N_t);
for it = 1:N_t
    best_ny(it) = sum(best_y(it,:).^2) * z_sample;
end

%% ── FIGURES ──────────────────────────────────────────────────────
set(0,'DefaultFigureVisible','on');

figure(1); clf; set(1,'Position',[100 500 560 420]);
mesh(zzz, ttt, best_y); view(-37.5+90, 30);
xlabel('x'); ylabel('t'); zlabel('C');
title('Concentration Distribution c(x,t)');
drawnow; pause(0.1);

figure(2); clf; set(2,'Position',[680 500 560 420]);
plot(ttt, sqrt(best_ny), 'k-', 'LineWidth', 1.5);
xlabel('t'); ylabel('||C||_2');
title('L2 Norm of Concentration');
grid on; drawnow; pause(0.1);

figure(3); clf; set(3,'Position',[100 50 560 420]);
plot(ttt, best_vv, 'b-', 'LineWidth', 1.5);
xlabel('t'); ylabel('v(t)');
title('Control Input v(t)');
grid on; drawnow; pause(0.1);

figure(5); clf; set(5,'Position',[680 50 560 420]);
plot(ttt, best_yo, 'b-', 'LineWidth', 2); hold on;
yline(ydd, 'r--', 'LineWidth', 2);
xlabel('Time t (s)'); ylabel('Outlet Concentration y(t) (g/L)');
title('Outlet Concentration - AC-RBF Incremental PID');
legend('AC-RBF PID','Target y* = 0.1 g/L','Location','northeast');
grid on;
ylim([0, max(max(best_yo)*1.1, 0.5)]);
drawnow; pause(0.1);

figure(6); clf; set(6,'Position',[100 50 800 500]);
fprintf('\nReward range: %.6f to %.6f\n', min(episode_reward), max(episode_reward));
fprintf('ISE range:    %.6f to %.6f\n',  min(episode_ISE),    max(episode_ISE));
subplot(2,1,1);
plot(1:num_episodes, episode_reward, 'b-o', 'LineWidth',1.5, 'MarkerSize',4);
xlabel('Episode'); ylabel('Total Reward');
title('Training Reward per Episode');
xlim([1 num_episodes]); grid on;
subplot(2,1,2);
plot(1:num_episodes, episode_ISE, 'r-o', 'LineWidth',1.5, 'MarkerSize',4);
xlabel('Episode'); ylabel('ISE');
title('ISE per Episode (lower = better)');
xlim([1 num_episodes]); grid on;
drawnow; pause(0.1);

figure(7); clf; set(7,'Position',[680 50 800 600]);
subplot(3,1,1);
plot(ttt, best_KP, 'b-', 'LineWidth', 1.2);
xlabel('Time t (s)'); ylabel('K_P');
title('Adaptive Gain K_P'); grid on;
subplot(3,1,2);
plot(ttt, best_KI, 'r-', 'LineWidth', 1.2);
xlabel('Time t (s)'); ylabel('K_I');
title('Adaptive Gain K_I'); grid on;
subplot(3,1,3);
plot(ttt, best_KD, 'g-', 'LineWidth', 1.2);
xlabel('Time t (s)'); ylabel('K_D');
title('Adaptive Gain K_D'); grid on;
drawnow; pause(0.1);

%% Fixed PID
fprintf('\nRunning Fixed PID...\n');
y_f=zeros(N_t,N_z); vv_f=zeros(1,N_t);
yo_f=zeros(1,N_t);  dd1_f=zeros(1,N_t);
for it=1:N_t; y_f(it,1)=cf; end
for iz=1:N_z; y_f(1,iz)=cf; end
yo_f(1)=y_f(1,N_z);
KPf=-0.001; KIf=-0.24; KDf=0.01;

for it=1:(N_t-1)
    for iz=2:N_z-1
        y_f(it+1,iz)=y_f(it,iz)+t_sample*(...
            D*(y_f(it,iz+1)-2*y_f(it,iz)+y_f(it,iz-1))/z_sample^2 ...
            -vv_f(it)*(y_f(it,iz)-y_f(it,iz-1))/z_sample ...
            -eta*y_f(it,iz)/(Km+y_f(it,iz)));
        if isnan(y_f(it+1,iz))||y_f(it+1,iz)<0
            y_f(it+1,iz)=y_f(it,iz);
        end
    end
    y_f(it+1,N_z)=y_f(it+1,N_z-1);
    yo_f(it+1)=max(y_f(it+1,N_z),0);
    dd1_f(it+1)=dd1_f(it)+t_sample*(yo_f(it+1)-ydd);
    dd1_f(it+1)=max(min(dd1_f(it+1),20),-20);
    vv_f(it+1)=KPf*(yo_f(it+1)-ydd)+KIf*dd1_f(it+1)...
              +KDf*(yo_f(it+1)-yo_f(it))/t_sample;
    vv_f(it+1)=max(min(vv_f(it+1),0.5),-0.5);
end

figure(10); clf; set(10,'Position',[100 50 800 500]);
plot(ttt, yo_f,    'b-', 'LineWidth', 2, 'DisplayName','Fixed PID');
hold on;
plot(ttt, best_yo, 'r-', 'LineWidth', 2, 'DisplayName','AC-RBF PID');
yline(ydd, 'k--',  'LineWidth', 1.5,     'DisplayName','Target y* = 0.1');
xlabel('Time t (s)'); ylabel('y(t) (g/L)');
title('Fixed PID vs Actor-Critic RBF Incremental PID');
legend('Location','northeast'); grid on;
drawnow; pause(0.1);

ISE_f=sum((yo_f   -ydd).^2)*t_sample;
ISE_d=sum((best_yo-ydd).^2)*t_sample;
IAE_f=sum(abs(yo_f   -ydd))*t_sample;
IAE_d=sum(abs(best_yo-ydd))*t_sample;
fprintf('\n========== Results ==========\n')
fprintf('Fixed PID:    ISE=%.4f  IAE=%.4f\n', ISE_f, IAE_f)
fprintf('AC-RBF PID:   ISE=%.4f  IAE=%.4f\n', ISE_d, IAE_d)
if ISE_f>0
    fprintf('ISE improvement: %.1f%%\n',(ISE_f-ISE_d)/ISE_f*100)
    fprintf('IAE improvement: %.1f%%\n',(IAE_f-IAE_d)/IAE_f*100)
end
fprintf('==============================\n')