%% ================================================================
%% DDPG Direct PID - Final Clean Version
%% Uses positional PID — no accumulation drift
%% ================================================================
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

fprintf('DDPG Direct PID - Final Clean Version\n');
fprintf('N_t=%d, N_z=%d\n\n', N_t, N_z);

%% PART 2: Network Setup
state_dim  = 3;
action_dim = 3;
hidden     = 64;

actor_layers = [
    featureInputLayer(state_dim,    'Name','state')
    fullyConnectedLayer(hidden,     'Name','fc1')
    reluLayer(                      'Name','relu1')
    fullyConnectedLayer(hidden,     'Name','fc2')
    reluLayer(                      'Name','relu2')
    fullyConnectedLayer(action_dim, 'Name','fc3')
    tanhLayer(                      'Name','tanh')];
actor_net = dlnetwork(actor_layers);

critic_layers = [
    featureInputLayer(state_dim+action_dim, 'Name','input')
    fullyConnectedLayer(hidden,  'Name','fc1')
    reluLayer(                   'Name','relu1')
    fullyConnectedLayer(hidden,  'Name','fc2')
    reluLayer(                   'Name','relu2')
    fullyConnectedLayer(1,       'Name','fc3')];
critic_net = dlnetwork(critic_layers);

actor_target  = actor_net;
critic_target = critic_net;
actor_lr  = 1e-4;
critic_lr = 1e-3;
tau       = 0.005;
gamma     = 0.99;

%% Replay Buffer
buffer_size = 50000;
batch_size  = 64;
replay.s    = zeros(state_dim,  buffer_size);
replay.a    = zeros(action_dim, buffer_size);
replay.r    = zeros(1,          buffer_size);
replay.s2   = zeros(state_dim,  buffer_size);
replay.done = zeros(1,          buffer_size);
replay.ptr  = 1;
replay.size = 0;

%% PART 3: Training Setup
num_episodes = 100;
noise_std    = 0.2;
noise_decay  = 0.98;
noise_min    = 0.01;

episode_reward = zeros(1, num_episodes);
episode_ISE    = zeros(1, num_episodes);
best_ISE = inf;
best_yo  = zeros(1, N_t);
best_vv  = zeros(1, N_t);
best_y   = zeros(N_t, N_z);
best_KP  = zeros(1, N_t);
best_KI  = zeros(1, N_t);
best_KD  = zeros(1, N_t);

KP_scale    = 0.01;
KI_scale    = 0.8;
KD_scale    = 0.05;
ctrl_interval = 5000;
gain_smooth   = 0.3;

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

    KP = 0; KI = 0; KD = 0;
    eI = 0;
    last_action  = zeros(3,1);
    total_reward = 0;
    running_ISE  = 0;

    for it = 1:(N_t-1)

        %% PDE update
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

        %% Error terms
        e_t  = yo(it+1) - ydd;
        eI   = eI + t_sample*e_t;
        eI   = max(min(eI, 50), -50);
        eD_t = (yo(it+1) - yo(it)) / t_sample;

        %% Normalised state
        e_n  = max(min(e_t,   5),  -5);
        eI_n = max(min(eI,   50), -50);
        eD_n = max(min(eD_t, 50), -50);
        state = [e_n; eI_n; eD_n];

        %% DDPG updates gains every ctrl_interval steps
        if mod(it, ctrl_interval)==0 || it==1
            raw = get_action(actor_net, state, noise_std);
            last_action = raw;

            KP_new = raw(1) * KP_scale;
            KI_new = raw(2) * KI_scale;
            KD_new = raw(3) * KD_scale;

            %% Smooth gain transition
            KP = (1-gain_smooth)*KP + gain_smooth*KP_new;
            KI = (1-gain_smooth)*KI + gain_smooth*KI_new;
            KD = (1-gain_smooth)*KD + gain_smooth*KD_new;
        end
           KP = max(min(KP,  0.01), -0.01);
           KI = max(min(KI,  0.05), -0.05);
           KD = max(min(KD,  0.025), -0.025);

        %% POSITIONAL PID — computed every step from current gains
        %% NO accumulation, NO delta_v, NO v_current
        vv(it+1) = KP*e_t + KI*eI + KD*eD_t;
        vv(it+1) = max(min(vv(it+1), 0.5), -0.5);

        KP_hist(it+1) = KP;
        KI_hist(it+1) = KI;
        KD_hist(it+1) = KD;

        %% Reward
        if e_t < 0   % concentration below target
           reward = -2*e_t^2 - 0.001*vv(it+1)^2;
        else
           reward = -e_t^2 - 0.001*vv(it+1)^2;
        end
        total_reward = total_reward + reward;
        running_ISE  = running_ISE + e_t^2 * t_sample;

        %% Next state
        e2   = max(min(yo(it+1)-ydd,  5),  -5);
        eI2  = max(min(eI,            50), -50);
        eD2  = max(min(eD_t,          50), -50);
        next_state = [e2; eI2; eD2];

        %% Only store reasonable experiences
        is_reasonable = (best_ISE == inf) || ...
                        (running_ISE < best_ISE * 3);

        if mod(it, 50)==0 && is_reasonable
            ptr = mod(replay.ptr-1, buffer_size)+1;
            replay.s(:,ptr)   = state;
            replay.a(:,ptr)   = last_action;
            replay.r(ptr)     = reward;
            replay.s2(:,ptr)  = next_state;
            replay.done(ptr)  = (it==N_t-1);
            replay.ptr        = ptr+1;
            replay.size       = min(replay.size+1, buffer_size);
        end

        %% Learn every 500 steps
        if replay.size >= batch_size && mod(it,500)==0
            idx  = randperm(replay.size, batch_size);
            s_b  = dlarray(single(replay.s(:,idx)),  'CB');
            a_b  = dlarray(single(replay.a(:,idx)),  'CB');
            r_b  = single(replay.r(idx));
            s2_b = dlarray(single(replay.s2(:,idx)), 'CB');
            d_b  = single(replay.done(idx));

            a_tgt  = predict(actor_target,  s2_b);
            sa_tgt = cat(1, s2_b, a_tgt);
            q_tgt  = extractdata(predict(critic_target, sa_tgt));
            y_tgt  = dlarray(single(r_b + gamma*(1-d_b).*q_tgt), 'CB');

            [cg,~] = dlfeval(@critic_loss_fn, critic_net, s_b, a_b, y_tgt);
            critic_net = dlupdate(@(w,g) w - critic_lr*g, critic_net, cg);

            ag = dlfeval(@actor_loss_fn, actor_net, critic_net, s_b);
            actor_net = dlupdate(@(w,g) w - actor_lr*g, actor_net, ag);

            actor_target  = soft_update(actor_target,  actor_net,  tau);
            critic_target = soft_update(critic_target, critic_net, tau);
        end

    end % timestep loop

    noise_std = max(noise_std * noise_decay, noise_min);
    ISE = sum((yo - ydd).^2) * t_sample;
    episode_reward(episode) = total_reward;
    episode_ISE(episode)    = ISE;

    if ISE < best_ISE
        best_ISE = ISE;
        best_yo  = yo;
        best_vv  = vv;
        best_y   = y;
        best_KP  = KP_hist;
        best_KI  = KI_hist;
        best_KD  = KD_hist;
        fprintf('  -> Best ISE=%.4f at episode %d\n', best_ISE, episode);
    end

    if mod(episode,5)==0 || episode==1
        fprintf('%-8d %-14.4f %-12.4f  KP=%.5f KI=%.4f KD=%.5f\n', ...
            episode, total_reward, ISE, KP, KI, KD);
    end

end

fprintf('\nDone! Best ISE=%.4f\n', best_ISE);
if all(best_yo==0); best_yo=yo; best_vv=vv; best_y=y; end

%% L2 norm
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
title('Outlet Concentration - DDPG Direct PID');
legend('DDPG PID','Target y* = 0.1 g/L','Location','northeast');
grid on;
ylim([0, max(max(best_yo)*1.1, 0.5)]);
drawnow; pause(0.1);

figure(6); clf; set(6,'Position',[100 50 800 500]);
fprintf('\nReward range: %.4f to %.4f\n', min(episode_reward), max(episode_reward));
fprintf('ISE range:    %.4f to %.4f\n',  min(episode_ISE),    max(episode_ISE));
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
plot(ttt, best_KP, 'b-', 'LineWidth', 1.5);
xlabel('Time t (s)'); ylabel('K_P');
title('DDPG Output: K_P over Time'); grid on;
subplot(3,1,2);
plot(ttt, best_KI, 'r-', 'LineWidth', 1.5);
xlabel('Time t (s)'); ylabel('K_I');
title('DDPG Output: K_I over Time'); grid on;
subplot(3,1,3);
plot(ttt, best_KD, 'g-', 'LineWidth', 1.5);
xlabel('Time t (s)'); ylabel('K_D');
title('DDPG Output: K_D over Time'); grid on;
drawnow; pause(0.1);

%% Fixed PID comparison
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

fprintf('Fixed PID range: %.4f to %.4f\n', min(yo_f), max(yo_f));
fprintf('DDPG range:      %.4f to %.4f\n', min(best_yo), max(best_yo));

figure(10); clf; set(10,'Position',[100 50 800 500]);
plot(ttt, yo_f,    'b-', 'LineWidth',2, 'DisplayName','Fixed PID');
hold on;
plot(ttt, best_yo, 'r-', 'LineWidth',2, 'DisplayName','DDPG Direct PID');
yline(ydd, 'k--',  'LineWidth',1.5,     'DisplayName','Target y* = 0.1');
xlabel('Time t (s)'); ylabel('y(t) (g/L)');
title('Fixed PID vs DDPG Direct PID');
legend('Location','northeast'); grid on;
ylim([0, 0.5]);
drawnow; pause(0.1);

ISE_f=sum((yo_f   -ydd).^2)*t_sample;
ISE_d=sum((best_yo-ydd).^2)*t_sample;
IAE_f=sum(abs(yo_f   -ydd))*t_sample;
IAE_d=sum(abs(best_yo-ydd))*t_sample;
fprintf('\n========== Results ==========\n')
fprintf('Fixed PID:       ISE=%.4f  IAE=%.4f\n', ISE_f, IAE_f)
fprintf('DDPG Direct PID: ISE=%.4f  IAE=%.4f\n', ISE_d, IAE_d)
if ISE_f>0
    fprintf('ISE improvement: %.1f%%\n',(ISE_f-ISE_d)/ISE_f*100)
    fprintf('IAE improvement: %.1f%%\n',(IAE_f-IAE_d)/IAE_f*100)
end
fprintf('==============================\n')

%% Helper functions
function [grads,loss] = critic_loss_fn(critic,s,a,y_target)
    sa=cat(1,s,a); q=forward(critic,sa);
    loss=mean((q-y_target).^2,'all');
    grads=dlgradient(loss,critic.Learnables);
end
function grads = actor_loss_fn(actor,critic,s)
    a=forward(actor,s); sa=cat(1,s,a);
    q=forward(critic,sa); loss=-mean(q,'all');
    grads=dlgradient(loss,actor.Learnables);
end
function action = get_action(net,state,noise_std)
    s=dlarray(single(state),'CB');
    a=extractdata(predict(net,s));
    if noise_std>0; a=a+noise_std*randn(size(a)); end
    action=double(max(min(a,1),-1));
end
function target = soft_update(target,source,tau)
    tp=target.Learnables; sp=source.Learnables;
    for i=1:height(tp)
        tp.Value{i}=tau*sp.Value{i}+(1-tau)*tp.Value{i};
    end
    target.Learnables=tp;
end