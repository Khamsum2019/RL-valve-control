%clear all;
close all;
clc;

%% ─── PART 1: Setup (supervisor's code - don't change) ───────────────────
zL=0; zU=1;
t0=0; tf=100;
t_sample=0.001; z_sample=0.05;
N_t=round((tf-t0)/t_sample)+1;
N_z=round((zU-zL)/z_sample)+1;

D=0.1;
Km=1;
eta=3;
cf=5;

for it=1:N_t
    ttt(it)=(it-1)*t_sample;
    vv(it)=0;
    yo(it)=0;
    dd1(it)=0;
    y(it,1)=cf;
    ny(it)=0;
end

ydd=0.3;  % target outlet concentration

for iz=1:N_z
    zzz(iz)=(iz-1)*z_sample;
    y(1,iz)=cf;
end
yo(1)=y(1,N_z);

%% ─── PART 2: PID Initial Gains (supervisor's code) ──────────────────────
KP = -0.001;
KI = -0.24;
KD = 0.01;

%% ─── PART 3: DDPG Setup (YOUR ADDITION) ─────────────────────────────────

% State space - equation (3-12)
% s(t) = [e(t), eI(t), eD(t), cin(t)]
obsInfo = rlNumericSpec([4 1], ...
    'LowerLimit', [-inf; -inf; -inf; 0], ...
    'UpperLimit', [inf; inf; inf; inf]);
obsInfo.Name = 'observations';
obsInfo.Description = 'error, error integral, error derivative, inlet concentration';

% Action space - equation (3-13)
% a(t) = [deltaKp, deltaKi, deltaKd]
actInfo = rlNumericSpec([3 1], ...
    'LowerLimit', [-0.2; -0.5; 0.0], ...
    'UpperLimit', [0.0; 0.0; 0.1]);
actInfo.Name = 'PID gain adjustments';

% Create Actor network (decides what to do)
actorNetwork = [
    featureInputLayer(4, 'Name', 'state')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(3, 'Name', 'output')
    tanhLayer('Name', 'tanh')
];
actorNetwork = dlnetwork(actorNetwork);

% Create Critic network (evaluates decisions)
statePath = [
    featureInputLayer(4, 'Name', 'state')
    fullyConnectedLayer(64, 'Name', 'fc_s1')
    reluLayer('Name', 'relu_s1')
];
actionPath = [
    featureInputLayer(3, 'Name', 'action')
    fullyConnectedLayer(64, 'Name', 'fc_a1')
    reluLayer('Name', 'relu_a1')
];
commonPath = [
    additionLayer(2, 'Name', 'add')
    fullyConnectedLayer(64, 'Name', 'fc_common')
    reluLayer('Name', 'relu_common')
    fullyConnectedLayer(1, 'Name', 'Q_value')
];
criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork, statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);
criticNetwork = connectLayers(criticNetwork, 'relu_s1', 'add/in1');
criticNetwork = connectLayers(criticNetwork, 'relu_a1', 'add/in2');
criticNetwork = dlnetwork(criticNetwork);

% Create Actor and Critic objects
actor  = rlContinuousDeterministicActor(actorNetwork, obsInfo, actInfo, ...
    'ObservationInputNames', {'state'});

critic = rlQValueFunction(criticNetwork, obsInfo, actInfo, ...
    'ObservationInputNames', {'state'}, ...
    'ActionInputNames',      {'action'});

% DDPG Agent Options
agentOpts = rlDDPGAgentOptions(...
    'SampleTime',                          t_sample, ...
    'TargetSmoothFactor',                  1e-3,     ...
    'DiscountFactor',                      0.99,     ...
    'MiniBatchSize',                       64,       ...
    'ExperienceBufferLength',              10000);

% Create DDPG agent
agent = rlDDPGAgent(actor, critic, agentOpts);

% Experience buffer to store (state, action, reward, next_state)
expBuffer.states      = zeros(4, 10000);
expBuffer.actions     = zeros(3, 10000);
expBuffer.rewards     = zeros(1, 10000);
expBuffer.nextStates  = zeros(4, 10000);
expBuffer.idx         = 1;
expBuffer.count       = 0;

%% ─── PART 4: Main Simulation Loop ───────────────────────────────────────

eD_prev = 0;  % previous error for derivative calculation

for it=1:(N_t-1)

    %% Step A: Update concentration in reactor (supervisor's PDE code)
    for iz=2:N_z-1
        y(it+1,iz) = y(it,iz) + t_sample*( ...
            D*(y(it,iz+1)-2*y(it,iz)+y(it,iz-1))/z_sample/z_sample ...
            - vv(it)*(y(it,iz)-y(it,iz-1))/z_sample ...
            - eta*y(it,iz)/(Km+y(it,iz)));
    end

    %% Step B: Outlet boundary condition (supervisor's code)
    y(it+1,N_z) = y(it+1,N_z-1);

    %% Step C: Measure outlet concentration (supervisor's code)
    yo(it+1) = y(it+1,N_z);

    %% Step D: Error integral (supervisor's code)
    for i=1:it+1
        dd1(it+1) = dd1(it+1) + t_sample*(yo(i)-ydd);
    end

    %% ── YOUR DDPG ADDITION STARTS HERE ──────────────────────────────────

    % Build state vector - equation (3-12)
    e_t  = yo(it+1) - ydd;             % tracking error e(t)
    eI_t = dd1(it+1);                  % error integral eI(t)
    eD_t = (yo(it+1)-yo(it))/t_sample; % error derivative eD(t)
    cin  = cf;                          % inlet concentration

    currentState = [e_t; eI_t; eD_t; cin];

    % DDPG selects action - equation (3-13)
    % action = [deltaKp, deltaKi, deltaKd]
    action = getAction(agent, {currentState});
    action = action{1};

    % Update PID gains using DDPG output
    KP = KP + action(1);
    KI = KI + action(2);
    KD = KD + action(3);

    % Keep gains within safe bounds
    KP = max(min(KP, -0.0001), -0.01);
    KI = max(min(KI, -0.1),    -0.5 );
    KD = max(min(KD,  0.1),     0.0 );

    %% ── YOUR DDPG ADDITION ENDS HERE ────────────────────────────────────

    %% Step E: PID control law - equation (3-9) (same formula, adaptive gains)
    vv(it+1) = KP*(yo(it+1)-ydd) ...
             + KI*dd1(it+1) ...
             + KD*(yo(it+1)-yo(it))/t_sample;

    %% Step F: Calculate reward - equation (3-14)
    % reward = -alpha*e² - beta*|delta_v| - gamma*||delta_gains||²
    alpha   = 1.0;
    beta    = 0.1;
    gamma_r = 0.05;

    reward = -alpha * e_t^2 ...
             - beta  * abs(vv(it+1) - vv(it)) ...
             - gamma_r * (action(1)^2 + action(2)^2 + action(3)^2);

    %% Step G: Build next state for experience storage
    if it < N_t-1
        e_next  = yo(it+1) - ydd;
        eI_next = dd1(it+1);
        eD_next = (yo(it+1)-yo(it))/t_sample;
        nextState = [e_next; eI_next; eD_next; cin];
    else
        nextState = currentState;
    end

    %% Step H: Store experience in buffer
    idx = mod(expBuffer.idx-1, 10000)+1;
    expBuffer.states(:,idx)     = currentState;
    expBuffer.actions(:,idx)    = action;
    expBuffer.rewards(idx)      = reward;
    expBuffer.nextStates(:,idx) = nextState;
    expBuffer.idx               = expBuffer.idx + 1;
    expBuffer.count             = min(expBuffer.count+1, 10000);

end

%% ─── PART 5: L2 Norm (supervisor's code) ────────────────────────────────
for it=1:N_t
    for iz=1:N_z
        ny(it) = ny(it) + y(it,iz)*y(it,iz)*z_sample;
    end
end

%% ─── PART 6: Plot Results ────────────────────────────────────────────────
figure(1)
mesh(zzz,ttt,y)
view(-37.5+90,30);
xlabel('x'); ylabel('t'); zlabel('C');
title('Concentration Distribution c(x,t)');

figure(2)
plot(ttt, sqrt(ny), 'k-', 'LineWidth', 1.5)
xlabel('t'); ylabel('||C||_2');
title('L2 Norm of Concentration');

figure(3)
plot(ttt, vv, 'b-', 'LineWidth', 1.5)
xlabel('t'); ylabel('v(t)');
title('Control Input - Flow Speed');

figure(5)
plot(ttt, yo, 'b-', 'LineWidth', 2)
hold on
yline(ydd, 'r--', 'LineWidth', 1.5, 'Label', 'Target y_d = 0.1')
xlabel('t'); ylabel('y_{out}(t)');
title('Outlet Concentration Tracking');
legend('DDPG PID', 'Target');
grid on