function ctrl = make_controller(type, seed, opts)
%MAKE_CONTROLLER Create a controller struct with its persistent (learnable) state.
%   type : 'fixed'  - supervisor's fixed-gain positional PID (baseline)
%          'rbfpid' - RBF-network-tuned incremental PID (Jacobian identification,
%                     gradient gain tuning; Liu Jinkun, Advanced PID Control)
%          'acrbf'  - proposed actor-critic RBF incremental PID (thesis Sec. 4),
%                     corrected: sampled/learned at Ts, dy/dv = +1, TD(0) critic
%          'acrbf_v3' - original thesis implementation, kept for reference
%   seed : RNG seed for the random initialisation (acrbf only)
%   opts : optional struct overriding learning rates etc.

if nargin < 2 || isempty(seed), seed = 1; end
if nargin < 3, opts = struct(); end
ctrl.type = type;

switch type
    case 'fixed'
        ctrl.KP = -0.001; ctrl.KI = -0.24; ctrl.KD = 0.01;

    case 'acrbf'   % corrected AC-RBF: learns and acts at the control period Ts
        rng(seed);
        ctrl.alpha_w  = getopt(opts, 'alpha_w',  1e-3);
        ctrl.alpha_v  = getopt(opts, 'alpha_v',  0.05);
        ctrl.alpha_mu = getopt(opts, 'alpha_mu', 1e-3);
        ctrl.alpha_sg = getopt(opts, 'alpha_sg', 1e-3);
        ctrl.gamma    = getopt(opts, 'gamma', 0.98);
        ctrl.th_scale = getopt(opts, 'th_scale', [1; 0.1; 0.01]);  % e, De, D2e per Ts
        ctrl.dydv     = +1;   % physical sign: more flow -> higher outlet concentration
        ctrl.wa = 0.01*randn(3, 3);
        ctrl.wc = 0.01*randn(3, 1);
        ctrl.mu = 0.1*randn(3, 3);
        ctrl.sg = ones(1, 3);
        ctrl.Kmax = [0.1; 0.5; 0.05];   % Lyapunov clipping bounds, Eq. (18)

    case 'acrbf_v3'   % original thesis code (bioreactor_actor_critic_PID_v3.m), for reference
        rng(seed);
        ctrl.alpha_w  = getopt(opts, 'alpha_w',  0.001);
        ctrl.alpha_v  = getopt(opts, 'alpha_v',  0.005);
        ctrl.alpha_mu = getopt(opts, 'alpha_mu', 0.001);
        ctrl.alpha_sg = getopt(opts, 'alpha_sg', 0.001);
        ctrl.gamma    = 0.98;
        ctrl.wa = 0.01*randn(3, 3);
        ctrl.wc = 0.01*randn(3, 1);
        ctrl.mu = 0.1*randn(3, 3);
        ctrl.sg = ones(1, 3);
        ctrl.Kmax = [0.1; 0.5; 0.05];   % Lyapunov clipping bounds, Eq. (18)

    case 'rbfpid'
        % Gains are kept in Liu's sign convention (e = y* - y, gains >= 0);
        % in the thesis convention (e = y - y*) they are the negatives.
        % Algorithm = reference code (CSDN KeepingMatlab/136555215, Liu Jinkun's
        % RBF-tuned PID). Its literal settings (xite=0.5, xitekp/ki/kd=0.15, ci=0,
        % bi=10, w=0.1, K0=0.01) give a Jacobian estimate ~0 on this plant, so the
        % gains never adapt (see check_rbfpid_settings.m). The defaults below
        % rescale the identifier (du input x100, centres over y in [0,3], bi=1) so
        % that the gain tuning is actually active.
        ctrl.xite = getopt(opts, 'xite', 0.25);   % identifier learning rate
        ctrl.alfa = getopt(opts, 'alfa', 0.05);   % identifier momentum
        ctrl.eta_k = getopt(opts, 'eta_k', [0.2; 0.2; 0.2]);  % gain learning rates [P;I;D]
        ctrl.du_scale = getopt(opts, 'du_scale', 0.01);  % du / du_scale enters the identifier
        nh = 6;
        ctrl.w  = getopt(opts, 'w0', zeros(nh, 1));
        ctrl.ci = getopt(opts, 'ci0', [zeros(1, nh); linspace(0, 3, nh); linspace(0, 3, nh)]);
        ctrl.bi = getopt(opts, 'bi0', ones(nh, 1));
        ctrl.w_1 = ctrl.w;  ctrl.w_2 = ctrl.w;
        ctrl.ci_1 = ctrl.ci; ctrl.ci_2 = ctrl.ci;
        ctrl.bi_1 = ctrl.bi; ctrl.bi_2 = ctrl.bi;
        % Initial gains: incremental equivalent of the baseline P and I terms
        % at Ts = 0.1 s (KI*Ts = 0.024); this method cannot start from zero.
        ctrl.K = getopt(opts, 'K0', [0.001; 0.024; 0.001]);
        ctrl.Kmax = [0.1; 0.5; 0.05];   % same safety box as AC-RBF

    otherwise
        error('Unknown controller type %s', type);
end
end

function v = getopt(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end
