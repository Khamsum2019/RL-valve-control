function [out, ctrl] = run_sim(ctrl, sc, P, store_field)
%RUN_SIM Simulate the bioreactor PDE (Eq. 8) in closed loop with one controller.
%   ctrl is returned with its learnable parameters updated, so calling run_sim
%   repeatedly with the returned ctrl = training over episodes (weights persist,
%   only the reactor state is reset).
%   out: t, y, v, K (3xNt gains in thesis convention e=y-y*), td (TD error,
%        AC-RBF only), reward (episode total), c (field every 100 steps, optional).

if nargin < 4, store_field = false; end
dt = P.dt; dx = P.dx; Nz = P.Nz;
Nt = round(sc.T/dt) + 1;
D = sc.D; eta = sc.eta; Km = sc.Km; ys = sc.ystar;
vmin = P.vmin; vmax = P.vmax; Nc = P.ctrl_interval;

y  = zeros(1, Nt); v = zeros(1, Nt); K = zeros(3, Nt); td = zeros(1, Nt);
c  = sc.c0(:).';
c(1) = sc.cf(1);
y(1) = max(c(Nz), 0);
if store_field, cfield = zeros(floor((Nt-1)/100)+1, Nz); cfield(1,:) = c; end
total_reward = 0;

% ---------------- episode-level controller state ----------------
switch ctrl.type
    case 'fixed'
        if sc.from_ss, I = sc.v0/ctrl.KI; else, I = 0; end
        v(1) = clamp(ctrl.KP*(y(1)-ys) + ctrl.KI*I, vmin, vmax);
        if ~sc.from_ss, v(1) = 0; end          % supervisor code: vv(1) = 0
        K(:,1) = [ctrl.KP; ctrl.KI; ctrl.KD];
    case 'acrbf'
        vcur = sc.v0; v(1) = vcur;
        if ~sc.from_ss, v(1) = 0; end
        e1 = y(1) - ys; e2 = e1; Vprev = 0; first = true;
        phi_p = zeros(3,1); xc_p = zeros(3,1); th_p = zeros(3,1);
        wa = ctrl.wa; wc = ctrl.wc; mu = ctrl.mu; sg = ctrl.sg; Kmax = ctrl.Kmax;
        Kc = min(max(wa'*phi_p, -Kmax), Kmax); K(:,1) = Kc;
    case 'acrbf_v3'
        vcur = sc.v0; e1 = y(1) - ys; e2 = 0; Vprev = 0;
        if sc.from_ss, v(1) = vcur; end         % thesis code: vv(1) = 0 at start-up
        wa = ctrl.wa; wc = ctrl.wc; mu = ctrl.mu; sg = ctrl.sg; Kmax = ctrl.Kmax;
    case 'rbfpid'
        vcur = sc.v0; v(1) = vcur;
        if ~sc.from_ss, v(1) = 0; end
        el1 = ys - y(1); el2 = el1; ysmp1 = y(1);
        x = [0; y(1); y(1)];
        K(:,1) = -ctrl.K;
end

for k = 1:Nt-1
    % ------------------------- plant ------------------------------
    ci = c(2:Nz-1);
    cn = ci + dt*(D*(c(3:Nz)-2*ci+c(1:Nz-2))/dx^2 ...
                  - v(k)*(ci-c(1:Nz-2))/dx - eta*ci./(Km+ci));
    bad = isnan(cn) | cn < 0;
    cn(bad) = ci(bad);
    c(2:Nz-1) = cn;
    c(1)  = sc.cf(k+1);
    c(Nz) = c(Nz-1);
    yk = max(c(Nz), 0);
    y(k+1) = yk;
    if store_field && mod(k, 100) == 0, cfield(k/100+1, :) = c; end

    % ------------------------ controller --------------------------
    switch ctrl.type
        case 'fixed'   % supervisor's positional PID, every time step
            e = yk - ys;
            I = clamp(I + dt*e, -20, 20);
            v(k+1) = clamp(ctrl.KP*e + ctrl.KI*I + ctrl.KD*(yk - y(k))/dt, vmin, vmax);
            K(:,k+1) = K(:,1);

        case 'acrbf'   % corrected AC-RBF, sampled at Ts = ctrl_interval*dt
            if mod(k, Nc) == 0
                e   = yk - ys;
                De  = e - e1;
                D2e = e - 2*e1 + e2;
                xc  = [De; e; D2e];                     % multiplies [KP; KI; KD]
                th  = clamp([e; De; D2e]./ctrl.th_scale, -5, 5);
                phi = zeros(3, 1);
                for j = 1:3
                    d = th - mu(:,j);
                    phi(j) = exp(-(d'*d)/(2*sg(j)^2 + 1e-8));
                end
                R  = 0.5*e^2;                           % cost incurred by the previous action
                Vc = wc'*phi;
                if ~first
                    dTD = clamp(R + ctrl.gamma*Vc - Vprev, -10, 10);
                    td(k+1) = dTD;
                    % actor: dC/dK = dTD * (dR/dy) * (dy/dv) * (dv/dK), credited to
                    % the gains that produced this outcome (phi_p, xc_p)
                    g = dTD*e*ctrl.dydv;
                    wa = wa - ctrl.alpha_w*g*(phi_p*xc_p');
                    % critic: TD(0) semi-gradient on V(t-1)
                    wc = wc + ctrl.alpha_v*dTD*phi_p;
                    for j = 1:3
                        d = th_p - mu(:,j);
                        mu(:,j) = mu(:,j) + ctrl.alpha_mu*dTD*wc(j)*phi_p(j)*d/(sg(j)^2 + 1e-8);
                        sg(j) = max(sg(j) + ctrl.alpha_sg*dTD*wc(j)*phi_p(j)*(d'*d)/(sg(j)^3 + 1e-8), 0.01);
                    end
                    total_reward = total_reward - R;
                end
                Kc = min(max(wa'*phi, -Kmax), Kmax);
                vcur = clamp(vcur + Kc'*xc, vmin, vmax);
                phi_p = phi; xc_p = xc; th_p = th; Vprev = wc'*phi;
                e2 = e1; e1 = e; first = false;
            end
            v(k+1) = vcur;
            K(:,k+1) = Kc;

        case 'acrbf_v3'   % thesis algorithm, line-for-line with v3 script
            e   = clamp(yk - ys, -5, 5);
            De  = clamp(e - e1, -5, 5);
            D2e = clamp(e - 2*e1 + e2, -5, 5);
            th  = [e; -De; -D2e];
            phi = zeros(3, 1);
            for j = 1:3
                d = th - mu(:,j);
                phi(j) = exp(-(d'*d)/(2*sg(j)^2 + 1e-8));
            end
            Kr = wa'*phi;
            Kc = min(max(Kr, -Kmax), Kmax);
            if mod(k, Nc) == 0 || k == 1
                vcur = clamp(vcur + Kc(1)*De + Kc(2)*e + Kc(3)*D2e, vmin, vmax);
            end
            v(k+1) = vcur;
            K(:,k+1) = Kc;

            R  = 0.5*e^2;
            Vc = wc'*phi;
            dTD = clamp(R + ctrl.gamma*Vc - Vprev, -10, 10);
            td(k+1) = dTD;
            dydu = -1;
            wa(:,1) = wa(:,1) - ctrl.alpha_w*dTD*e*De*phi*dydu;
            wa(:,2) = wa(:,2) + ctrl.alpha_w*dTD*e^2*phi*dydu;
            wa(:,3) = wa(:,3) - ctrl.alpha_w*dTD*e*D2e*phi*dydu;
            wc = wc + ctrl.alpha_v*dTD*phi;
            for j = 1:3
                d = th - mu(:,j);
                mu(:,j) = mu(:,j) + ctrl.alpha_mu*dTD*wc(j)*phi(j)*d/(sg(j)^2 + 1e-8);
                d = th - mu(:,j);
                sg(j) = max(sg(j) + ctrl.alpha_sg*dTD*wc(j)*phi(j)*(d'*d)/(sg(j)^3 + 1e-8), 0.01);
            end
            e2 = e1; e1 = e; Vprev = Vc;
            total_reward = total_reward - R;

        case 'rbfpid'  % RBF identifier + gradient gain tuning, every 0.1 s
            if mod(k, Nc) == 0
                [ctrl, Kn, du, x, el1, el2, ysmp1] = rbfpid_update(ctrl, x, yk, ys, el1, el2, ysmp1);
                vcur = clamp(vcur + du, vmin, vmax);
                K(:,k+1) = -Kn;
            else
                K(:,k+1) = K(:,k);
            end
            v(k+1) = vcur;
    end
end

if any(strcmp(ctrl.type, {'acrbf', 'acrbf_v3'}))
    ctrl.wa = wa; ctrl.wc = wc; ctrl.mu = mu; ctrl.sg = sg;
end

out.t = (0:Nt-1)*dt;
out.y = y; out.v = v; out.K = K; out.td = td; out.cf = sc.cf;
out.reward = total_reward;
if store_field
    out.c = cfield;
    out.tc = (0:size(cfield,1)-1)*100*dt;
end
end

function [ctrl, Kn, du, x, el1, el2, ysmp1] = rbfpid_update(ctrl, x, yk, ys, el1, el2, ysmp1)
% --- online identification of y(k) = f(du(k-1), y(k-1), y(k-2)) ---
nh = numel(ctrl.w);
dist2 = sum((x - ctrl.ci).^2, 1)';
h  = exp(-dist2./(2*ctrl.bi.^2));
em = yk - ctrl.w'*h;
w  = ctrl.w_1 + ctrl.xite*em*h + ctrl.alfa*(ctrl.w_1 - ctrl.w_2);
bi = ctrl.bi_1 + ctrl.xite*em*w.*h.*dist2./ctrl.bi.^3 + ctrl.alfa*(ctrl.bi_1 - ctrl.bi_2);
bi = max(bi, 0.05);
ci = ctrl.ci_1 + ctrl.xite*em*((w.*h./bi.^2)'.*(x - ctrl.ci)) + ctrl.alfa*(ctrl.ci_1 - ctrl.ci_2);
ctrl.w_2 = ctrl.w_1; ctrl.w_1 = w; ctrl.w = w;
ctrl.bi_2 = ctrl.bi_1; ctrl.bi_1 = bi; ctrl.bi = bi;
ctrl.ci_2 = ctrl.ci_1; ctrl.ci_1 = ci; ctrl.ci = ci;
% --- Jacobian estimate dy/du ---
h  = exp(-sum((x - ci).^2, 1)'./(2*bi.^2));
yu = sum(w.*h.*(ci(1,:)' - x(1))./bi.^2)/ctrl.du_scale;
% --- gradient tuning of the PID gains (Liu convention e = y* - y) ---
e  = ys - yk;
xc = [e - el1; e; e - 2*el1 + el2];
Kn = ctrl.K + ctrl.eta_k.*(e*yu*xc);
Kn = min(max(Kn, 0), ctrl.Kmax);
ctrl.K = Kn;
du = Kn'*xc;
x  = [du/ctrl.du_scale; yk; ysmp1];
el2 = el1; el1 = e; ysmp1 = yk;
assert(numel(h) == nh);
end

function z = clamp(z, lo, hi)
z = min(max(z, lo), hi);
end
