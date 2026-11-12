%% EXP2_CASE_STUDIES  Fixed PID vs RBF-PID vs AC-RBF PID (Muduli-style case studies)
%   Case 1  start-up from c(x,0) = cf                    (training scenario)
%   Case 2  multiple step changes of inlet concentration  (from steady state)
%   Case 3  plant-parameter uncertainty (nominal / set A / set B) + inlet steps
%   Case 4  random, band-limited inlet concentration
% Indices: ISE, IAE, ITAE (and settling time for Case 1).
%   Fixed PID : supervisor's gains, positional form, no learning
%   RBF-PID   : RBF identifier + gradient gain tuning, purely online, single pass,
%               starts from hand-chosen initial gains (it cannot start from zero)
%   AC-RBF    : trained N_TRAIN episodes on Case 1 from zero gains, then keeps
%               adapting online in every test case; 5 seeds (mean, std reported)
clear; close all; clc;
here = fileparts(mfilename('fullpath')); addpath(here);
figdir = fullfile(here, 'figures'); if ~exist(figdir, 'dir'), mkdir(figdir); end
resdir = fullfile(here, 'results'); if ~exist(resdir, 'dir'), mkdir(resdir); end

N_TRAIN = 100;      % chosen from exp1_episode_study
seeds   = 1:5;

P = params_default();
[ss.c, ss.v] = steady_state(P);
fprintf('Steady state: v* = %.5f m/s\n', ss.v);

cases  = {'startup', 'steps', 'nominal_p', 'paramA', 'paramB', 'random'};
labels = {'Case 1: start-up', 'Case 2: inlet steps', 'Case 3: nominal parameters', ...
          'Case 3: parameter set A', 'Case 3: parameter set B', 'Case 4: random inlet'};
meth   = {'Fixed PID', 'RBF-PID', 'AC-RBF PID'};
col    = {[0.85 0.33 0.10], [0.30 0.65 0.20], [0.00 0.45 0.74]};
sty    = {'--', '-.', '-'};
nC = numel(cases);

%% ---------------- train AC-RBF (all seeds) ----------------
ac = cell(1, numel(seeds));
for is = 1:numel(seeds)
    [~, sn] = train_adaptive(make_controller('acrbf', seeds(is)), ...
                             make_scenario('startup', P, ss), P, N_TRAIN, N_TRAIN);
    ac{is} = sn{1};
end
fprintf('AC-RBF trained (%d episodes x %d seeds)\n', N_TRAIN, numel(seeds));

%% ---------------- run all cases ----------------
OUT = cell(nC, 3);          % traces used for figures (AC-RBF: seed 1)
M = struct('ISE', nan(nC, 3), 'IAE', nan(nC, 3), 'ITAE', nan(nC, 3), 'Ts', nan(nC, 3));
Msd = struct('ISE', nan(nC, 1), 'IAE', nan(nC, 1), 'ITAE', nan(nC, 1), 'Ts', nan(nC, 1));
for ic = 1:nC
    sc = make_scenario(cases{ic}, P, ss);
    store = strcmp(cases{ic}, 'startup');
    OUT{ic,1} = run_sim(make_controller('fixed'),  sc, P, store);
    OUT{ic,2} = run_sim(make_controller('rbfpid'), sc, P, store);
    for im = 1:2
        m = compute_metrics(OUT{ic,im}, P.ystar, P.settle_band);
        M.ISE(ic,im) = m.ISE; M.IAE(ic,im) = m.IAE; M.ITAE(ic,im) = m.ITAE; M.Ts(ic,im) = m.Ts;
    end
    A = zeros(numel(seeds), 4);
    for is = 1:numel(seeds)
        o = run_sim(ac{is}, sc, P, store && is == 1);
        if is == 1, OUT{ic,3} = o; end
        m = compute_metrics(o, P.ystar, P.settle_band);
        A(is,:) = [m.ISE m.IAE m.ITAE m.Ts];
    end
    M.ISE(ic,3) = mean(A(:,1)); M.IAE(ic,3) = mean(A(:,2));
    M.ITAE(ic,3) = mean(A(:,3)); M.Ts(ic,3) = mean(A(:,4));
    Msd.ISE(ic) = std(A(:,1)); Msd.IAE(ic) = std(A(:,2));
    Msd.ITAE(ic) = std(A(:,3)); Msd.Ts(ic) = std(A(:,4));
end
save(fullfile(resdir, 'exp2_case_studies.mat'), 'M', 'Msd', 'cases', 'labels', 'meth', 'N_TRAIN', 'seeds');

%% ---------------- tables ----------------
imp = @(base, new) 100*(base - new)/base;
fid = fopen(fullfile(resdir, 'exp2_metrics.txt'), 'w');
for out = [1 fid]
    fprintf(out, 'Three-method comparison (AC-RBF: %d training episodes, mean over %d seeds; std in brackets)\n\n', ...
        N_TRAIN, numel(seeds));
    for ic = 1:nC
        fprintf(out, '%s\n', labels{ic});
        fprintf(out, '  %-11s %-20s %-20s %-22s', 'Method', 'ISE', 'IAE', 'ITAE');
        if ic == 1, fprintf(out, ' %s', 'Ts (s)'); end
        fprintf(out, '\n');
        for im = 1:3
            if im < 3
                fprintf(out, '  %-11s %-20.5f %-20.5f %-22.4f', meth{im}, M.ISE(ic,im), M.IAE(ic,im), M.ITAE(ic,im));
                if ic == 1, fprintf(out, ' %.2f', M.Ts(ic,im)); end
            else
                fprintf(out, '  %-11s %-20s %-20s %-22s', meth{im}, ...
                    sprintf('%.5f (%.1e)', M.ISE(ic,3), Msd.ISE(ic)), ...
                    sprintf('%.5f (%.1e)', M.IAE(ic,3), Msd.IAE(ic)), ...
                    sprintf('%.4f (%.1e)', M.ITAE(ic,3), Msd.ITAE(ic)));
                if ic == 1, fprintf(out, ' %.2f (%.2f)', M.Ts(ic,3), Msd.Ts(ic)); end
            end
            fprintf(out, '\n');
        end
        fprintf(out, '  AC-RBF improvement vs Fixed PID: ISE %.1f%%, IAE %.1f%%, ITAE %.1f%%\n', ...
            imp(M.ISE(ic,1), M.ISE(ic,3)), imp(M.IAE(ic,1), M.IAE(ic,3)), imp(M.ITAE(ic,1), M.ITAE(ic,3)));
        fprintf(out, '  AC-RBF improvement vs RBF-PID:   ISE %.1f%%, IAE %.1f%%, ITAE %.1f%%\n\n', ...
            imp(M.ISE(ic,2), M.ISE(ic,3)), imp(M.IAE(ic,2), M.IAE(ic,3)), imp(M.ITAE(ic,2), M.ITAE(ic,3)));
    end
end
fclose(fid);
fid = fopen(fullfile(resdir, 'exp2_metrics.tex'), 'w');
for ic = 1:nC
    fprintf(fid, '%% %s\n', labels{ic});
    for im = 1:3
        fprintf(fid, '%s & %.4f & %.4f & %.4f', meth{im}, M.ISE(ic,im), M.IAE(ic,im), M.ITAE(ic,im));
        if ic == 1, fprintf(fid, ' & %.1f', M.Ts(ic,im)); end
        fprintf(fid, ' \\\\\n');
    end
end
fclose(fid);

%% ---------------- figures ----------------
% Case 1: outlet concentration with zoom
o = OUT(1,:);
fig = figure('Position', [100 100 760 420], 'Color', 'w');
for im = 1:3, plot(o{im}.t, o{im}.y, sty{im}, 'Color', col{im}, 'LineWidth', 1.6); hold on; end
yline(P.ystar, 'k:', 'LineWidth', 1.2);
xlabel('Time t (s)'); ylabel('Outlet concentration y(t) (g/L)'); grid on;
legend([meth, {'Target y^* = 0.1 g/L'}], 'Location', 'northeast');
title('Case 1: start-up');
axes('Position', [0.40 0.35 0.45 0.33]); box on;
for im = 1:3, plot(o{im}.t, o{im}.y, sty{im}, 'Color', col{im}, 'LineWidth', 1.3); hold on; end
yline(P.ystar, 'k:'); xlim([0 100]); ylim([0 0.2]); grid on; title('zoom');
exportgraphics(fig, fullfile(figdir, 'case1_outlet.png'), 'Resolution', 300);

fig = figure('Position', [100 100 760 360], 'Color', 'w');
for im = 1:3, plot(o{im}.t, o{im}.v, sty{im}, 'Color', col{im}, 'LineWidth', 1.4); hold on; end
xlabel('Time t (s)'); ylabel('Flow rate v(t) (m/s)'); grid on; legend(meth, 'Location', 'northeast');
title('Case 1: control input');
exportgraphics(fig, fullfile(figdir, 'case1_control.png'), 'Resolution', 300);

plot_gains_td(o{3}, o{2}, 'Case 1', fullfile(figdir, 'case1_gains_td.png'));

fig = figure('Position', [100 100 600 450], 'Color', 'w');
zz = (0:P.Nz-1)*P.dx;
mesh(zz, o{3}.tc, o{3}.c); view(-37.5+90, 30);
xlabel('x (m)'); ylabel('t (s)'); zlabel('c(x,t) (g/L)');
title('Case 1: concentration field, AC-RBF PID');
exportgraphics(fig, fullfile(figdir, 'case1_field.png'), 'Resolution', 300);

% Case 2: inlet steps
plot_disturbance_case(OUT(2,:), meth, col, sty, P, 'Case 2: step changes of inlet concentration', ...
    fullfile(figdir, 'case2_steps.png'));
plot_gains_td(OUT{2,3}, OUT{2,2}, 'Case 2', fullfile(figdir, 'case2_gains_td.png'));

% Case 3: parameter uncertainty
fig = figure('Position', [100 100 760 760], 'Color', 'w');
for k = 1:3
    ic = 2 + k;
    subplot(3, 1, k);
    for im = 1:3
        plot(OUT{ic,im}.t, OUT{ic,im}.y, sty{im}, 'Color', col{im}, 'LineWidth', 1.4); hold on;
    end
    yline(P.ystar, 'k:'); grid on; ylabel('y(t) (g/L)'); title(labels{ic});
    if k == 1, legend(meth, 'Location', 'northeast'); end
end
xlabel('Time t (s)');
exportgraphics(fig, fullfile(figdir, 'case3_parameters.png'), 'Resolution', 300);

fig = figure('Position', [100 100 760 360], 'Color', 'w');
lin = {'-', '--', ':'};
for k = 1:3
    plot(OUT{2+k,3}.t, OUT{2+k,3}.y, lin{k}, 'Color', col{3}, 'LineWidth', 1.5); hold on;
end
yline(P.ystar, 'k:'); grid on; xlabel('Time t (s)'); ylabel('y(t) (g/L)');
legend({'Nominal', 'Set A (\eta -3%, D +5%, K_m +3%)', 'Set B (\eta +20%, D -20%, K_m -20%)'}, 'Location', 'northeast');
title('Case 3: AC-RBF PID under parameter variations');
exportgraphics(fig, fullfile(figdir, 'case3_acrbf_sets.png'), 'Resolution', 300);

% Case 4: random inlet concentration + error distributions
plot_disturbance_case(OUT(6,:), meth, col, sty, P, 'Case 4: random inlet concentration', ...
    fullfile(figdir, 'case4_random.png'));
fig = figure('Position', [100 100 620 380], 'Color', 'w');
xs = linspace(-0.03, 0.03, 400);
for im = 1:3
    e = OUT{6,im}.y - P.ystar; mu_ = mean(e); sd = std(e);
    plot(xs, exp(-(xs-mu_).^2/(2*sd^2))/(sd*sqrt(2*pi)), sty{im}, 'Color', col{im}, 'LineWidth', 1.6, ...
        'DisplayName', sprintf('%s: \\mu = %.2e, \\sigma = %.4f', meth{im}, mu_, sd)); hold on;
end
xlabel('Tracking error e = y - y^* (g/L)'); ylabel('Probability density'); grid on;
legend('Location', 'northeast'); title('Case 4: normal fit of the tracking error');
exportgraphics(fig, fullfile(figdir, 'case4_error_distribution.png'), 'Resolution', 300);

fprintf('\nFigures saved to %s\n', figdir);

%% ---------------- local plotting helpers ----------------
function plot_disturbance_case(o, meth, col, sty, P, ttl, file)
fig = figure('Position', [100 100 760 720], 'Color', 'w');
subplot(3, 1, 1);
plot(o{1}.t, o{1}.cf, 'k-', 'LineWidth', 1.3); grid on; ylabel('c_f(t) (g/L)'); title(ttl);
subplot(3, 1, 2);
for im = 1:3, plot(o{im}.t, o{im}.y, sty{im}, 'Color', col{im}, 'LineWidth', 1.4); hold on; end
yline(P.ystar, 'k:'); grid on; ylabel('y(t) (g/L)'); legend(meth, 'Location', 'northeast');
subplot(3, 1, 3);
for im = 1:3, plot(o{im}.t, o{im}.v, sty{im}, 'Color', col{im}, 'LineWidth', 1.4); hold on; end
grid on; ylabel('v(t) (m/s)'); xlabel('Time t (s)');
exportgraphics(fig, file, 'Resolution', 300);
end

function plot_gains_td(oa, orb, ttl, file)
fig = figure('Position', [100 100 760 820], 'Color', 'w');
nm = {'K_P', 'K_I', 'K_D'};
for i = 1:3
    subplot(4, 1, i);
    plot(oa.t, oa.K(i,:), 'Color', [0 0.45 0.74], 'LineWidth', 1.3); hold on;
    plot(orb.t, orb.K(i,:), '-.', 'Color', [0.30 0.65 0.20], 'LineWidth', 1.2);
    grid on; ylabel(nm{i});
    if i == 1, title([ttl ': adaptive gains (thesis sign convention, e = y - y^*)']);
               legend('AC-RBF PID', 'RBF-PID', 'Location', 'best'); end
end
subplot(4, 1, 4);
idx = oa.td ~= 0;
plot(oa.t(idx), oa.td(idx), 'Color', [0 0.45 0.74], 'LineWidth', 1.1);
grid on; ylabel('\delta_{TD}'); xlabel('Time t (s)'); title('AC-RBF temporal-difference error');
exportgraphics(fig, file, 'Resolution', 300);
end
