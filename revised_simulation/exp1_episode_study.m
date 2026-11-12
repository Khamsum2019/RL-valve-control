%% EXP1_EPISODE_STUDY  How many training episodes does AC-RBF need?
% Trains the AC-RBF controller on the start-up scenario for N_max episodes with
% several random seeds. Network snapshots after 30, 50, 100 (and 200) episodes
% are evaluated with ISE, IAE and ITAE on
%   (a) the training scenario (Case 1, start-up)  and
%   (b) an unseen disturbance scenario (Case 2, inlet steps).
% The episode count is chosen as the smallest N after which longer training no
% longer gives a meaningful improvement in the three indices.
clear; close all; clc;
here = fileparts(mfilename('fullpath')); addpath(here);
figdir = fullfile(here, 'figures'); if ~exist(figdir, 'dir'), mkdir(figdir); end
resdir = fullfile(here, 'results'); if ~exist(resdir, 'dir'), mkdir(resdir); end

P = params_default();
[ss.c, ss.v] = steady_state(P);
scS = make_scenario('startup', P, ss);
scT = make_scenario('steps',   P, ss);

seeds   = 1:5;
N_max   = 200;
snap_at = [30 50 100 200];
nS = numel(seeds); nN = numel(snap_at);

H = struct('ISE', zeros(nS, N_max), 'IAE', zeros(nS, N_max), ...
           'ITAE', zeros(nS, N_max), 'reward', zeros(nS, N_max));
E = struct();   % evaluation of snapshots: [seed x N] per metric per scenario
for f = {'S_ISE','S_IAE','S_ITAE','S_Ts','T_ISE','T_IAE','T_ITAE','time'}
    E.(f{1}) = zeros(nS, nN);
end

for is = 1:nS
    ctrl = make_controller('acrbf', seeds(is));
    t_start = tic;
    [h, snaps] = train_adaptive(ctrl, scS, P, N_max, snap_at);
    t_total = toc(t_start);
    H.ISE(is,:) = h.ISE; H.IAE(is,:) = h.IAE; H.ITAE(is,:) = h.ITAE; H.reward(is,:) = h.reward;
    for in = 1:nN
        mS = compute_metrics(run_sim(snaps{in}, scS, P), P.ystar, P.settle_band);
        mT = compute_metrics(run_sim(snaps{in}, scT, P), P.ystar, P.settle_band);
        E.S_ISE(is,in) = mS.ISE; E.S_IAE(is,in) = mS.IAE; E.S_ITAE(is,in) = mS.ITAE; E.S_Ts(is,in) = mS.Ts;
        E.T_ISE(is,in) = mT.ISE; E.T_IAE(is,in) = mT.IAE; E.T_ITAE(is,in) = mT.ITAE;
        E.time(is,in) = t_total*snap_at(in)/N_max;
    end
    fprintf('seed %d done (%.0f s)\n', seeds(is), t_total);
end
save(fullfile(resdir, 'exp1_episode_study.mat'), 'H', 'E', 'seeds', 'snap_at', 'N_max');

%% ---------------- table: mean +/- std over seeds ----------------
fid = fopen(fullfile(resdir, 'exp1_episode_table.txt'), 'w');
for out = [1 fid]
    fprintf(out, 'AC-RBF training-episode study (%d seeds, mean +/- std)\n', nS);
    fprintf(out, '%-6s | %-34s | %-34s | %s\n', 'N', ...
        'Case 1 start-up: ISE / IAE / ITAE', 'Case 2 inlet steps: ISE / IAE / ITAE', 'Ts (s), train time (s)');
    for in = 1:nN
        fprintf(out, '%-6d | %.4f / %.4f / %.4f+-%.4f | %.5f / %.4f / %.3f+-%.3f | %.2f, %.0f\n', snap_at(in), ...
            mean(E.S_ISE(:,in)), mean(E.S_IAE(:,in)), mean(E.S_ITAE(:,in)), std(E.S_ITAE(:,in)), ...
            mean(E.T_ISE(:,in)), mean(E.T_IAE(:,in)), mean(E.T_ITAE(:,in)), std(E.T_ITAE(:,in)), ...
            mean(E.S_Ts(:,in)), mean(E.time(:,in)));
    end
    fprintf(out, '\nRelative change of the mean index when doubling/extending training:\n');
    for in = 2:nN
        rc = @(F) 100*(mean(F(:,in)) - mean(F(:,in-1)))/mean(F(:,in-1));
        fprintf(out, '  %3d -> %3d : Case1 ISE %+.2f%%  IAE %+.2f%%  ITAE %+.2f%% | Case2 ISE %+.2f%%  IAE %+.2f%%  ITAE %+.2f%%\n', ...
            snap_at(in-1), snap_at(in), rc(E.S_ISE), rc(E.S_IAE), rc(E.S_ITAE), rc(E.T_ISE), rc(E.T_IAE), rc(E.T_ITAE));
    end
end
fclose(fid);
% LaTeX rows (Case 1 / Case 2, mean values)
fid = fopen(fullfile(resdir, 'exp1_episode_table.tex'), 'w');
for in = 1:nN
    fprintf(fid, '%d & %.4f & %.4f & %.4f & %.5f & %.4f & %.3f \\\\\n', snap_at(in), ...
        mean(E.S_ISE(:,in)), mean(E.S_IAE(:,in)), mean(E.S_ITAE(:,in)), ...
        mean(E.T_ISE(:,in)), mean(E.T_IAE(:,in)), mean(E.T_ITAE(:,in)));
end
fclose(fid);

%% ---------------- figures ----------------
ep = 1:N_max;
names = {'ISE', 'IAE', 'ITAE'};
fig = figure('Position', [100 100 760 720], 'Color', 'w');
for i = 1:3
    subplot(3, 1, i);
    Y = H.(names{i}); mu_ = mean(Y, 1); sd = std(Y, 0, 1);
    lo = max(mu_ - sd, 0.5*min(mu_));   % keep the band positive on the log axis
    fill([ep fliplr(ep)], [lo fliplr(mu_+sd)], [0.75 0.82 0.95], 'EdgeColor', 'none'); hold on;
    plot(ep, mu_, 'b-', 'LineWidth', 1.5);
    for n = snap_at(1:3), xline(n, 'k--', sprintf('N=%d', n), 'LabelVerticalAlignment', 'top'); end
    set(gca, 'YScale', 'log'); grid on; xlim([1 N_max]);
    ylabel(names{i});
    if i == 1, title(sprintf('AC-RBF training on Case 1 (mean \\pm std, %d seeds)', nS)); end
end
xlabel('Episode');
exportgraphics(fig, fullfile(figdir, 'exp1_training_curves.png'), 'Resolution', 300);

fig = figure('Position', [100 100 900 320], 'Color', 'w');
Ms = {'S_ISE','S_IAE','S_ITAE'};
for i = 1:3
    subplot(1, 3, i);
    errorbar(1:nN, mean(E.(Ms{i}), 1), std(E.(Ms{i}), 0, 1), 'bo-', 'LineWidth', 1.4, 'MarkerFaceColor', 'b');
    set(gca, 'XTick', 1:nN, 'XTickLabel', string(snap_at)); xlim([0.5 nN+0.5]); grid on;
    xlabel('Training episodes N'); ylabel(names{i}); title(['Case 1: ' names{i}]);
end
exportgraphics(fig, fullfile(figdir, 'exp1_index_vs_episodes.png'), 'Resolution', 300);

fprintf('\nSaved results to %s and figures to %s\n', resdir, figdir);
type(fullfile(resdir, 'exp1_episode_table.txt'));
