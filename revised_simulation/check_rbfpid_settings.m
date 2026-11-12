%% CHECK_RBFPID_SETTINGS  RBF-PID with the reference (CSDN) settings vs the rescaled defaults
% Supplementary check: with the literal reference settings the identifier's
% Jacobian estimate is ~0, so the PID gains stay at their initial 0.01.
clear; clc;
here = fileparts(mfilename('fullpath')); addpath(here);
resdir = fullfile(here, 'results'); if ~exist(resdir, 'dir'), mkdir(resdir); end
P = params_default(); [ss.c, ss.v] = steady_state(P);

ref = struct('xite', 0.5, 'alfa', 0.05, 'eta_k', [0.15; 0.15; 0.15], 'du_scale', 1, ...
             'w0', 0.1*ones(6,1), 'ci0', zeros(3,6), 'bi0', 10*ones(6,1), 'K0', [0.01; 0.01; 0.01]);
cfg = {ref, struct()}; nm = {'reference settings', 'rescaled (default)'};
cases = {'startup', 'steps', 'nominal_p', 'paramA', 'paramB', 'random'};

fid = fopen(fullfile(resdir, 'rbfpid_settings_check.txt'), 'w');
for out = [1 fid]
    fprintf(out, '%-10s | %-48s | %s\n', 'Case', ['RBF-PID, ' nm{1} ': ISE / IAE / ITAE, final K'], ...
        ['RBF-PID, ' nm{2} ': ISE / IAE / ITAE, final K']);
end
for ic = 1:numel(cases)
    sc = make_scenario(cases{ic}, P, ss);
    s = cell(1, 2);
    for i = 1:2
        o = run_sim(make_controller('rbfpid', 1, cfg{i}), sc, P);
        m = compute_metrics(o, P.ystar, P.settle_band);
        s{i} = sprintf('%.5f / %.4f / %8.3f, [%.3f %.3f %.3f]', m.ISE, m.IAE, m.ITAE, o.K(:,end));
    end
    for out = [1 fid], fprintf(out, '%-10s | %-48s | %s\n', cases{ic}, s{1}, s{2}); end
end
fclose(fid);
