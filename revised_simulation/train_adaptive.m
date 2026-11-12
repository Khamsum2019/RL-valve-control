function [hist, snaps] = train_adaptive(ctrl, sc, P, N, snap_at)
%TRAIN_ADAPTIVE Train an adaptive controller for N episodes on scenario sc.
%   Learnable parameters persist across episodes; only the reactor is reset.
%   hist  : per-episode ISE, IAE, ITAE, Ts, reward
%   snaps : cell array of controller copies taken after episodes listed in snap_at

if nargin < 5, snap_at = []; end
hist.ISE = zeros(1, N); hist.IAE = hist.ISE; hist.ITAE = hist.ISE;
hist.Ts = hist.ISE; hist.reward = hist.ISE;
snaps = cell(1, numel(snap_at));
for ep = 1:N
    [out, ctrl] = run_sim(ctrl, sc, P);
    m = compute_metrics(out, P.ystar, P.settle_band);
    hist.ISE(ep) = m.ISE; hist.IAE(ep) = m.IAE; hist.ITAE(ep) = m.ITAE;
    hist.Ts(ep) = m.Ts; hist.reward(ep) = out.reward;
    idx = find(snap_at == ep);
    if ~isempty(idx), snaps{idx} = ctrl; end
end
end
