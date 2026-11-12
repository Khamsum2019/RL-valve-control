function m = compute_metrics(out, ystar, band)
%COMPUTE_METRICS ISE, IAE, ITAE (Muduli et al. Eqs. 35-37) and settling time.
%   ISE  = int e^2 dt,  IAE = int |e| dt,  ITAE = int t|e| dt  (rectangle rule)
%   Ts   = last time |e| leaves the +/- band*y* tube (Inf if never settles)
%   emax_late = max |e| after t = 10 s (peak deviation once start-up is over)

e  = out.y - ystar;
dt = out.t(2) - out.t(1);
m.ISE  = sum(e.^2)*dt;
m.IAE  = sum(abs(e))*dt;
m.ITAE = sum(out.t.*abs(e))*dt;
outside = find(abs(e) > band*ystar, 1, 'last');
if isempty(outside), m.Ts = 0;
elseif outside == numel(e), m.Ts = Inf;
else, m.Ts = out.t(outside + 1);
end
m.emax_late = max(abs(e(out.t >= 10)));
end
