function [c_ss, v_ss] = steady_state(P)
%STEADY_STATE Steady flow rate v* and profile c*(x) giving y = y* (nominal plant).
%   Bisection on v; for each v the discretised PDE is marched to steady state.

lo = 0; hi = P.vmax;
for iter = 1:40
    v = 0.5*(lo + hi);
    c = march(P, v);
    if c(end) > P.ystar, hi = v; else, lo = v; end
end
v_ss = 0.5*(lo + hi);
c_ss = march(P, v_ss);
end

function c = march(P, v)
Nz = P.Nz; dt = P.dt; dx = P.dx;
c = P.cf*ones(1, Nz);
for k = 1:200000
    ci = c(2:Nz-1);
    cn = ci + dt*(P.D*(c(3:Nz)-2*ci+c(1:Nz-2))/dx^2 ...
                  - v*(ci-c(1:Nz-2))/dx - P.eta*ci./(P.Km+ci));
    c_old = c;
    c(2:Nz-1) = cn;
    c(Nz) = c(Nz-1);
    if mod(k, 1000) == 0 && max(abs(c - c_old)) < 1e-12, break; end
end
end
