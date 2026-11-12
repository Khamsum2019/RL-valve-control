function P = params_default()
%PARAMS_DEFAULT Nominal bioreactor, discretisation and actuator parameters.
%   Same values as Sections 3.2-3.5 of the thesis.

P.D     = 0.1;      % axial diffusion coefficient (m^2/s)
P.eta   = 3;        % maximum specific reaction rate (1/s)
P.Km    = 1;        % Monod half-saturation constant (g/L)
P.cf    = 3;        % nominal inlet concentration (g/L)
P.ystar = 0.1;      % target outlet concentration (g/L)

P.dt = 0.001;       % time step (s)
P.dx = 0.05;        % spatial step (m)
P.Nz = round(1/P.dx) + 1;

% One actuator range for ALL controllers so the comparison is fair.
% (Sec. 3.1 of the thesis: v in [0, 0.5] m/s.)
P.vmin = 0;
P.vmax = 0.5;

P.ctrl_interval = 100;  % adaptive controllers update v every 100 steps (0.1 s)
P.settle_band   = 0.05; % settling band = +/-5% of y*
end
