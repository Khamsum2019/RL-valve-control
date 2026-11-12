function sc = make_scenario(name, P, ss)
%MAKE_SCENARIO Simulation scenarios (case studies), modelled on Muduli et al. (2025).
%   name : 'startup'  - Case 1: start-up from c(x,0)=cf (the thesis scenario,
%                       also used for training)
%          'steps'    - Case 2: multiple step changes of inlet concentration cf
%                       starting from steady state (analogue of step load disturbances)
%          'nominal_p' / 'paramA' / 'paramB' - Case 3: inlet steps with nominal /
%                       perturbed plant parameters (analogue of Muduli Table IV)
%          'random'   - Case 4: random, band-limited inlet concentration cf(t)
%   ss   : struct with fields c (steady profile) and v (steady flow), from steady_state.

sc.name  = name;
sc.ystar = P.ystar;
sc.D = P.D; sc.eta = P.eta; sc.Km = P.Km;

switch name
    case 'startup'
        sc.T = 100;
        sc.cf = P.cf*ones(1, round(sc.T/P.dt)+1);
        sc.c0 = P.cf*ones(1, P.Nz);
        sc.from_ss = false;
        sc.v0 = 0.01;        % initial flow of the incremental controllers (thesis code)

    case 'steps'
        sc.T = 200;
        t = (0:round(sc.T/P.dt))*P.dt;
        cf = P.cf*ones(size(t));
        cf(t >= 20)  = 3.6;   % +20%
        cf(t >= 70)  = 3.0;
        cf(t >= 120) = 2.4;   % -20%
        cf(t >= 170) = 3.0;
        sc.cf = cf;
        sc.c0 = ss.c;
        sc.from_ss = true;
        sc.v0 = ss.v;

    case {'nominal_p', 'paramA', 'paramB'}
        % Plant parameters change at t = 0 (start from the NOMINAL steady state),
        % then two inlet steps. Slower kinetics quickly make y* unreachable with
        % v >= 0 (y > y* even at v = 0), so set A is deliberately mild.
        sc.T = 150;
        t = (0:round(sc.T/P.dt))*P.dt;
        cf = P.cf*ones(size(t));
        cf(t >= 50)  = 3.2;
        cf(t >= 100) = 2.7;
        sc.cf = cf;
        sc.c0 = ss.c;
        sc.from_ss = true;
        sc.v0 = ss.v;
        f = [1 1 1];
        if strcmp(name, 'paramA'), f = [0.97 1.05 1.03]; end  % eta, D, Km
        if strcmp(name, 'paramB'), f = [1.20 0.80 0.80]; end
        sc.eta = f(1)*P.eta; sc.D = f(2)*P.D; sc.Km = f(3)*P.Km;
        sc.factors = f;

    case 'random'
        sc.T = 300;
        t = (0:round(sc.T/P.dt))*P.dt;
        s = RandStream('mt19937ar', 'Seed', 2026);   % same disturbance for every method
        hold_s = 5;                                   % new random level every 5 s
        lev = P.cf + (rand(s, 1, ceil(sc.T/hold_s)+1) - 0.5);   % cf in [2.5, 3.5]
        raw = lev(floor(t/hold_s) + 1);
        tau = 2;                                      % first-order smoothing (s)
        a = P.dt/(tau + P.dt);
        cf = raw;
        cf(1) = P.cf;
        for k = 2:numel(t), cf(k) = cf(k-1) + a*(raw(k) - cf(k-1)); end
        sc.cf = cf;
        sc.c0 = ss.c;
        sc.from_ss = true;
        sc.v0 = ss.v;

    otherwise
        error('Unknown scenario %s', name);
end
end
