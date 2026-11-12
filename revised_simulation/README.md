# Revised simulation (three-method comparison, Muduli-style case studies)

Run from this folder in MATLAB (R2025b tested):

```matlab
exp1_episode_study   % ~4 min: AC-RBF trained 200 episodes x 5 seeds, evaluated at N = 30/50/100/200
exp2_case_studies    % ~4 min: Fixed PID vs RBF-PID vs AC-RBF on Cases 1-4
```

Outputs go to `results/` (txt tables, LaTeX rows, .mat) and `figures/` (300-dpi PNG).

| File | Role |
|---|---|
| `params_default.m` | plant, grid, actuator range `[0, 0.5]` (shared by all methods) |
| `make_scenario.m` | Case 1 start-up, Case 2 inlet steps, Case 3 parameter sets, Case 4 random inlet |
| `make_controller.m` | `fixed`, `rbfpid`, `acrbf` (corrected), `acrbf_v3` (original thesis code) |
| `run_sim.m` | PDE (Eq. 8) + controller loop; learnable state persists across calls |
| `train_adaptive.m` | episode training (network kept, reactor reset) |
| `compute_metrics.m` | ISE, IAE, ITAE, settling time (+/-5% of y*) |

## Changes to the AC-RBF controller (vs `bioreactor_actor_critic_PID_v3.m`)

1. Sampling, learning and control all at Ts = 0.1 s. In v3, De and D2e were 1 ms differences
   applied every 0.1 s, so the P and D terms were effectively zero and K_I sat on the -0.5 clip
   100% of the time (identical to a plain I-controller, K_I = -0.5).
2. dy/dv = +1 (more flow gives a higher outlet concentration), matching the paper's own physics.
3. Critic: TD(0) update with phi(t-1); actor credits the gains that were actually applied.
4. Inputs normalised: theta = [e, De/0.1, D2e/0.01].
5. Learning rates alpha_w = 1e-3, alpha_v = 0.05 (selected on the training scenario only).

RBF-PID = RBF Jacobian identification + gradient gain tuning, same algorithm as the reference
code (CSDN KeepingMatlab/136555215, from Liu Jinkun, *Advanced PID Control and MATLAB Simulation*),
purely online, initial gains (0.001, 0.024, 0.001). The reference code's literal settings
(ci = 0, bi = 10, w = 0.1, K0 = 0.01) make the Jacobian estimate ~0 on this plant, so its gains never
adapt; the identifier is therefore rescaled. `check_rbfpid_settings.m` compares both
(`results/rbfpid_settings_check.txt`).
