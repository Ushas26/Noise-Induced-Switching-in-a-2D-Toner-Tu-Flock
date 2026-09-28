# Noise-Induced-Switching-in-a-2D-Toner-Tu-Flock

A Python simulation of noise-induced polarisation loss in a stochastic 2D Toner-Tu model of collective motion. The script estimates the mean first-passage time (MFPT) for a fully polarised flock to lose its global order, and examines how that time scales with noise strength, as a first test of large-deviation (Arrhenius-type) behaviour.

Overview

Flocking systems such as birds, fish and active matter can sit in a stable, ordered (polarised) state. Noise can eventually push them out of it. This project asks how long that takes as a function of noise intensity.

The code:

Integrates a stochastic Toner-Tu field equation on a periodic 2D grid.
Tracks the global polarisation P = |<u>| over time.
Records the first time P drops below a threshold.
Averages over many independent runs to get the MFPT for each noise level.
Plots MFPT against noise, on a log scale, and as log(tau) vs 1/epsilon, the standard large-deviation diagnostic.
Model

The velocity field u = (ux, uy) on an nx x ny periodic lattice evolves as

du/dt = (alpha - beta |u|^2) u  +  D * Laplacian(u)  -  lambda (u . grad) u  +  sqrt(eps) * noise
Term	Meaning
(alpha - beta|u|^2) u	Local ordering / saturation (polarised state at |u| = sqrt(alpha/beta))
D * Laplacian(u)	Spatial diffusion (5-point stencil)
-lambda (u . grad) u	Toner-Tu advective nonlinearity (central differences)
sqrt(eps) * noise	Additive space-time white noise

Numerics: explicit Euler-Maruyama, lattice spacing 1, periodic boundaries, initial condition ux = 1, uy = 0 (fully polarised).

Default parameters: nx = ny = 32, alpha = beta = 1, lambda = 0.5, D = 0.2, dt = 0.01.

Project structure

Everything lives in a single script, run_simulation.py.

Component	Purpose
TonerTu2D	Model class: Laplacian, advection, reaction and the full right-hand side
simulate	SPDE integrator; returns the polarisation time series only (no field snapshots, for speed and memory)
polarization	Global order parameter |<u>|
first_passage_time, estimate_mfpt	Threshold-crossing time and Monte Carlo MFPT estimate
FFS	Skeleton for Forward Flux Sampling (crossing probabilities, rate). Not yet used in the main run
Instanton1D	1D optimal-path (instanton) toy for the reduced double-well drift alpha*u - beta*u^3. Not yet used in the main run
Usage

Requirements: Python 3.8+, numpy, scipy, matplotlib

bash
pip install numpy scipy matplotlib
python run_simulation.py

The script sweeps epsilon in {0.35, 0.40, 0.45, 0.50, 0.60, 0.70} with 30 runs per value (T = 60, threshold = 0.2), prints the estimated MFPTs, and opens four figures:

MFPT vs epsilon
Example polarisation trajectories for each epsilon, with the threshold marked
MFPT vs epsilon (semi-log)
log(tau) vs 1/epsilon (large-deviation test)

To save figures instead of only displaying them, uncomment the plt.savefig(...) lines and set a valid output path.

Runtime note: this is a pure NumPy loop (6 noise levels x 30 runs x up to 6,000 steps on a 32x32 grid), so expect it to take a while. Reduce N_RUNS or grid size for quick tests.

Notes and limitations
Noise range. With alpha = beta = 1 the polarised state is strongly stable, and spatial averaging over the lattice suppresses noise on the global order parameter. Much smaller epsilon values (around 0.003-0.007) produced no switching within any practical simulation time, so the sweep uses larger values where crossings are observed. These are moderate-noise, not truly rare-event, conditions.
Large-deviation test is exploratory. An approximately linear log(tau) vs 1/epsilon curve would be consistent with Arrhenius/Freidlin-Wentzell scaling, but with this range and sample size it is indicative rather than conclusive.
Censoring. MFPT is averaged only over runs that cross within T. If some runs do not cross, the estimate is biased low. If none cross, np.inf is returned and that point is dropped from the plots.
Threshold choice. The MFPT depends on the polarisation threshold (0.2 here).
Instanton and FFS modules are scaffolding for planned rare-event analysis at lower noise.
Possible extensions
Wire up FFS to reach the low-noise regime directly.
Compare the 1D instanton action against the measured log(tau) slope.
Vary system size, lambda and D to study finite-size and advection effects.
Replace explicit Euler with a semi-implicit or higher-order scheme and add seeding for reproducibility.
Vectorise across runs or port to JAX/Numba for speed.
Keywords

Toner-Tu, active matter, flocking, stochastic PDE, Euler-Maruyama, mean first-passage time, rare events, large deviations, forward flux sampling, instanton
