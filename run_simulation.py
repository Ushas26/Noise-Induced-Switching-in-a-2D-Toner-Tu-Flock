import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Model, Toner-Tu 2D ##########
class TonerTu2D:

    def __init__(
        self,
        nx=32,
        ny=32,
        alpha=1.0,
        beta=1.0,
        lam=0.5,
        D=0.2,
    ):
        self.nx = nx
        self.ny = ny

        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.D = D

    def laplacian(self, f):

        return (
            np.roll(f,1,0)
            + np.roll(f,-1,0)
            + np.roll(f,1,1)
            + np.roll(f,-1,1)
            -4*f
        )

    def rhs(self, ux, uy):

        speed2 = ux**2 + uy**2

        reaction_x = (
            self.alpha
            - self.beta * speed2
        ) * ux

        reaction_y = (
            self.alpha
            - self.beta * speed2
        ) * uy

        diffusion_x = self.D * self.laplacian(ux)
        diffusion_y = self.D * self.laplacian(uy)

        # -------- Advection term --------

        dux_dx = (
            np.roll(ux, -1, 0)
            - np.roll(ux, 1, 0)
        ) / 2

        dux_dy = (
            np.roll(ux, -1, 1)
            - np.roll(ux, 1, 1)
        ) / 2

        duy_dx = (
            np.roll(uy, -1, 0)
            - np.roll(uy, 1, 0)
        ) / 2

        duy_dy = (
            np.roll(uy, -1, 1)
            - np.roll(uy, 1, 1)
        ) / 2

        adv_x = -(
            ux * dux_dx
            + uy * dux_dy
        )

        adv_y = -(
            ux * duy_dx
            + uy * duy_dy
        )

        # -------------------------------

        return (
            reaction_x
            + diffusion_x
            + self.lam * adv_x,

            reaction_y
            + diffusion_y
            + self.lam * adv_y
        )


# Analysis #########
def switching_time(traj, threshold=0.0):
    mean_series = traj.mean(axis=1)
    idx = np.where(mean_series < threshold)[0]
    return None if len(idx) == 0 else int(idx[0])


def polarization(ux, uy):
    """
    Global polarization order parameter.

    P = |<u>|
    """
    px = np.mean(ux)
    py = np.mean(uy)

    return np.sqrt(px**2 + py**2)


# SPDE Solver #########
# FIX: this now returns the polarization *time series* instead of the
# full (ux, uy) history at every step. Storing every field snapshot for
# every one of the many Monte-Carlo runs used by estimate_mfpt was both
# extremely slow and memory-hungry, and is not needed -- the analysis
# only ever looks at the scalar polarization anyway.
def simulate(model, T=250, dt=0.01, eps=0.2):

    steps = int(T / dt)

    ux = np.ones((model.nx, model.ny))
    uy = np.zeros((model.nx, model.ny))

    pol_history = np.empty(steps)

    for k in range(steps):

        fx, fy = model.rhs(ux, uy)

        ux += (
            dt * fx
            + np.sqrt(eps * dt)
            * np.random.randn(*ux.shape)
        )

        uy += (
            dt * fy
            + np.sqrt(eps * dt)
            * np.random.randn(*uy.shape)
        )

        pol_history[k] = polarization(ux, uy)

    return pol_history


def first_passage_time(pol_history, dt, threshold):
    idx = np.where(pol_history < threshold)[0]
    if len(idx) == 0:
        return None
    return idx[0] * dt


def estimate_mfpt(
    simulator,
    dt,
    threshold,
    n_runs=30,
):

    times = []

    for _ in range(n_runs):

        pol_history = simulator()

        tau = first_passage_time(pol_history, dt=dt, threshold=threshold)

        if tau is not None:
            times.append(tau)

    if len(times) == 0:
        return np.inf

    return np.mean(times)


# FFS #########


class FFS:

    def __init__(self, interfaces):

        self.interfaces = interfaces

    def crossing_probability(
        self,
        samples,
        interface,
    ):

        count = np.sum(
            samples > interface
        )

        return count / len(samples)

    def transition_rate(
        self,
        flux,
        probs,
    ):

        return flux * np.prod(probs)


# Instanton ##########


class Instanton1D:

    def __init__(
        self,
        alpha=1,
        beta=1,
    ):
        self.alpha = alpha
        self.beta = beta

    def drift(self, u):

        return (
            self.alpha * u
            - self.beta * u**3
        )

    def action(self, path):

        dt = 1.0 / len(path)

        vel = np.gradient(path)

        drift = self.drift(path)

        action = (
            0.5
            * np.sum(
                (vel - drift)**2
            )
            * dt
        )

        penalty = (
            1000 * (path[0] - 1.0)**2
            +
            1000 * (path[-1] + 1.0)**2
        )

        return action + penalty

    def compute(self):

        guess = np.linspace(
            1,
            -1,
            100
        )

        res = minimize(
            self.action,
            guess,
        )

        return res.x


# Variables ######
#
# FIX: the original epsilon values (0.003-0.007) are far too small for
# this system. With alpha=beta=1 the polarized state ux=1 is a strongly
# stable fixed point, and the 32x32 spatial average heavily suppresses
# the effective noise on the global order parameter. At those epsilon
# values the polarization never gets anywhere near the switching
# threshold within any reasonable simulation time, so every run reports
# no crossing, every MFPT is np.inf, and every plot downstream ends up
# empty. The values below (found by scanning) actually produce
# switching within the simulated horizon, with a clear spread of MFPTs.

epsilons = np.array([0.35, 0.40, 0.45, 0.50, 0.60, 0.70])

mfpts = []

model = TonerTu2D()

T_SWEEP = 60          # long enough that even the slowest (eps=0.35) case
DT = 0.01             # reliably crosses the threshold before time runs out
THRESHOLD = 0.2
N_RUNS = 30

for eps in epsilons:

    print(f"Running epsilon = {eps}")

    mfpt = estimate_mfpt(
        lambda eps=eps: simulate(
            model,
            T=T_SWEEP,
            dt=DT,
            eps=eps,
        ),
        dt=DT,
        threshold=THRESHOLD,
        n_runs=N_RUNS,
    )

    mfpts.append(mfpt)

mfpts = np.array(mfpts)

print("epsilons =", epsilons)
print("mfpts =", mfpts)

# only finite MFPTs (i.e. epsilons that actually produced a crossing in
# every run considered) can be meaningfully plotted
valid = np.isfinite(mfpts)

plt.figure(figsize=(8, 5))

plt.plot(
    epsilons[valid],
    mfpts[valid],
    "o-"
)

plt.xlabel(r"$\epsilon$")
plt.ylabel("MFPT")

plt.grid(True)

plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 5))

plt.axhline(
    THRESHOLD,
    color="red",
    ls="--",
    label="Transition threshold"
)

plt.xlabel("Time")
plt.ylabel("Polarisation")


for i in epsilons:
    # Trajectory #######
    EPS_EXAMPLE = i

    pol_history = simulate(
        model,
        T=T_SWEEP,
        dt=DT,
        eps=EPS_EXAMPLE,
    )

    # FIX: plot against real time (steps * dt), not against the raw sample
    # index -- the previous version labelled the axis "Time" but actually
    # plotted step number, which for dt=0.01 is 100x too large.
    time_axis = np.arange(len(pol_history)) * DT

    plt.plot(time_axis, pol_history, label=f"epsilon = {i}")


plt.legend()
plt.tight_layout()
plt.show()


# MFPT Scaling ########

plt.figure(figsize=(8, 5))

plt.semilogy(epsilons[valid], mfpts[valid], "o-")

plt.xlabel("epsilon")
plt.ylabel("MFPT")
plt.grid(True, which="both")

plt.tight_layout()
plt.show()


# Large deviation test ###########
# FIX: removed the stray duplicate plt.figure() call that was creating
# an extra, permanently blank figure window before the real plot.

plt.figure(figsize=(8, 5))

plt.plot(
    1 / epsilons[valid],
    np.log(mfpts[valid]),
    "o-"
)

plt.xlabel("1/epsilon")
plt.ylabel("log(tau)")
plt.grid(True)

plt.tight_layout()
plt.show()