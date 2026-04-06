from dataclasses import dataclass


@dataclass
class Geometry:
    length: float
    r0: float  # rod radius or thickness h
    axs: float | None = None  # cross-sectional area A = w * h
    jxs: float | None = None  # torsional constant J
    ixs1: float | None = None  # 2nd moment of area I1 (primary bending)
    ixs2: float | None = None  # 2nd moment of area I2 (secondary bending)


@dataclass
class Material:
    density: float
    youngs_rod: float = 0.0
    poisson_rod: float = 0.0


@dataclass
class SimParams:
    dt: float = 0.01
    total_time: float = 10.0
    tol: float = 1e-6
    ftol: float = 1e-4
    dtol: float = 1e-4
    max_iter: int = 100
    log_step: int = 1
    static_sim: bool = False
