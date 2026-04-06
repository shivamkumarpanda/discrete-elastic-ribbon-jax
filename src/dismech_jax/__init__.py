import jax
jax.config.update("jax_enable_x64", True)

from .stencils import Triplet
from .states import TripletState
from .params import Geometry, Material, SimParams
from .models import DER, Kirchhoff, Sano, Sadowsky, Audoly, Wunderlich
from .systems import Rod, BC
from .solver import solve
from .time_stepper import TimeStepper, SimulationResult
from .geometry import (
    load_geometry_txt,
    ensure_initial_geometry,
    create_rod_from_nodes,
    fix_nodes,
    move_nodes,
)
