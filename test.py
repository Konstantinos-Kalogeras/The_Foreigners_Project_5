import numpy as np
import topfarm
import matplotlib.pyplot as plt
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import (
    MultiWFBoundaryConstraint,
    BoundaryType,
)
from topfarm.constraint_components.constraint_aggregation import (
    DistanceConstraintAggregation,
)
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import (
    PyWakeAEPCostModelComponent,
)
from topfarm.easy_drivers import EasyScipyOptimizeDriver, EasySGDDriver
from topfarm.plotting import XYPlotComp
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.utils.gradients import autograd
from py_wake.validation.lillgrund import LillgrundSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent


#########################################    sample code thuis is the exampke 
wind_turbines = GenericWindTurbine("GenWT", 100.6, 2000, 150)
site = LillgrundSite()
wf_model = Bastankhah_PorteAgel_2014(
    site,
    wind_turbines,
    k=0.0324555,  # default value from BastankhahGaussianDeficit
)

# generate initial positions
grid_size = 2
wt_x1, wt_y1 = np.meshgrid(
    np.linspace(0, wind_turbines.diameter() * grid_size, grid_size),
    np.linspace(0, wind_turbines.diameter() * grid_size, grid_size),
)
wt_x1, wt_y1 = wt_x1.flatten(), wt_y1.flatten()
wt_x2 = wt_x1 + wind_turbines.diameter() * grid_size * 3.0
wt_y2 = wt_y1
wt_y3 = wt_y1 + wind_turbines.diameter() * grid_size * 3.0
wt_x3 = wt_x1
X_full = np.concatenate([wt_x1, wt_x2, wt_x3])
Y_full = np.concatenate([wt_y1, wt_y2, wt_y3])
n_wt = len(X_full)
print(f"Initial layout has {n_wt} wind turbines")

# plot initial layout
plt.figure()
plt.plot(X_full, Y_full, "x", c="magenta")
# put indeces on the wind turbines
for i in range(n_wt):
    plt.text(X_full[i] + 10, Y_full[i], str(i + 1), fontsize=12)
plt.axis("equal")
plt.show()

n_wt_sf = n_wt // 3
wf1_mask = np.zeros(n_wt, dtype=bool)
wf1_mask[:n_wt_sf] = True
wf2_mask = np.zeros(n_wt, dtype=bool)
wf2_mask[n_wt_sf : n_wt_sf * 2] = True
wf3_mask = ~(wf1_mask | wf2_mask)  # the rest of turbines

print(f"Turbines belonging to wind farm 1: {np.where(wf1_mask)[0]}")
print(f"Turbines belonging to wind farm 2: {np.where(wf2_mask)[0]}")
print(f"Turbines belonging to wind farm 3: {np.where(wf3_mask)[0]}")

# utility functions to construct the boundary constraint
def _get_radius(x, y):  # fmt: skip
    return np.sqrt((x - x.mean()) ** 2 + (y - y.mean()) ** 2).max() + 100
def _get_center(x, y):  # fmt: skip
    return np.array([x.mean(), y.mean()])
def _get_corners(x: np.ndarray, y: np.ndarray, radius, stype='rect'):  # fmt: skip
    cx = x.mean()
    cy = y.mean()
    if stype == "rect":
        return np.array(
            [
                [cx + radius, cy + radius],
                [cx - radius, cy - radius],
                [cx + radius, cy - radius],
                [cx - radius, cy + radius],
            ]
        )
    if stype == "rot":
        return np.array(
            [
                [cx, cy + radius],
                [cx + radius, cy],
                [cx, cy - radius],
                [cx - radius, cy],
            ]
        )
    if stype == "hex":
        return np.array(
            [
                [cx + radius, cy],
                [cx + radius / 2, cy + radius * np.sqrt(3) / 2],
                [cx - radius / 2, cy + radius * np.sqrt(3) / 2],
                [cx - radius, cy],
                [cx - radius / 2, cy - radius * np.sqrt(3) / 2],
                [cx + radius / 2, cy - radius * np.sqrt(3) / 2],
            ]
        )
    raise ValueError(f"Unknown shape type: {stype}")

constr_type = BoundaryType.CONVEX_HULL  # or BoundaryType.CONVEX_HULL
wt_groups = [
    np.arange(n_wt // 3),
    np.arange(n_wt // 3, n_wt // 3 * 2),
    np.arange(n_wt // 3 * 2, n_wt),
]

if constr_type == BoundaryType.CIRCLE:
    constraint_comp = MultiWFBoundaryConstraint(
        geometry=[
            (_get_center(wt_x1, wt_y1), _get_radius(wt_x1, wt_y1)),
            (_get_center(wt_x2, wt_y2), _get_radius(wt_x2, wt_y2)),
            (_get_center(wt_x3, wt_y3), _get_radius(wt_x3, wt_y3)),
        ],
        wt_groups=wt_groups,
        boundtype=constr_type,
    )
elif constr_type == BoundaryType.CONVEX_HULL:
    radius = (
        np.sqrt((wt_x1 - wt_x1.mean()) ** 2 + (wt_y1 - wt_y1.mean()) ** 2).max() + 150
    )
    constraint_comp = MultiWFBoundaryConstraint(
        geometry=[
            _get_corners(wt_x1, wt_y1, radius, stype="rot"),
            _get_corners(wt_x2, wt_y2, radius, stype="hex"),
            _get_corners(wt_x3, wt_y3, radius, stype="rect"),
        ],
        wt_groups=wt_groups,
        boundtype=constr_type,
    )
else:
    raise ValueError(f"Unknown constraint type: {constr_type}")

# let's see how the boundaries look like
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(X_full, Y_full, "x", c="magenta")
for i in range(n_wt):
    plt.text(X_full[i] + 10, Y_full[i], str(i + 1), fontsize=12)
plt.axis("equal")
constraint_comp.get_comp(n_wt).plot(ax1)
plt.show()


np.random.seed(42)
# Wind Resouces
full_wd = np.arange(0, 360, 1)  # wind directions
full_ws = np.arange(3, 25, 1)  # wind speeds
freqs = site.local_wind(  # sector frequencies
    X_full,
    Y_full,
    wd=full_wd,
    ws=full_ws,
).Sector_frequency_ilk[0, :, 0]
# weibull parameters
As = site.local_wind(X_full, Y_full, wd=full_wd, ws=full_ws).Weibull_A_ilk[0, :, 0]
ks = site.local_wind(X_full, Y_full, wd=full_wd, ws=full_ws).Weibull_k_ilk[0, :, 0]
N_SAMPLES = 25  # play with the number of samples


# sample wind resources
def wind_resource_sample():
    idx = np.random.choice(np.arange(full_wd.size), N_SAMPLES, p=freqs / freqs.sum())
    wd = full_wd[idx]
    ws = As[idx] * np.random.weibull(ks[idx])
    return wd, ws


# aep function - SGD
def aep_func(x, y, full=False, **kwargs):
    wd, ws = wind_resource_sample()
    aep_sgd = wf_model(x, y, wd=wd, ws=ws, time=not full).aep().sum().values * 1e6
    return aep_sgd


# gradient function - SGD
def aep_jac(x, y, **kwargs):
    wd, ws = wind_resource_sample()
    jx, jy = wf_model.aep_gradients(
        gradient_method=autograd, wrt_arg=["x", "y"], x=x, y=y, ws=ws, wd=wd, time=True
    )
    daep_sgd = np.array([np.atleast_2d(jx), np.atleast_2d(jy)]) * 1e6
    return daep_sgd


# AEP Cost Model Component - SGD
sgd_cost_comp = AEPCostModelComponent(
    input_keys=[topfarm.x_key, topfarm.y_key],
    n_wt=n_wt,
    cost_function=aep_func,
    cost_gradient_function=aep_jac,
)

# AEP Cost Model Component - SLSQP
slsqp_cost_comp = PyWakeAEPCostModelComponent(
    windFarmModel=wf_model, n_wt=n_wt, grad_method=autograd
)

driver_type = "SGD"  # "SLSQP" or "SGD"
min_spacing = wind_turbines.diameter() * 2

if driver_type == "SLSQP":
    constraints = [
        constraint_comp,
        SpacingConstraint(min_spacing=min_spacing),
    ]
    driver = EasyScipyOptimizeDriver(
        optimizer="SLSQP",
        # might not be enough for the optimizer to converge
        maxiter=1000,
    )
    cost_comp = slsqp_cost_comp
elif driver_type == "SGD":
    constraints = DistanceConstraintAggregation(
        constraint_comp,
        n_wt=n_wt,
        min_spacing_m=min_spacing,
        windTurbines=wind_turbines,
    )
    driver = EasySGDDriver(
        # might not be enough for the optimizer to converge
        maxiter=1000,
        speedupSGD=True,
        learning_rate=wind_turbines.diameter() / 5,
        gamma_min_factor=0.1,
    )
    cost_comp = sgd_cost_comp
else:
    raise ValueError(f"Unknown driver: {driver_type}")

problem = TopFarmProblem(
    design_vars={"x": X_full, "y": Y_full},
    n_wt=n_wt,
    constraints=constraints,
    cost_comp=cost_comp,
    driver=driver,
    plot_comp=XYPlotComp(),
)

cost, state, recorder = problem.optimize(disp=True)