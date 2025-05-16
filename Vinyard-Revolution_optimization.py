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
from py_wake.site._site import UniformWeibullSite, PowerShear
import pickle
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.constraint_components.boundary import XYBoundaryConstraint

# Extracting coordinatates of wind turbines and boundarys
#RESPECT PATH DIRECTORIES
#/Users/ompatel/Documents/Proj5_Engin480/The_Foreigners_Project_5/Vinyard_utm_boundary.pkl  'Om'
#E:\Spring 2025\ENGIN 480\Project_5\The_Foreigners_Project_5\Vinyard_utm_boundary.pkl  'Kons'
with open(r'/Users/ompatel/Documents/Proj5_Engin480/The_Foreigners_Project_5/Vinyard_utm_boundary.pkl', 'rb') as f:
    boundary1 = np.array(pickle.load(f))

#/Users/ompatel/Documents/Proj5_Engin480/The_Foreigners_Project_5/Vinyard_utm_layout.pkl  'Om'
#E:\Spring 2025\ENGIN 480\Project_5\The_Foreigners_Project_5\Vinyard_utm_layout.pkl  'Kons'
with open(r'/Users/ompatel/Documents/Proj5_Engin480/The_Foreigners_Project_5/Vinyard_utm_layout.pkl', 'rb') as f:
    xinit1,yinit1 = np.array(pickle.load(f))

#/Users/ompatel/Documents/Proj5_Engin480/The_Foreigners_Project_5/Revolution_utm_boundary.pkl 'Om'
#E:\Spring 2025\ENGIN 480\Project_5\The_Foreigners_Project_5\Revolution_utm_boundary.pkl 'Kons'
with open(r'/Users/ompatel/Documents/Proj5_Engin480/The_Foreigners_Project_5/Revolution_utm_boundary.pkl', 'rb') as g:
    boundary2 = np.array(pickle.load(g))

#/Users/ompatel/Documents/Proj5_Engin480/The_Foreigners_Project_5/Revolution_utm_layout.pkl 'Om'
#E:\Spring 2025\ENGIN 480\Project_5\The_Foreigners_Project_5\Revolution_utm_layout.pkl 'Kons'
with open(r'/Users/ompatel/Documents/Proj5_Engin480/The_Foreigners_Project_5/Revolution_utm_layout.pkl', 'rb') as g:
    xinit2,yinit2 = np.array(pickle.load(g))

#number of iterations for opt
maxiter = 1000
tol = 1e-6

##########################################################################################################################
class SG_11200(GenericWindTurbine): #Revwind Turbine
    def __init__(self):
        GenericWindTurbine.__init__(self, name='SG 11-200', diameter=200, hub_height=100, 
                                    power_norm=11000, turbulence_intensity=0.07) #intensity varies from 6-8%

class RevolutionWindData(UniformWeibullSite):
    def __init__(self, ti= 0.07, shear=PowerShear(h_ref=100, alpha = 0.1)):
        f = [6.5294, 7.4553, 6.2232, 5.8886, 4.7439, 4.5632, 7.1771, 12.253, 13.8541, 10.3711, 11.5819, 9.3593]
        a = [9.93, 10.64, 9.87, 8.85, 8.46, 8.26, 10.45, 11.75, 11.40, 10.82, 11.95, 10.08]
        k = [2.385, 1.822, 1.979, 1.842, 1.607, 1.486, 1.865, 2.256, 2.678, 2.170, 2.455, 2.506]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        # self.initial_position = np.array([site.x, site.y]).T
        self.name = 'Revolution South Fork Wind'

class Haliade_X(GenericWindTurbine): #Vineyard Turbine
    def __init__(self):
        GenericWindTurbine.__init__(self, name='Haliade-X', diameter=220, hub_height=150, 
                                    power_norm=13000, turbulence_intensity=0.07) #intensity varies from 6-8%

class VinyardWind2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=PowerShear(h_ref=150, alpha=0.1), wd = 270):
        f =[6.4452, 7.6731, 6.4753, 6.0399, 4.8786, 4.5063, 7.318, 11.7828, 13.0872, 11.1976, 11.1351, 9.461]
         # f parameter list was multiplied by 0.01 from gwc file using chatGpt
        a = [10.26, 10.44, 9.52, 8.96, 9.58, 9.72, 11.48, 13.25, 12.46, 11.40, 12.35, 10.48]
        k = [2.225, 1.697, 1.721, 1.689, 1.525, 1.498, 1.686, 2.143, 2.369, 2.186, 2.385, 2.404]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        # self.initial_position = np.array([site.x, site.y]).T
        self.name = 'Vineyard Wind Farm'
##########################################################################################################################

# Calling and redifining the wind terbines and wind data
wt_revolution = SG_11200()
wt_vinyard = Haliade_X()

site_2 = RevolutionWindData()
site_1 = VinyardWind2()

# calling the type fo model that we will be using for the 2 sites
wf1_model = Bastankhah_PorteAgel_2014(
    site_1,
    wt_vinyard,
    k=0.0324555,  # default value from BastankhahGaussianDeficit
)

# defining the wind terbine locations of Virginia
wt_x1, wt_y1 = xinit1, yinit1

# defining the wind terbine locations of Costal
wt_x2, wt_y2 = xinit2, yinit2

# Collecting the wind terbine locations into one list using numpy
X_full = np.concatenate([wt_x1, wt_x2])
Y_full = np.concatenate([wt_y1, wt_y2])

# Amount of wind terbines and print the number 
n_wt = len(X_full)
print(f"Initial layout has {n_wt} wind turbines")

# plot initial layout
# plt.figure()
# plt.plot(X_full, Y_full, "x", c="magenta")
# # put indeces on the wind turbines
# for i in range(n_wt):
#     plt.text(X_full[i] + 10, Y_full[i], str(i + 1), fontsize=12)
# plt.axis("equal")
# plt.show()

print('done')

# masking Vinyard wind
n_wt_sf = n_wt - 63                 
# n_wt is the amount of turbines included in this figure there are 76 total turbines 
# 0-63 are vinyards we need to isolate them to mask them so 67-12 = 63

# Masking vinyard and costal
wf1_mask = np.zeros(n_wt, dtype=bool)
wf1_mask[:n_wt_sf] = True
wf2_mask = ~(wf1_mask)  # the rest of turbines

# printing which mask belongs to which wind farm 
print(f"Turbines belonging to wind farm 1: {np.where(wf1_mask)[0]}") # verifiing that our calulations were correct 
print(f"Turbines belonging to wind farm 2: {np.where(wf2_mask)[0]}")

# putting the different wind farms in to two disktict list's 
wt_groups = [np.arange(n_wt-63), np.arange(n_wt-63, n_wt)]


# defining the constraints 
constr_type = BoundaryType.POLYGON
constraint_comp = MultiWFBoundaryConstraint(geometry = [boundary1, boundary2], wt_groups=wt_groups,
        boundtype=constr_type)

# # let's see how the boundaries look like
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# plt.plot(X_full, Y_full, "x", c="magenta")
# for i in range(n_wt):
#     plt.text(X_full[i] + 10, Y_full[i], str(i + 1), fontsize=12)
# plt.axis("equal")
# constraint_comp.get_comp(n_wt).plot(ax1)
# plt.show()

print('done')

# Difining wind direction and speed 
full_wd = np.arange(0, 360, 1)  # wind directions
full_ws = np.arange(3, 25, 1)  # wind speeds
freqs = site_1.local_wind(  # sector frequencies
    X_full,
    Y_full,
    wd=full_wd,
    ws=full_ws,
    h = 150
).Sector_frequency_ilk[0, :, 0]
# weibull parameters
As = site_1.local_wind(X_full, Y_full, wd=full_wd, ws=full_ws, h=150).Weibull_A_ilk[0, :, 0]
ks = site_1.local_wind(X_full, Y_full, wd=full_wd, ws=full_ws, h=150).Weibull_k_ilk[0, :, 0]
N_SAMPLES = 25  # play with the number of samples

# def wind_resource_sample(fixed_wd=270):
#     # `As` and `ks` are already for wd=270, so we sample directly
#     ws = As * np.random.weibull(ks, size=N_SAMPLES)
    
#     # Create a wind direction array filled with the fixed direction
#     wd = np.full(N_SAMPLES, fixed_wd)

#     return wd, ws


# Randomizing and runnning different wind directions and speeds 
def wind_resource_sample():
    idx = np.random.choice(np.arange(full_wd.size), N_SAMPLES, p=freqs / freqs.sum())
    wd = full_wd[idx]
    ws = As[idx] * np.random.weibull(ks[idx])
    return wd, ws



# Partial equations calculations 
def daep_func(x,y, full=False, **kwargs):
    daep = wf1_model.aep_gradients(gradient_method=autograd, wrt_arg=['x','y'], x=x,
                                y=y)
    return daep
print('done')

# aep function - SGD
def aep_func(x, y, full=False, **kwargs):
    wd, ws = wind_resource_sample()
    aep_sgd = wf1_model(x, y, wd=wd, ws=ws, time=not full).aep().sum().values * 1e6
    return aep_sgd

# gradient function - SGD
def aep_jac(x, y, **kwargs):
    wd, ws = wind_resource_sample()
    jx, jy = wf1_model.aep_gradients(
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
    windFarmModel=wf1_model, n_wt=n_wt, grad_method=autograd
)


# defining the driver
driver_type = "SGD"  # "SLSQP" or "SGD"
min_spacing = wt_vinyard.diameter() * 2

# If SLSQP define use these constraints and spacing  
if driver_type == "SLSQP":
    constraints = [
        constraint_comp,
        SpacingConstraint(min_spacing=min_spacing),
    ]
    driver = EasyScipyOptimizeDriver(
        optimizer="SLSQP",
        maxiter=10000,
    )
    cost_comp = slsqp_cost_comp

# If SGD define use these constraints and spacing  
elif driver_type == "SGD":
    constraints = DistanceConstraintAggregation(
        constraint_comp,
        n_wt=n_wt,
        min_spacing_m=min_spacing,
        windTurbines=wt_vinyard,
    )
    driver = EasySGDDriver(
        maxiter=10000,
        speedupSGD=True,
        learning_rate=wt_vinyard.diameter() / 5,
        gamma_min_factor=0.1,
    )
    cost_comp = sgd_cost_comp

# if unrecogizeed driver print Unkown driver
else:
    raise ValueError(f"Unknown driver: {driver_type}")


# wd = 270  we only care about this direction bec this direction is the only 
# way that costal will impact vinyard
# similarly when we do the vise versa of this challenge what effect dose 
# vinyard have on costal the wd that will intrest us will be 90 degrees
                                            
# defining problem for optimizer calling both farms to be optimized at the same time
problem = TopFarmProblem(
    design_vars={"x": X_full, "y": Y_full},
    n_wt=n_wt,
    constraints=constraints,
    cost_comp=cost_comp,
    driver=driver,
    plot_comp=XYPlotComp(),
)

# Initialting recorder, AEP cost, and optimization problem
cost, state, recorder = problem.optimize(disp = True)


# Where to save the recording what the file should be names as 
recorder.save('optimization_Vinyard_respecting_Revolution_2')

print('done')