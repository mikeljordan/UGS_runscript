"""
Python script runs and visualizes 1D high-enthalpy geothermal compositional
flow simulations using PorePy.

Simulation cases include:
  - Single-phase (high, moderate, low pressure)
  - Two-phase (high, low pressure)

This script:
  - Creates geothermal models with appropriate BC and IC
  - Loads precomputed thermodynamic data on a discrete parametric phz- and pTz-spaces from VTK files
  - Runs time-dependent simulations using a unified compositional flow  model in Porepy
  - Generates and saves simulation results compared with CSMP++ reference data (Weis et al., DOI: 10.1111/gfl.12080).

"""

from __future__ import annotations
import time
import numpy as np
import porepy as pp
import os

# Import model configurations
from porepy.examples.geothermal_flow.model_configuration.flow_model_configuration import (
    SinglePhaseFlowModelConfiguration as SinglePhaseFlowModel,
    TwoPhaseFlowModelConfiguration as TwoPhaseFlowModel,
)
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler
import porepy.examples.geothermal_flow.data_extractor_util as data_util

# Import geometric setup for the model domain
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import SimpleGeometryHorizontal as ModelGeometry

# Boundary & Initial Conditions
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (
    BCSinglePhaseHighPressure, 
    BCSinglePhaseModeratePressure, 
    BCSinglePhaseLowPressure,
    BCTwoPhaseHighPressure, 
    BCTwoPhaseLowPressure
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (
    ICSinglePhaseHighPressure, 
    ICSinglePhaseModeratePressure, 
    ICSinglePhaseLowPressure,
    ICTwoPhaseHighPressure, 
    ICTwoPhaseLowPressure
)

# Simulation configurations
SIMULATION_CASES = {
    "single_phase_HP": {    # High-pressure single-phase (Figure 2, Case 1)
        "BC": BCSinglePhaseHighPressure,
        "IC": ICSinglePhaseHighPressure,
        "FlowModel": SinglePhaseFlowModel,
        "tf": 250 * 365 * 86400,
        "dt": 365 * 86400,
    },
    "single_phase_MP": {  # Moderate-pressure single-phase (Figure 2, Case 2)
        "BC": BCSinglePhaseModeratePressure,
        "IC": ICSinglePhaseModeratePressure,
        "FlowModel": SinglePhaseFlowModel,
        "tf": 120 * 365 * 86400,  # 120 years
        "dt": 365 * 86400,  # 1 years 
    },
    "single_phase_LP": { # Low-pressure single-phase (Figure 2, Case 3)
        "BC": BCSinglePhaseLowPressure,
        "IC": ICSinglePhaseLowPressure,
        "FlowModel": SinglePhaseFlowModel,
        "tf": 1500 * 365 * 86400,
        "dt": 365 * 86400,
    },
    "two_phase_HP": {  # High-pressure two-phase (Figure 3)
        "BC": BCTwoPhaseHighPressure,
        "IC": ICTwoPhaseHighPressure,
        "FlowModel": TwoPhaseFlowModel,
        "tf": 200 * 365 * 86400,
        "dt": 100 * 86400,
    },
    "two_phase_LP": { # Low-pressure two-phase (Figure 4)
        "BC": BCTwoPhaseLowPressure,
        "IC": ICTwoPhaseLowPressure,
        "FlowModel": TwoPhaseFlowModel,
        "tf": 2000.0 * 365.0 * 86400,
        "dt": 365.0 * 86400,
    },
}

# Define material properties
solid_constants = pp.SolidConstants(
    {
        "permeability": 1.0e-15,  # m^2
        "porosity": 0.1,  # dimensionless
        "thermal_conductivity": 1.9,  # W/(m.K)
        "density": 2700.0,  # kg/m^3
        "specific_heat_capacity": 880.0,  # J/(kg.K)
    }
)
material_constants = {"solid": solid_constants}


def create_dynamic_model(BC, IC, FlowModel):
    """Create a geothermal model class with specific BC, IC, and Flow Model."""
    class GeothermalSingleComponentFlowModel(ModelGeometry, BC, IC, FlowModel):
        def after_nonlinear_convergence(self) -> None:
            """Print solver statistics after each nonlinear iteration."""
            super().after_nonlinear_convergence()
            print(f"Number of iterations: {self.nonlinear_solver_statistics.num_iteration}")
            print(f"Time value (years): {self.time_manager.time / (365 * 86400):.2f}")
            print(f"Time index: {self.time_manager.time_index}\n")

        def after_simulation(self):
            """Export results after the simulation."""
            self.exporter.write_pvd()
    return GeothermalSingleComponentFlowModel


def run_simulation(
    case_name : str,
    config : dict[str, any],
    correl_vtk_phz : str
):

    """
    Run a simulation based on the provided configuration.

    Args:
        case_name (str): Name of the simulation case.
        config (dict): Dictionary containing BC, IC, Flow Model, and simulation time settings.
        correl_vtk_phz (str): Path to the VTK file for phase/fluid mixture thermodynamic property sampling.

    The function loads the model, prepares the simulation, 
    runs it, and plot the results, which are then saved in the same directory as the script.
    """
    print(f"\n Running simulation: {case_name}")  
    BC, IC, FlowModel = config["BC"], config["IC"], config["FlowModel"]
    tf, dt = config["tf"], config["dt"]

    # Create dynamic model
    GeothermalModel = create_dynamic_model(BC, IC, FlowModel)
    
    # Simulation time settings
    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
        iter_max=100,
        print_info=True
    )

    # Model parameters
    params = {
        "material_constants": material_constants,
        "eliminate_reference_phase": True,
        "eliminate_reference_component": True,
        "time_manager": time_manager,
        "prepare_simulation": False,
        "reduce_linear_system_q": False,
        "nl_convergence_tol": np.inf,
        "nl_convergence_tol_res": 1.0e-3,
        "max_iterations": 100,
    }

    # Initialize model
    model = GeothermalModel(params)

    # Load VTK files
    correl_vtk_ptz = "/workdir/porepy/src/porepy/examples/geothermal_flow/model_configuration/constitutive_description/driesner_vtk_files/XTP_l2_modified.vtk"
    brine_vtk_sampler_phz = VTKSampler(correl_vtk_phz)
    brine_vtk_sampler_phz.conversion_factors = (1.0, 1.0e-3, 1.0e-5)
    model.vtk_sampler = brine_vtk_sampler_phz
    brine_vtk_sampler_ptz = VTKSampler(correl_vtk_ptz)
    brine_vtk_sampler_ptz.conversion_factors = (1.0, 1.0, 1.0e-5)  # (z,t,p)
    brine_vtk_sampler_ptz.translation_factors = (0.0, -273.15, 0.0)  # (z,t,p)
    model.vtk_sampler_ptz = brine_vtk_sampler_ptz

    # Prepare and run simulation
    start_time = time.time()
    model.prepare_simulation()
    print(f"Elapsed time for preparation: {time.time() - start_time:.2f} seconds")
    print(f"Simulation prepared for total DoFs: {model.equation_system.num_dofs()}")
    print(f"Grid info: {model.mdg}")

    # Export geometry
    # model.exporter.write_vtu()
    start_time = time.time()

    # Run the simulation
    pp.run_time_dependent_model(model, params)
    print(f"Elapsed time for simulation: {time.time() - start_time:.2f} seconds")
    print(f"Total DoFs: {model.equation_system.num_dofs()}")
    print(f"Grid info: {model.mdg}")

    # Retrieve grid and boundary info
    grid = model.mdg.subdomains()[0]

    # Compute mass flux
    darcy_flux = model.darcy_flux(model.mdg.subdomains()).value(model.equation_system)
    inlet_idx, outlet_idx = model.get_inlet_outlet_sides(grid)
    print(f"Inflow values: {darcy_flux[inlet_idx]}")
    print(f"Outflow values: {darcy_flux[outlet_idx]}")

    # Get the last time step's solution data
    pvd_file = "./visualization/data.pvd"
    mesh = data_util.get_last_mesh_from_pvd(pvd_file)

    # Load saved csmp++ temperature, pressure, and saturation data
    num_points = 200
    dx = 2.0 / (num_points - 1)  # cell size
    xc = np.arange(0.0, 2.0 + dx, dx)
    pressure_csmp = []
    temperature_csmp = []
    saturation_csmp = []

    OUTPUT_DIR = os.getcwd()
    if case_name == "single_phase_HP":
        save_path = os.path.join(OUTPUT_DIR, f"{case_name}.png")
        simulation_time = 250

        pressure_csmp, temperature_csmp = fig_4A_load_and_project_reference_data(xc)

        data_util.plot_temp_pressure_comparison(
            mesh,
            pressure_csmp,
            temperature_csmp,
            xc,
            [150, 350],
            np.linspace(150, 350, 5),
            [25, 50],
            np.linspace(25, 50, 6),
            simulation_time,
            save_path,
        )
    if case_name == "single_phase_MP":
        save_path = os.path.join(OUTPUT_DIR, f"{case_name}.png")
        simulation_time = 120
        pressure_csmp, temperature_csmp = data_util.fig_4C_load_and_project_reference_data(xc)

        data_util.plot_temp_pressure_comparison(
            mesh,
            pressure_csmp,
            temperature_csmp,
            xc,
            [295, 460],
            np.linspace(300, 450, 4),
            [20, 40],
            np.linspace(20, 40, 5),
            simulation_time,
            save_path,
        )
    if case_name == "single_phase_LP":
        save_path = os.path.join(OUTPUT_DIR, f"{case_name}.png")
        simulation_time = 1500
        pressure_csmp, temperature_csmp = data_util.fig_4E_load_and_project_reference_data(xc)

        data_util.plot_temp_pressure_comparison(
            mesh,
            pressure_csmp,
            temperature_csmp,
            xc,
            [280, 513],
            np.linspace(300, 500, 5),
            [0, 15],
            np.linspace(0.0, 15, 6),
            simulation_time,
            save_path,
        )
    
    if case_name == "two_phase_HP":
        simulation_time = 200
        # Extract the 'pressure' data (cell data)
        centroids = mesh.cell_centers().points
        x_coords = centroids[:, 0]*1e-3
        # Load saturation data
        s_gas = mesh.cell_data['s_gas']
        s_liq = 1 - s_gas
        mask = (s_liq >= 0.1) & (s_liq < 1.0)
        filtered_coords = centroids[mask][:,0]*1e-3
        min_x = np.min(filtered_coords)
        max_x = np.max(filtered_coords)
        # Extract the 'pressure' and 'temperature' data (cell data)
        pressure = mesh.cell_data['pressure'] * 1e-6  # in MPa
        temperature = -273.15 + mesh.cell_data['temperature']  # in oC

        data_util.plot_temp_pressure_two_phase(
            x_coords,
            temperature,
            [145, 405],
            np.linspace(150, 400, 6),
            pressure,
            [0, 20],
            np.linspace(0, 20, 5),
            min_x,
            max_x,
            simulation_time,
            "porepy",
            os.path.join(OUTPUT_DIR, "two_phase_porepy_HP.png")
        )
        
        data_util.plot_liquid_saturation(
            x_coords,
            s_liq,
            min_x,
            max_x,
            os.path.join(OUTPUT_DIR, "two_phase_saturation_porepy_HP.png")
        )

        # Load CSMP++ results
        pressure_csmp, temperature_csmp, saturation_csmp = data_util.fig_5_load_and_project_reference_data(xc)
        # TWO-PHASE REGION
        mask = (saturation_csmp >= 0.23) & (saturation_csmp < 1.0)
        filtered_coords = centroids[mask][:, 0]*1e-3
        min_x_csmp = np.min(filtered_coords)
        max_x_csmp = np.max(filtered_coords)

        data_util.plot_temp_pressure_two_phase(
            x_coords,
            temperature_csmp,
            [145, 405],
            np.linspace(150, 400, 6),
            pressure_csmp,
            [0, 20],
            np.linspace(0, 20, 5),
            min_x_csmp,
            max_x_csmp,
            simulation_time,
            "csmp++",
            os.path.join(OUTPUT_DIR, "two_phase_csmp_HP.png")
        )

        data_util.plot_liquid_saturation(
            x_coords,
            saturation_csmp,
            min_x_csmp,
            max_x_csmp,
            os.path.join(OUTPUT_DIR, "two_phase_saturation_csmp_HP.png")
        )

    if case_name == "two_phase_LP":
        simulation_time = 200
        # Extract the 'pressure' data (cell data)
        centroids = mesh.cell_centers().points
        x_coords = centroids[:, 0]*1e-3
        # Load saturation data
        s_gas = mesh.cell_data['s_gas']
        s_liq = 1 - s_gas
        mask = (s_liq >= 0.1) & (s_liq < 0.7)
        filtered_coords = centroids[mask][:, 0]*1e-3
        min_x = np.min(filtered_coords)
        max_x = np.max(filtered_coords)
        # Extract the 'pressure' and 'temperature' data (cell data)
        pressure = mesh.cell_data['pressure'] * 1e-6  # in MPa
        temperature = -273.15 + mesh.cell_data['temperature']  # in oC

        data_util.plot_temp_pressure_two_phase(
            x_coords,
            temperature,
            [150, 300],
            np.linspace(150, 300, 4),
            pressure,
            [1, 4],
            np.linspace(1, 4, 4),
            min_x,
            max_x,
            simulation_time,
            "porepy",
            os.path.join(OUTPUT_DIR, "two_phase_porepy_LP.png")
        )

        data_util.plot_liquid_saturation(
            x_coords,
            s_liq,
            min_x,
            max_x,
            os.path.join(OUTPUT_DIR, "two_phase_saturation_porepy_LP.png")
        )

        # Load CSMP++ results
        pressure_csmp, temperature_csmp, saturation_csmp = data_util.fig_6_load_and_project_reference_data(xc)
        # TWO-PHASE REGION
        mask = (saturation_csmp >= 0.23) & (saturation_csmp < 1.0)
        filtered_coords = centroids[mask][:, 0]*1e-3
        min_x_csmp = np.min(filtered_coords)
        max_x_csmp = np.max(filtered_coords)

        data_util.plot_temp_pressure_two_phase(
            x_coords,
            temperature_csmp,
            [150, 300],
            np.linspace(150, 300, 4),
            pressure_csmp,
            [1, 4],
            np.linspace(1, 4, 4),
            min_x_csmp,
            max_x_csmp,
            simulation_time,
            "csmp++",
            os.path.join(OUTPUT_DIR, "two_phase_csmp_LP.png")
        )

        data_util.plot_liquid_saturation(
            x_coords,
            saturation_csmp,
            min_x_csmp,
            max_x_csmp,
            os.path.join(OUTPUT_DIR, "two_phase_saturation_csmp_LP.png")
        )


# ------------------------------------------------------
# Run Simulations for All Configured Cases
# ------------------------------------------------------

# Define file paths for VTK files used for thermodynamic property sampling
file_prefix = "/workdir/porepy/src/porepy/examples/geothermal_flow/model_configuration/constitutive_description/driesner_vtk_files/"
correl_vtk_phz_1 = f"{file_prefix}XHP_l2_original_sc.vtk"
correl_vtk_phz_2 = f"{file_prefix}XHP_l2_original_all.vtk"

for case_name, config in SIMULATION_CASES.items():
    if case_name in {'single_phase_MP'}:
        run_simulation(case_name, config, correl_vtk_phz_1)
    else:
        run_simulation(case_name, config, correl_vtk_phz_2)