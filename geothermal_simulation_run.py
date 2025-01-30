from __future__ import annotations

import time
import numpy as np
import porepy as pp
import os

from porepy.examples.geothermal_flow.model_configuration.flow_model_configuration import (
    SinglePhaseFlowModelConfiguration as SinglePhaseFlowModel,
    TwoPhaseFlowModelConfiguration as TwoPhaseFlowModel,
)
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

from porepy.examples.geothermal_flow.data_extractor_util import *

# Geometry description
from model_configuration.geometry_description.geometry_market import SimpleGeometryHorizontal as ModelGeometry

# Boundary & Initial Conditions
from model_configuration.bc_description.bc_market import (
    BCSinglePhaseHighPressure, 
    BCSinglePhaseModeratePressure, 
    BCSinglePhaseLowPressure,
    BCTwoPhaseHighPressure, 
    BCTwoPhaseLowPressure
)
from model_configuration.ic_description.ic_market import (
    ICSinglePhaseHighPressure, 
    ICSinglePhaseModeratePressure, 
    ICSinglePhaseLowPressure,
    ICTwoPhaseHighPressure, 
    ICTwoPhaseLowPressure
)

# Simulation configurations
SIMULATION_CASES = {  # Figure 2: Case 1
    "single_phase_HP": {
        "BC": BCSinglePhaseHighPressure,
        "IC": ICSinglePhaseHighPressure,
        "FlowModel": SinglePhaseFlowModel,
        "tf": 250 * 365 * 86400,  # 250 years
        "dt": 1.0 * 365 * 86400,  # 1 year 
    },
    "single_phase_MP": {  # Figure 2: Case 2
        "BC": BCSinglePhaseModeratePressure,
        "IC": ICSinglePhaseModeratePressure,
        "FlowModel": SinglePhaseFlowModel,
        "tf": 120 * 365 * 86400,  # 120 years
        "dt": 1.0 * 365 * 86400,  # 1 years 
    },
    "single_phase_LP": {  # Figure 2: Case 3
        "BC": BCSinglePhaseLowPressure,
        "IC": ICSinglePhaseLowPressure,
        "FlowModel": SinglePhaseFlowModel,
        "tf": 1500 * 365 * 86400,
        "dt": 1.0 * 365 * 86400,
    },
    "two_phase_HP": {
        "BC": BCTwoPhaseHighPressure,
        "IC": ICTwoPhaseHighPressure,
        "FlowModel": TwoPhaseFlowModel,
        "tf": 200 * 365 * 86400,
        "dt": 1.0 * 100 * 86400,
    },
    "two_phase_LP": {
        "BC": BCTwoPhaseLowPressure,
        "IC": ICTwoPhaseLowPressure,
        "FlowModel": TwoPhaseFlowModel,
        "tf": 2000.0 * 365.0 * 86400,     # 2000
        "dt": 1.0 * 365.0 * 86400,
    },
}

# Material Properties
solid_constants = pp.SolidConstants(
    {
        "permeability": 1.0e-15,
        "porosity": 0.1,
        "thermal_conductivity": 1.9,
        "density": 2700.0,
        "specific_heat_capacity": 880.0,
    }
)
material_constants = {"solid": solid_constants}


def create_dynamic_model(BC, IC, FlowModel):
    """Dynamically create a geothermal model class with specific BC, IC, and Flow Model."""
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
    file_prefix : str
):

    """Run a geothermal simulation based on the provided configuration."""
    print(f"\n Running simulation: {case_name}")  
    BC, IC, FlowModel = config["BC"], config["IC"], config["FlowModel"]
    tf, dt = config["tf"], config["dt"]

    # Create dynamic model
    GeothermalModel = create_dynamic_model(BC, IC, FlowModel)
    
    # Time settings
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
    file_name_phz = file_prefix
    # file_name_ptz = "model_configuration/constitutive_description/driesner_vtk_files/XTP_l2_modified.vtk"
    file_name_ptz = "/Users/michealoguntola/Desktop/MainPorepy/composite-flow-latest/porepy/src/porepy/examples/geothermal_flow/model_configuration/constitutive_description/driesner_vtk_files/" + "XTP_l2_modified.vtk"

    brine_vtk_sampler_phz = VTKSampler(file_name_phz)
    brine_vtk_sampler_phz.conversion_factors = (1.0, 1.0e-3, 1.0e-5)
    model.vtk_sampler = brine_vtk_sampler_phz

    brine_vtk_sampler_ptz = VTKSampler(file_name_ptz)
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

    # Run time-dependent simulation
    start_time = time.time()
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
    mesh = get_last_mesh_from_pvd(pvd_file)

    # Load saved csmp++ temperature, pressure, and/or saturation data accordingly
    num_points = 200
    dx = 2.0 / (num_points - 1)
    xc = np.arange(0.0, 2.0 + dx, dx)

    pressure_csmp = []
    temperature_csmp = []
    saturation_csmp = []

    folder_dir = os.getcwd()  # os.path.dirname(os.path.abspath(pvd_file))
    if case_name == "single_phase_HP":
        save_path = os.path.join(folder_dir, "single_phase_HP.png")
        simulation_time = 250

        pressure_csmp, temperature_csmp = fig_4A_load_and_project_reference_data(xc)
        plot_temp_pressure_comparison(
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
        save_path = os.path.join(folder_dir, "single_phase_MP.png")
        simulation_time = 120
        pressure_csmp, temperature_csmp = fig_4C_load_and_project_reference_data(xc)
        plot_temp_pressure_comparison(
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
        save_path = os.path.join(folder_dir, "single_phase_LP.png")
        simulation_time = 1500
        pressure_csmp, temperature_csmp = fig_4E_load_and_project_reference_data(xc)
        plot_temp_pressure_comparison(
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

        plot_temp_pressure_two_phase(
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
            os.path.join(folder_dir, "two_phase_porepy_HP.png")
        )
        
        plot_liquid_saturation(
            x_coords,
            s_liq,
            min_x,
            max_x,
            os.path.join(folder_dir, "two_phase_saturation_porepy_HP.png")
        )

        # Load CSMP++ Data
        pressure_csmp, temperature_csmp, saturation_csmp = fig_5_load_and_project_reference_data(xc)
        ### TWO-PHASE REGION
        mask = (saturation_csmp >= 0.23) & (saturation_csmp < 1.0)
        filtered_coords = centroids[mask][:,0]*1e-3
        min_x_csmp = np.min(filtered_coords)
        max_x_csmp = np.max(filtered_coords)

        plot_temp_pressure_two_phase(
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
            os.path.join(folder_dir, "two_phase_csmp_HP.png")
        )

        plot_liquid_saturation(
            x_coords,
            saturation_csmp,
            min_x_csmp,
            max_x_csmp,
            os.path.join(folder_dir, "two_phase_saturation_csmp_HP.png")
        )

    if case_name == "two_phase_LP":
        simulation_time = 2000
        
        # Extract the 'pressure' data (cell data)
        centroids = mesh.cell_centers().points
        x_coords = centroids[:, 0]*1e-3

        # Load saturation data
        s_gas = mesh.cell_data['s_gas']
        # s_gas = np.ceil(s_gas * 10) / 10
        s_liq = 1 - s_gas
        mask = (s_liq >= 0.1) & (s_liq < 0.7)
        filtered_coords = centroids[mask][:,0]*1e-3
        min_x = np.min(filtered_coords)
        max_x = np.max(filtered_coords)

        # Extract the 'pressure' and 'temperature' data (cell data)
        pressure = mesh.cell_data['pressure'] * 1e-6  # in MPa
        temperature = -273.15 + mesh.cell_data['temperature']  # in oC

        plot_temp_pressure_two_phase(
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
            os.path.join(folder_dir, "two_phase_porepy_LP.png")
        )
        
        plot_liquid_saturation(
            x_coords,
            s_liq,
            min_x,
            max_x,
            os.path.join(folder_dir, "two_phase_saturation_porepy_LP.png")
        )

        # Load CSMP++ Data
        pressure_csmp, temperature_csmp, saturation_csmp = fig_6_load_and_project_reference_data(xc)
        ### TWO-PHASE REGION
        mask = (saturation_csmp >= 0.23) & (saturation_csmp < 1.0)
        filtered_coords = centroids[mask][:,0]*1e-3
        min_x_csmp = np.min(filtered_coords)
        max_x_csmp = np.max(filtered_coords)

        plot_temp_pressure_two_phase(
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
            os.path.join(folder_dir, "two_phase_csmp_LP.png")
        )
        plot_liquid_saturation(
            x_coords,
            saturation_csmp,
            min_x_csmp,
            max_x_csmp,
            os.path.join(folder_dir, "two_phase_saturation_csmp_LP.png")
        )


# Run simulations based on configuration
file_prefix = "model_configuration/constitutive_description/driesner_vtk_files/"
correl_vtk_phz_1 = f"{file_prefix}XHP_l2_original_sc.vtk"
correl_vtk_phz_2 = f"{file_prefix}XHP_l2_original_all.vtk"
for case_name, config in SIMULATION_CASES.items():
    if case_name in {'single_phase_MP'}:
        run_simulation(case_name, config, correl_vtk_phz_1)
    else:
        run_simulation(case_name, config, correl_vtk_phz_2)
