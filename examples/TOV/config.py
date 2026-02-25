"""
TOV Simulation Configuration.

Defines the configuration parameters for TOV star evolution.
"""

import os


class TOVConfig:
    """Configuration class for TOV star evolution."""

    def __init__(self,
                 r_max=50.0,
                 num_points=500,
                 K=100.0,
                 Gamma=2.0,
                 rho_central=1.28e-3,
                 t_final=200.0,
                 cfl_factor=0.1,
                 integration_method='fixed',
                 reconstructor="mp5",
                 solver_method="newton",
                 riemann_solver="hll",
                 evolution_mode="dynamic",
                 collapse_perturbation=False,
                 collapse_amplitude=0.01,
                 enable_restart=False,
                 enable_data_saving=True,
                 save_timeseries=True,
                 skip_plots=False,
                 snapshot_interval=500,
                 evolution_interval=500,
                 jax_run=None):

        self.r_max = r_max
        self.num_points = num_points
        self.K = K
        self.Gamma = Gamma
        self.rho_central = rho_central
        self.t_final = t_final
        self.cfl_factor = cfl_factor
        self.integration_method = integration_method
        self.reconstructor = reconstructor
        self.solver_method = solver_method
        self.riemann_solver = riemann_solver
        self.evolution_mode = evolution_mode
        self.collapse_perturbation = collapse_perturbation
        self.collapse_amplitude = collapse_amplitude
        self.enable_restart = enable_restart
        self.enable_data_saving = enable_data_saving
        self.save_timeseries = save_timeseries
        self.skip_plots = skip_plots
        self.snapshot_interval = snapshot_interval
        self.evolution_interval = evolution_interval

        if jax_run is None:
            jax_run = os.environ.get("JAX_RUN", "1").lower() in ("1", "true", "yes")
        self.jax_run = jax_run

        self._resolve_from_env()

    def _resolve_from_env(self):
        """Override config from environment variables if set."""
        env_vars = {
            'num_points': ('NUM_POINTS', int),
            't_final': ('T_FINAL', float),
            'enable_restart': ('ENABLE_RESTART', lambda x: x.lower() in ("1", "true", "yes")),
            'enable_data_saving': ('ENABLE_DATA_SAVING', lambda x: x.lower() in ("1", "true", "yes")),
            'skip_plots': ('SKIP_PLOTS', lambda x: x.lower() in ("1", "true", "yes")),
        }

        for attr, (env_name, converter) in env_vars.items():
            value = os.environ.get(env_name)
            if value is not None:
                try:
                    setattr(self, attr, converter(value))
                except (ValueError, TypeError):
                    pass

    @property
    def folder_name(self):
        """Generate folder name based on parameters."""
        return f"tov_evolution_data_refact_rmax{self.r_max}" + ("_jax" if self.jax_run else "")

    @property
    def plot_suffix(self):
        """Generate plot suffix based on mode and backend."""
        mode_suffix = "_cow" if self.evolution_mode == "cowling" else "_dyn"
        backend_suffix = "_jax" if self.jax_run else ""
        return mode_suffix + backend_suffix

    def get_atmosphere_params(self):
        """Get atmosphere parameters."""
        from source.matter.hydro.atmosphere import AtmosphereParams
        rho_floor_base = 1e-13
        p_floor_base = self.K * (rho_floor_base)**self.Gamma
        return AtmosphereParams(
            rho_floor=rho_floor_base,
            p_floor=p_floor_base
        )

    def get_eos(self):
        """Get equation of state."""
        from source.matter.hydro.eos import IdealGasEOS
        return IdealGasEOS(gamma=self.Gamma)

    def get_output_dir(self, script_dir):
        """Return the full output directory path for this simulation."""
        from examples.TOV.utils.io import get_star_folder_name
        root = os.path.join(script_dir, "data", self.folder_name)
        star = get_star_folder_name(
            self.rho_central, self.num_points, self.K, self.Gamma,
            self.evolution_mode, self.reconstructor,
        )
        return os.path.join(root, star)

    def __repr__(self):
        return (f"TOVConfig(r_max={self.r_max}, num_points={self.num_points}, "
                f"K={self.K}, Gamma={self.Gamma}, rho_central={self.rho_central}, "
                f"t_final={self.t_final}, mode={self.evolution_mode}, "
                f"backend={'JAX' if self.jax_run else 'Numba'})")
