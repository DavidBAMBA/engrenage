import numpy as np
import matplotlib.pyplot as plt
import os

# --- Importaciones de Engrenage ---
# Asegúrate de que la ruta a la carpeta 'source' esté en el PYTHONPATH
# o ejecuta este script desde el directorio raíz de engrenage.
from core.grid import Grid
from core.statevector import StateVector
from core.rhsevolution import RHSEvolution
from bssn.bssnrhs import BSSNRHS
from matter.nomatter import NoMatter

# --- Importaciones de tu módulo de Hidrodinámica ---
from matter.hydro.relativistic_fluid import RelativisticFluid
from matter.hydro.eos import IdealGasEOS
from matter.hydro.reconstruction import MinmodReconstruction
from matter.hydro.riemann import HLLERiemannSolver

# --------------------------------------------------------------------------
# PASO 1: CONFIGURACIÓN INICIAL DE LA BLAST WAVE (de blast_wave_setup.py)
# --------------------------------------------------------------------------
def setup_blast_wave_initial_data(r, p_inner, p_outer, rho_inner, rho_outer, r_discontinuity, eos):
    """
    Configura los datos iniciales para el problema de la onda de choque esférica relativista.
    Crea una discontinuidad de presión y densidad en r = r_discontinuity.
    La velocidad es inicialmente cero en todas partes.
    """
    N = len(r)
    rho0 = np.zeros(N)
    vr = np.zeros(N)
    pressure = np.zeros(N)

    # Discontinuidad aguda
    inner_mask = r < r_discontinuity
    outer_mask = r >= r_discontinuity
    
    rho0[inner_mask] = rho_inner
    rho0[outer_mask] = rho_outer
    pressure[inner_mask] = p_inner
    pressure[outer_mask] = p_outer
    
    # Calcular variables conservadoras desde las primitivas
    eps = eos.eps_from_rho_p(rho0, pressure)
    h = 1.0 + eps + pressure / rho0
    W = 1.0 / np.sqrt(1.0 - vr**2) # Inicialmente W=1 ya que vr=0
    
    D = rho0 * W
    Sr = rho0 * h * W**2 * vr
    tau = rho0 * h * W**2 - pressure - D
    
    return {
        'primitives': {'rho0': rho0, 'vr': vr, 'pressure': pressure},
        'conservatives': {'D': D, 'Sr': Sr, 'tau': tau}
    }

# --------------------------------------------------------------------------
# PASO 2: FUNCIÓN PRINCIPAL PARA EJECUTAR LA SIMULACIÓN
# --------------------------------------------------------------------------
def run_simulation():
    """
    Función principal que configura y ejecuta la simulación de la blast wave.
    """
    print("🚀 Iniciando simulación de la Blast Wave Relativista...")

    # --- Parámetros de la simulación ---
    N = 400              # Resolución espacial
    r_max = 1.0          # Radio máximo del dominio
    t_final = 0.4        # Tiempo final de la evolución
    output_frequency = 5 # Frecuencia para guardar datos y mostrar plots

    # --- Configuración de la física ---
    # Parámetros para un test clásico (similar a Martí & Müller 2015)
    p_inner = 1.0
    p_outer = 0.1
    rho_inner = 1.0
    rho_outer = 0.125
    r_discontinuity = 0.5
    gamma_eos = 4.0 / 3.0 # EoS de gas de radiación
    
    # --- Configuración de la malla (Grid) ---
    grid = Grid(N, 0.0, r_max)
    
    # --- Configuración de la materia (el corazón de tu implementación) ---
    eos = IdealGasEOS(gamma=gamma_eos)
    reconstructor = MinmodReconstruction(limiter_type="mc")
    riemann_solver = HLLERiemannSolver()

    # Aquí se instancia tu clase principal
    hydro_fluid = RelativisticFluid(
        eos=eos,
        spacetime_mode="fixed_minkowski", # ¡CLAVE para tu test!
        reconstructor=reconstructor,
        riemann_solver=riemann_solver
    )
    
    # --- Configuración de los datos iniciales ---
    initial_data = setup_blast_wave_initial_data(
        grid.r, p_inner, p_outer, rho_inner, rho_outer, r_discontinuity, eos
    )
    
    # --- Creación del vector de estado inicial ---
    # El vector de estado combina variables BSSN (vacías aquí) y de materia
    num_total_vars = BSSNRHS.NUM_VARS + hydro_fluid.NUM_MATTER_VARS
    initial_state_vec = StateVector(num_total_vars, N)

    # Inyectar las variables conservadoras en el vector de estado
    initial_state_vec.set_vars_from_dict(initial_data['conservatives'])
    
    # --- Configuración del motor de evolución de Engrenage ---
    # Como no evolucionamos el espaciotiempo, usamos NoMatter() para la parte de BSSN
    # pero tu 'hydro_fluid' se pasa como el objeto de materia real.
    rhs_evolution = RHSEvolution(
        grid=grid,
        bssn_rhs=BSSNRHS(background=None), # No se usará, pero es requerido
        matter=hydro_fluid,
        initial_state=initial_state_vec,
        time=0.0
    )
    
    # --- Bucle de evolución ---
    print("Iniciando bucle de evolución temporal...")
    
    # Configurar la figura para el ploteo en vivo
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Evolución de la Blast Wave Relativista")
    
    # Guardar datos iniciales para comparación
    rho_initial = initial_data['primitives']['rho0'].copy()

    for i, (t, state_vector) in enumerate(rhs_evolution.evolve(t_final, dt_cfl=0.4)):
        if i % output_frequency == 0:
            print(f"t = {t:.3f} / {t_final:.3f}")
            
            # Extraer las variables primitivas para plotear
            # Se acceden a través del objeto 'hydro_fluid' que se actualiza en cada paso
            rho0 = hydro_fluid.rho0
            vr = hydro_fluid.vr
            pressure = hydro_fluid.pressure
            W = hydro_fluid.W

            # Limpiar y actualizar plots
            for ax in axes.flat:
                ax.clear()

            # Densidad
            axes[0, 0].plot(grid.r, rho_initial, 'k--', label='Inicial', alpha=0.5)
            axes[0, 0].plot(grid.r, rho0, 'b-', label=f't = {t:.2f}')
            axes[0, 0].set_ylabel("Densidad (ρ₀)")
            axes[0, 0].legend()
            
            # Presión
            axes[0, 1].plot(grid.r, pressure, 'r-')
            axes[0, 1].set_ylabel("Presión (p)")

            # Velocidad
            axes[1, 0].plot(grid.r, vr, 'g-')
            axes[1, 0].set_ylabel("Velocidad radial (vʳ)")
            axes[1, 0].set_xlabel("Radio (r)")

            # Factor de Lorentz
            axes[1, 1].plot(grid.r, W, 'm-')
            axes[1, 1].set_ylabel("Factor de Lorentz (W)")
            axes[1, 1].set_xlabel("Radio (r)")

            for ax in axes.flat:
                ax.grid(True, linestyle=':', alpha=0.6)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.pause(0.01)

    print("✅ Simulación completada.")
    plt.ioff()
    
    # Guardar el plot final
    output_dir = "blast_wave_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    final_plot_path = os.path.join(output_dir, "blast_wave_final_state.png")
    fig.savefig(final_plot_path)
    print(f"Plot final guardado en: {final_plot_path}")
    
    plt.show()

# --------------------------------------------------------------------------
# PUNTO DE ENTRADA
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Para que funcione, asegúrate de que el directorio `source` de engrenage
    # esté en tu PYTHONPATH, o ejecuta este script desde el directorio raíz.
    # Ejemplo: export PYTHONPATH=$PYTHONPATH:/path/to/engrenage/source
    run_simulation()