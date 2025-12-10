#!/usr/bin/env python
"""
Quick Density Profile Plotter

Este script plotea rápidamente el perfil de densidad de los snapshots
guardados durante la evolución de TOV.

Uso:
    python plot_density_quick.py
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import os
from glob import glob

# Directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'tov_evolution_data')
plots_dir = os.path.join(script_dir, 'plots')

# Crear directorio de plots si no existe
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

def load_latest_snapshots(data_dir, suffix="_iso"):
    """Carga los snapshots más recientes."""
    pattern = os.path.join(data_dir, f'tov_snapshots{suffix}_*.h5')
    snapshot_files = glob(pattern)

    if not snapshot_files:
        print(f"No se encontraron archivos de snapshots en {data_dir}")
        return None

    # Usar el archivo más reciente
    snapshot_file = sorted(snapshot_files)[-1]
    print(f"\nCargando snapshots de: {os.path.basename(snapshot_file)}")

    try:
        with h5py.File(snapshot_file, 'r') as f:
            # Cargar grid
            r = f['grid/r'][:]

            # Obtener todos los snapshots disponibles
            snap_names = sorted([k for k in f['snapshots'].keys()],
                              key=lambda x: int(x.split('_')[1]))

            print(f"Total de snapshots encontrados: {len(snap_names)}")

            snapshots = []
            for snap_name in snap_names:
                snap = f['snapshots'][snap_name]
                step = snap.attrs['step']
                time = snap.attrs['time']

                # Cargar densidad si está disponible
                rho0 = None
                if 'primitives/rho0' in snap:
                    rho0 = snap['primitives/rho0'][:]
                elif 'rho0' in snap:
                    rho0 = snap['rho0'][:]

                snapshots.append({
                    'step': step,
                    'time': time,
                    'r': r,
                    'rho0': rho0
                })

            return snapshots

    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        print("Asegúrate de que la simulación no esté corriendo o espera a que termine.")
        return None

def load_metadata(data_dir, suffix="_iso"):
    """Carga metadatos de la simulación."""
    pattern = os.path.join(data_dir, f'tov_metadata{suffix}_*.json')
    metadata_files = glob(pattern)

    if not metadata_files:
        return None

    metadata_file = sorted(metadata_files)[-1]
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return metadata

def plot_density_evolution(snapshots, metadata=None):
    """Plotea la evolución del perfil de densidad."""

    # Seleccionar snapshots para plotear (inicial, intermedios, final)
    n_snaps = len(snapshots)
    if n_snaps >= 4:
        indices = [0, n_snaps//3, 2*n_snaps//3, n_snaps-1]
    elif n_snaps >= 2:
        indices = [0, n_snaps-1]
    else:
        indices = [0]

    print(f"\nPloteando {len(indices)} snapshots:")
    for idx in indices:
        snap = snapshots[idx]
        print(f"  Step {snap['step']:6d}, t = {snap['time']:.6e}")

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colores para cada snapshot
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))

    # Plotear cada snapshot
    for i, idx in enumerate(indices):
        snap = snapshots[idx]
        r = snap['r']
        rho0 = snap['rho0']

        if rho0 is not None:
            ax.plot(r, np.maximum(rho0, 1e-20),
                       color=colors[i], linewidth=2,
                       label=f"Step {snap['step']}, t={snap['time']:.3f}")

    # Radio de la estrella (si está en metadata)
    if metadata and 'R_star' in metadata:
        R_star = metadata['R_star']
        ax.axvline(R_star, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label=f'R_star = {R_star:.3f} M')

    # Configuración de la gráfica
    ax.set_xlabel('Radio r [M]', fontsize=12)
    ax.set_ylabel('Densidad bariónica ρ₀ [M⁻²]', fontsize=12)
    ax.set_title('Evolución del Perfil de Densidad - TOV', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)

    # Agregar información de parámetros si está disponible
    if metadata:
        info_text = f"K={metadata.get('K', '?')}, Γ={metadata.get('Gamma', '?')}, "
        info_text += f"ρ_c={metadata.get('rho_central', '?'):.6f}"
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Guardar figura
    output_file = os.path.join(plots_dir, 'density_profile_evolution.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Gráfica guardada en: {output_file}")

    plt.show()

    return fig

def plot_latest_density(snapshots):
    """Plotea solo el snapshot más reciente."""

    if not snapshots:
        print("No hay snapshots disponibles.")
        return

    snap = snapshots[-1]
    r = snap['r']
    rho0 = snap['rho0']

    if rho0 is None:
        print("No se encontró información de densidad en el snapshot.")
        return

    print(f"\nPloteando snapshot más reciente:")
    print(f"  Step {snap['step']}, t = {snap['time']:.6e}")

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotear densidad
    ax.plot(r, np.maximum(rho0, 1e-20),
               color='blue', linewidth=2, label=f"t = {snap['time']:.6e}")

    # Configuración
    ax.set_xlabel('Radio r [M]', fontsize=12)
    ax.set_ylabel('Densidad bariónica ρ₀ [M⁻²]', fontsize=12)
    ax.set_title(f'Perfil de Densidad - Step {snap["step"]}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=11)

    plt.tight_layout()

    # Guardar
    output_file = os.path.join(plots_dir, 'density_profile_latest.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Gráfica guardada en: {output_file}")

    plt.show()

    return fig

def create_density_video(snapshots, metadata=None, fps=10, max_frames=None):
    """Crea un video de la evolución del perfil de densidad.

    Args:
        snapshots: Lista de snapshots
        metadata: Metadata de la simulación
        fps: Frames por segundo para el video
        max_frames: Máximo número de frames (None = todos los snapshots)
    """
    print(f"\n{'='*70}")
    print("CREANDO VIDEO DE EVOLUCIÓN DE DENSIDAD")
    print(f"{'='*70}")

    if not snapshots:
        print("No hay snapshots disponibles.")
        return

    # Decidir qué snapshots usar
    n_snaps = len(snapshots)
    if max_frames and n_snaps > max_frames:
        # Submuestrear uniformemente
        indices = np.linspace(0, n_snaps-1, max_frames, dtype=int)
        frames_to_use = [snapshots[i] for i in indices]
        print(f"\nUsando {max_frames} de {n_snaps} snapshots (submuestreado)")
    else:
        frames_to_use = snapshots
        print(f"\nUsando todos los {n_snaps} snapshots")

    r = frames_to_use[0]['r']

    # Determinar rango de densidad para mantener escala consistente
    all_rho = [s['rho0'] for s in frames_to_use if s['rho0'] is not None]
    if not all_rho:
        print("No hay datos de densidad en los snapshots.")
        return

    rho_min_all = min(np.min(np.maximum(rho, 1e-20)) for rho in all_rho)
    rho_max = max(np.max(rho) for rho in all_rho)

    # Radio de la estrella
    R_star = metadata.get('R_star', None) if metadata else None

    # ==================================================================
    # AJUSTE AUTOMÁTICO DE LÍMITES PARA ENFOCARSE EN LA ESTRELLA
    # ==================================================================
    # Eje X: Mostrar hasta ~2x el radio de la estrella (o detectar automáticamente)
    if R_star:
        x_max = R_star * 2.0  # Mostrar hasta 2x el radio estelar
    else:
        # Si no hay R_star, detectar automáticamente donde termina la estrella
        # (donde la densidad cae por debajo de 100x el piso atmosférico)
        rho_threshold = rho_min_all * 100
        rho_initial = frames_to_use[0]['rho0']
        surface_idx = np.where(rho_initial < rho_threshold)[0]
        if len(surface_idx) > 0:
            R_star_estimated = r[surface_idx[0]]
            x_max = R_star_estimated * 2.0
            print(f"Radio estelar estimado: {R_star_estimated:.3f} M")
        else:
            x_max = r[-1]  # Usar todo el dominio si no se puede estimar

    # Eje Y: Rango relevante para la estrella (no la atmósfera)
    # Usar un piso más realista (10^-6) en lugar del piso atmosférico (10^-10)
    rho_min_plot = rho_max * 1e-6  # 6 órdenes de magnitud por debajo del máximo
    rho_max_plot = rho_max * 1.5    # 1.5x el máximo para dar espacio

    print(f"Rango de densidad (plot): [{rho_min_plot:.3e}, {rho_max_plot:.3e}]")
    print(f"Límite en X: [0, {x_max:.3f}] M")
    print(f"Radio de estrella: {R_star:.3f} M" if R_star else "Radio desconocido")

    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 7))

    # Línea inicial (vacía)
    line, = ax.plot([], [], 'b-', linewidth=2.5)

    # Línea del radio estelar
    if R_star:
        ax.axvline(R_star, color='red', linestyle='--', linewidth=2,
                  alpha=0.5, label=f'R* = {R_star:.3f} M', zorder=1)

    # Configuración de ejes con límites ajustados
    ax.set_xlim(0, x_max)
    ax.set_ylim(rho_min_plot, rho_max_plot)
    ax.set_xlabel('Radio r [M]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Densidad bariónica ρ₀ [M⁻²]', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Solo mostrar leyenda si hay R_star
    if R_star:
        ax.legend(loc='upper right', fontsize=11)

    # Texto para mostrar tiempo y step
    time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Info de parámetros
    if metadata:
        K_val = metadata.get('K', '?')
        Gamma_val = metadata.get('Gamma', '?')
        rho_c_val = metadata.get('rho_central', None)

        info_text = f"K={K_val}, Γ={Gamma_val}"
        if rho_c_val is not None:
            try:
                info_text += f", ρ_c(t=0)={float(rho_c_val):.6f}"
            except (ValueError, TypeError):
                info_text += f", ρ_c(t=0)={rho_c_val}"

        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    def init():
        """Inicialización de la animación."""
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(frame_idx):
        """Actualización para cada frame."""
        snap = frames_to_use[frame_idx]
        rho0 = snap['rho0']

        if rho0 is not None:
            line.set_data(r, np.maximum(rho0, 1e-20))

            # Actualizar texto
            time_text.set_text(f"Step: {snap['step']:6d}\nTime: {snap['time']:.6e} M")

        # Barra de progreso en consola
        if frame_idx % max(1, len(frames_to_use)//20) == 0:
            progress = (frame_idx + 1) / len(frames_to_use) * 100
            print(f"\rGenerando frames: {progress:.1f}% ({frame_idx+1}/{len(frames_to_use)})", end='')

        return line, time_text

    # Crear animación
    print("\nCreando animación...")
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(frames_to_use),
                                  interval=1000/fps,  # intervalo en ms
                                  blit=True,
                                  repeat=True)

    # Guardar video
    print("\n\nGuardando video...")

    # Intentar con diferentes escritores
    writers = [
        ('pillow', 'density_evolution.gif', 'GIF'),
        ('ffmpeg', 'density_evolution.mp4', 'MP4'),
    ]

    saved = False
    for writer_name, filename, format_name in writers:
        try:
            output_file = os.path.join(plots_dir, filename)

            if writer_name == 'pillow':
                writer = animation.PillowWriter(fps=fps)
            elif writer_name == 'ffmpeg':
                writer = animation.FFMpegWriter(fps=fps, bitrate=2000)

            anim.save(output_file, writer=writer, dpi=100)
            print(f"\n✓ Video guardado como {format_name}: {output_file}")
            saved = True
            break
        except Exception as e:
            print(f"\n✗ No se pudo guardar como {format_name}: {e}")
            continue

    if not saved:
        print("\n✗ No se pudo guardar el video con ningún formato.")
        print("Instala ffmpeg para mejor soporte: 'conda install ffmpeg' o 'apt install ffmpeg'")

    plt.close(fig)
    print(f"\n{'='*70}")
    print("Video completado!")
    print(f"{'='*70}\n")

    return anim if saved else None

def main():
    print("="*70)
    print("QUICK DENSITY PROFILE PLOTTER")
    print("="*70)

    # Cargar metadata
    print("\nCargando metadatos...")
    metadata = load_metadata(data_dir)
    if metadata:
        print(f"  K = {metadata.get('K', '?')}")
        print(f"  Gamma = {metadata.get('Gamma', '?')}")
        print(f"  ρ_central = {metadata.get('rho_central', '?')}")
        print(f"  R_star = {metadata.get('R_star', '?')}")

    # Cargar snapshots
    print("\nCargando snapshots...")
    snapshots = load_latest_snapshots(data_dir)

    if not snapshots:
        print("\n¡No se pudieron cargar los snapshots!")
        print("Posibles razones:")
        print("  1. La simulación todavía está corriendo (archivos bloqueados)")
        print("  2. No hay archivos de snapshots en el directorio")
        print("\nSolución: Espera a que termine la simulación o detenla temporalmente.")
        return

    print(f"\n✓ {len(snapshots)} snapshots cargados exitosamente")

    # Menú de opciones
    print("\n" + "="*70)
    print("Opciones:")
    print("  1. Plotear evolución completa (varios snapshots)")
    print("  2. Plotear solo el snapshot más reciente")
    print("  3. Crear video de la evolución de densidad")
    print("="*70)

    choice = input("\nSelecciona una opción (1, 2, o 3) [default: 1]: ").strip()

    if choice == '2':
        plot_latest_density(snapshots)
    elif choice == '3':
        # Configuración del video
        print("\nConfiguración del video:")
        fps_input = input("  FPS (frames por segundo) [default: 10]: ").strip()
        fps = int(fps_input) if fps_input else 10

        max_frames_input = input(f"  Máximo de frames (0 = todos los {len(snapshots)}) [default: 0]: ").strip()
        max_frames = int(max_frames_input) if max_frames_input and int(max_frames_input) > 0 else None

        create_density_video(snapshots, metadata, fps=fps, max_frames=max_frames)
    else:
        plot_density_evolution(snapshots, metadata)

    print("\n" + "="*70)
    print("¡Listo!")
    print("="*70)

if __name__ == "__main__":
    main()
