#!/usr/bin/env python
"""
Script para crear video MP4 de la evolución del pulso hidrodinámico.
Lee los datos CSV generados por test4.py y crea una animación.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import glob
from pathlib import Path

def load_metadata(data_dir):
    """Carga metadatos de la simulación."""
    metadata_file = os.path.join(data_dir, "metadata.csv")
    metadata = {}

    try:
        with open(metadata_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    key, value = row[0], row[1]
                    try:
                        # Try to convert to number
                        if '.' in value:
                            metadata[key] = float(value)
                        else:
                            metadata[key] = int(value)
                    except ValueError:
                        metadata[key] = value
    except FileNotFoundError:
        print(f"Warning: No metadata file found at {metadata_file}")
        metadata = {"N": 64, "r_max": 1.0, "t_final": 0.1}

    return metadata

def load_state_data(filename):
    """Carga un archivo de estado CSV."""
    data = {
        'r': [],
        'lapse': [],
        'phi': [],
        'D': [],
        'Sr': [],
        'tau': []
    }

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            if len(row) >= 6:
                data['r'].append(float(row[0]))
                data['lapse'].append(float(row[1]))
                data['phi'].append(float(row[2]))
                data['D'].append(float(row[3]))
                data['Sr'].append(float(row[4]))
                data['tau'].append(float(row[5]))

    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])

    return data

def create_animation(data_dir="hydro_evolution_data", output_file="hydro_evolution.mp4"):
    """Crea la animación MP4."""

    print(f"🎬 Creando video de evolución hidrodinámicas...")
    print(f"   Directorio de datos: {data_dir}")
    print(f"   Archivo de salida: {output_file}")

    # Cargar metadata
    metadata = load_metadata(data_dir)
    print(f"   Metadatos: N={metadata.get('N', '?')}, t_final={metadata.get('t_final', '?')}")

    # Encontrar todos los archivos de estado
    state_files = sorted(glob.glob(os.path.join(data_dir, "state_*.csv")))
    if not state_files:
        print(f"❌ No se encontraron archivos de estado en {data_dir}")
        return False

    print(f"   Encontrados {len(state_files)} archivos de estado")

    # Cargar todos los datos
    print("   Cargando datos...")
    all_data = []
    times = []

    for i, filename in enumerate(state_files):
        data = load_state_data(filename)
        all_data.append(data)

        # Extraer tiempo del nombre del archivo o usar índice
        try:
            # Asumimos dt_save = 0.001 y archivo ordenado
            dt_save = metadata.get('dt_save', 0.001)
            times.append(i * dt_save)
        except:
            times.append(i * 0.001)

    r = all_data[0]['r']  # Grid común

    # Preparar figura
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Evolución Hidrodinámica: Acoplamiento BSSN-Materia", fontsize=14)

    # Configurar subplots
    ax1.set_xlabel('r'); ax1.set_ylabel('Densidad (D)')
    ax1.set_title('Densidad de Masa'); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('r'); ax2.set_ylabel('Momentum (Sr)')
    ax2.set_title('Densidad de Momento'); ax2.grid(True, alpha=0.3)

    ax3.set_xlabel('r'); ax3.set_ylabel('Lapse (α)')
    ax3.set_title('Función Lapse'); ax3.grid(True, alpha=0.3)

    ax4.set_xlabel('r'); ax4.set_ylabel('Factor Conformal (φ)')
    ax4.set_title('Factor Conformal BSSN'); ax4.grid(True, alpha=0.3)

    # Determinar límites y escalas
    D_max = max(np.max(data['D']) for data in all_data)
    Sr_max = max(np.max(np.abs(data['Sr'])) for data in all_data)
    lapse_min = min(np.min(data['lapse']) for data in all_data)
    lapse_max = max(np.max(data['lapse']) for data in all_data)
    phi_min = min(np.min(data['phi']) for data in all_data)
    phi_max = max(np.max(data['phi']) for data in all_data)

    # Configurar límites
    ax1.set_xlim(r[0], r[-1]); ax1.set_ylim(0, D_max * 1.1)
    ax2.set_xlim(r[0], r[-1]); ax2.set_ylim(-Sr_max * 1.1, Sr_max * 1.1)
    ax3.set_xlim(r[0], r[-1]); ax3.set_ylim(lapse_min * 0.999, lapse_max * 1.001)
    ax4.set_xlim(r[0], r[-1]); ax4.set_ylim(phi_min * 1.001, phi_max * 1.001)

    # Líneas para la animación
    line_D, = ax1.plot([], [], 'b-', linewidth=2, label='D(r,t)')
    line_Sr, = ax2.plot([], [], 'r-', linewidth=2, label='Sr(r,t)')
    line_lapse, = ax3.plot([], [], 'g-', linewidth=2, label='α(r,t)')
    line_phi, = ax4.plot([], [], 'm-', linewidth=2, label='φ(r,t)')

    # Añadir leyendas
    ax1.legend(); ax2.legend(); ax3.legend(); ax4.legend()

    # Texto para tiempo
    time_text = fig.text(0.02, 0.98, '', transform=fig.transFigure, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def animate(frame):
        """Función de animación."""
        if frame >= len(all_data):
            return line_D, line_Sr, line_lapse, line_phi, time_text

        data = all_data[frame]
        t = times[frame]

        # Actualizar datos
        line_D.set_data(r, data['D'])
        line_Sr.set_data(r, data['Sr'])
        line_lapse.set_data(r, data['lapse'])
        line_phi.set_data(r, data['phi'])

        # Actualizar tiempo
        time_text.set_text(f't = {t:.4f}')

        return line_D, line_Sr, line_lapse, line_phi, time_text

    # Crear animación
    print("   Generando frames...")
    anim = animation.FuncAnimation(
        fig, animate, frames=len(all_data),
        interval=100,  # 100ms entre frames = 10 FPS
        blit=True, repeat=True
    )

    # Guardar como MP4
    print(f"   Guardando video como {output_file}...")
    try:
        writer = animation.FFMpegWriter(fps=10, bitrate=1800)
        anim.save(output_file, writer=writer)
        print(f"✅ Video creado exitosamente: {output_file}")

        # Estadísticas
        file_size = os.path.getsize(output_file) / (1024*1024)  # MB
        duration = len(all_data) / 10  # seconds at 10 FPS
        print(f"   📊 Duración: {duration:.1f}s, Tamaño: {file_size:.1f}MB")

        return True

    except Exception as e:
        print(f"❌ Error creando video: {e}")
        print("   Asegúrate de tener ffmpeg instalado: sudo apt install ffmpeg")

        # Fallback: guardar como GIF
        print("   Intentando crear GIF...")
        try:
            gif_file = output_file.replace('.mp4', '.gif')
            anim.save(gif_file, writer='pillow', fps=5)
            print(f"✅ GIF creado: {gif_file}")
            return True
        except Exception as e2:
            print(f"❌ Error creando GIF: {e2}")
            return False

def create_summary_plot(data_dir="hydro_evolution_data", output_file="evolution_summary.png"):
    """Crea un plot resumen de la evolución."""

    print(f"📊 Creando plot resumen...")

    # Cargar metadata
    metadata = load_metadata(data_dir)

    # Encontrar archivos
    state_files = sorted(glob.glob(os.path.join(data_dir, "state_*.csv")))
    if not state_files:
        print(f"❌ No se encontraron archivos de estado")
        return False

    # Cargar datos inicial, medio y final
    indices = [0, len(state_files)//2, -1]
    labels = ['Inicial', 'Medio', 'Final']
    colors = ['blue', 'green', 'red']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Evolución Temporal: Comparación", fontsize=14)

    for i, (idx, label, color) in enumerate(zip(indices, labels, colors)):
        if idx >= len(state_files):
            continue

        data = load_state_data(state_files[idx])
        r = data['r']

        ax1.plot(r, data['D'], color=color, label=f'{label}', alpha=0.7)
        ax2.plot(r, data['Sr'], color=color, label=f'{label}', alpha=0.7)
        ax3.plot(r, data['lapse'], color=color, label=f'{label}', alpha=0.7)
        ax4.plot(r, data['phi'], color=color, label=f'{label}', alpha=0.7)

    # Configurar plots
    ax1.set_xlabel('r'); ax1.set_ylabel('D'); ax1.set_title('Densidad')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('r'); ax2.set_ylabel('Sr'); ax2.set_title('Momento')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3.set_xlabel('r'); ax3.set_ylabel('α'); ax3.set_title('Lapse')
    ax3.legend(); ax3.grid(True, alpha=0.3)

    ax4.set_xlabel('r'); ax4.set_ylabel('φ'); ax4.set_title('Factor Conformal')
    ax4.legend(); ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Plot resumen guardado: {output_file}")

    return True

if __name__ == "__main__":
    print("🎬 GENERADOR DE VIDEO: Evolución Hidrodinámica")
    print("=" * 60)

    # Verificar directorio de datos
    data_dir = "hydro_evolution_data"
    if not os.path.exists(data_dir):
        print(f"❌ Directorio {data_dir} no encontrado.")
        print("   Primero ejecuta test4.py para generar los datos.")
        exit(1)

    # Crear video principal
    success = create_animation(data_dir, "hydro_evolution.mp4")

    # Crear plot resumen
    create_summary_plot(data_dir, "evolution_summary.png")

    if success:
        print("\n🎉 ¡Video creado exitosamente!")
        print("   Para ver el video:")
        print("   1. hydro_evolution.mp4 - Video completo de evolución")
        print("   2. evolution_summary.png - Comparación inicial/final")
    else:
        print("\n⚠️  Hubo problemas creando el video.")
        print("   Revisa que ffmpeg esté instalado: sudo apt install ffmpeg")