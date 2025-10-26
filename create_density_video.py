#!/usr/bin/env python3
"""
Create video of TOV density evolution from saved snapshots.
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from tqdm import tqdm
import imageio

# Configuration
SNAPSHOT_FILE = "tov_evolution_data2/tov_snapshots_20251025_090507.h5"
OUTPUT_DIR = "tov_plots"
NUM_GHOSTS = 3

# Video configuration
FPS = 5  # Frames per second
FRAME_SKIP = 1  # Use every Nth snapshot (1 means use all available snapshots)
DPI = 150
USE_MP4 = False  # Set to True to create MP4 (requires imageio[ffmpeg]), False for GIF

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_snapshots(h5file, skip=1):
    """Load all snapshots with optional skip interval."""
    snapshots = []
    snapshot_keys = sorted([k for k in h5file['snapshots'].keys() if k.startswith('step_')])

    print(f"Total snapshots available: {len(snapshot_keys)}")
    print(f"Loading every {skip} snapshots...")

    for i, key in enumerate(tqdm(snapshot_keys[::skip], desc="Loading snapshots")):
        grp = h5file['snapshots'][key]
        step = grp.attrs['step']
        t = grp.attrs['time']

        # Get primitives
        rho = grp['primitives']['rho0'][:]
        vr = grp['primitives']['vr'][:]
        p = grp['primitives']['p'][:]

        snapshots.append({
            'step': step,
            't': t,
            'rho': rho,
            'vr': vr,
            'p': p
        })

    print(f"Loaded {len(snapshots)} snapshots for video")
    return snapshots

def create_density_video(snapshots, grid_r, output_file):
    """Create animated video of density evolution using imageio."""

    # Interior points (exclude ghosts)
    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_int = grid_r[interior]

    # Find global min/max for reference
    rho_max_global = max(snap['rho'][interior].max() for snap in snapshots)
    rho_min = 0.0

    print(f"\nGlobal density range: [{rho_min:.6e}, {rho_max_global:.6e}]")

    # Create frames
    frames = []
    print(f"\nGenerating {len(snapshots)} frames...")

    for i, snap in enumerate(tqdm(snapshots, desc="Creating frames")):
        # Create figure for this frame
        fig, ax = plt.subplots(figsize=(12, 7))

        rho = snap['rho'][interior]
        t = snap['t']
        step = snap['step']

        # Get local maximum for this frame
        rho_max_local = rho.max()

        # Plot density
        ax.plot(r_int, rho, 'b-', linewidth=2, label='Baryon Density')

        # Add text annotations
        ax.text(0.02, 0.95, f'Time: {t:.4f}', transform=ax.transAxes,
                fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.text(0.02, 0.88, f'Step: {step:,}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.text(0.02, 0.81, f'Max ρ₀: {rho_max_local:.6e}', transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        # Set axis properties with dynamic y-axis
        ax.set_xlim(r_int[0], r_int[-1])
        ax.set_ylim(rho_min, rho_max_local * 1.1)
        ax.set_xlabel('Radius r', fontsize=14, fontweight='bold')
        ax.set_ylabel(r'Baryon Density $\rho_0$', fontsize=14, fontweight='bold')
        ax.set_title('TOV Star Evolution - Density Profile (Auto-scaled)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=12)
        ax.axhline(0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)

        # Convert plot to image
        fig.canvas.draw()
        # Get the RGBA buffer from the figure
        width, height = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape((height, width, 4))
        # Convert RGBA to RGB
        image = buf[:, :, :3].copy()
        frames.append(image)

        plt.close(fig)

    # Save video or GIF
    print(f"\nSaving animation to: {output_file}")
    if output_file.endswith('.mp4'):
        try:
            imageio.mimsave(output_file, frames, fps=FPS, codec='libx264', quality=8)
        except Exception as e:
            print(f"Error saving MP4: {e}")
            print("Trying to save as GIF instead...")
            output_file = output_file.replace('.mp4', '.gif')
            imageio.mimsave(output_file, frames, duration=1000/FPS, loop=0)
            print(f"Saved as GIF: {output_file}")
    else:
        # Save as GIF
        duration = 1000 / FPS  # milliseconds per frame
        imageio.mimsave(output_file, frames, duration=duration, loop=0)

    print(f"Animation saved successfully!")
    return frames

def create_density_video_with_zoom(snapshots, grid_r, output_file):
    """Create animated video with main plot and zoomed inset using imageio."""

    # Interior points (exclude ghosts)
    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)
    r_int = grid_r[interior]

    # Find global min/max for reference
    rho_max_global = max(snap['rho'][interior].max() for snap in snapshots)
    rho_min = 0.0

    # Find surface radius (approximate)
    r_surface = 10.0  # Approximate stellar surface

    print(f"\nGlobal density range: [{rho_min:.6e}, {rho_max_global:.6e}]")

    # Create frames
    frames = []
    print(f"\nGenerating {len(snapshots)} frames with zoom...")

    for i, snap in enumerate(tqdm(snapshots, desc="Creating frames with zoom")):
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], hspace=0.3)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_zoom = fig.add_subplot(gs[0, 1])

        rho = snap['rho'][interior]
        t = snap['t']
        step = snap['step']

        # Get local maximum for this frame
        rho_max_local = rho.max()

        # Find maximum in zoom region
        zoom_mask = r_int <= r_surface
        rho_zoom = rho[zoom_mask]
        rho_max_zoom = rho_zoom.max() if len(rho_zoom) > 0 else rho_max_local

        # Main plot
        ax_main.plot(r_int, rho, 'b-', linewidth=2.5, label='Baryon Density')
        ax_main.text(0.02, 0.95, f'Time: {t:.4f}', transform=ax_main.transAxes,
                     fontsize=14, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        ax_main.text(0.02, 0.88, f'Step: {step:,}', transform=ax_main.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        ax_main.text(0.02, 0.81, f'Max ρ₀: {rho_max_local:.6e}', transform=ax_main.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        ax_main.set_xlim(r_int[0], r_int[-1])
        ax_main.set_ylim(rho_min, rho_max_local * 1.1)
        ax_main.set_xlabel('Radius r', fontsize=14, fontweight='bold')
        ax_main.set_ylabel(r'Baryon Density $\rho_0$', fontsize=14, fontweight='bold')
        ax_main.set_title('Full Domain (Auto-scaled)', fontsize=14, fontweight='bold')
        ax_main.grid(True, alpha=0.3, linestyle='--')
        ax_main.legend(loc='upper right', fontsize=12)

        # Zoom plot (focus on stellar interior)
        ax_zoom.plot(r_int, rho, 'r-', linewidth=2.5, label='Interior Detail')
        ax_zoom.set_xlim(0, r_surface)
        ax_zoom.set_ylim(rho_min, rho_max_zoom * 1.1)
        ax_zoom.set_xlabel('Radius r', fontsize=12, fontweight='bold')
        ax_zoom.set_ylabel(r'$\rho_0$', fontsize=12, fontweight='bold')
        ax_zoom.set_title('Stellar Interior - Zoom (Auto-scaled)', fontsize=14, fontweight='bold')
        ax_zoom.grid(True, alpha=0.3, linestyle='--')
        ax_zoom.legend(loc='upper right', fontsize=10)

        plt.suptitle('TOV Star Evolution - Baryon Density', fontsize=18, fontweight='bold', y=0.98)

        # Convert plot to image
        fig.canvas.draw()
        # Get the RGBA buffer from the figure
        width, height = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape((height, width, 4))
        # Convert RGBA to RGB
        image = buf[:, :, :3].copy()
        frames.append(image)

        plt.close(fig)

    # Save video or GIF
    print(f"\nSaving animation to: {output_file}")
    if output_file.endswith('.mp4'):
        try:
            imageio.mimsave(output_file, frames, fps=FPS, codec='libx264', quality=8)
        except Exception as e:
            print(f"Error saving MP4: {e}")
            print("Trying to save as GIF instead...")
            output_file = output_file.replace('.mp4', '.gif')
            imageio.mimsave(output_file, frames, duration=1000/FPS, loop=0)
            print(f"Saved as GIF: {output_file}")
    else:
        # Save as GIF
        duration = 1000 / FPS  # milliseconds per frame
        imageio.mimsave(output_file, frames, duration=duration, loop=0)

    print(f"Animation saved successfully!")
    return frames

def main():
    """Main execution."""
    print("="*70)
    print("TOV Density Evolution - Video Creator")
    print("="*70)
    print(f"Snapshot file: {SNAPSHOT_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Frame skip: {FRAME_SKIP} (using every {FRAME_SKIP}th snapshot)")
    print(f"FPS: {FPS}")
    print("="*70)

    # Check if snapshot file exists
    if not os.path.exists(SNAPSHOT_FILE):
        print(f"\nERROR: Snapshot file not found: {SNAPSHOT_FILE}")
        print("Please check the file path and try again.")
        return

    # Open HDF5 file and load data
    with h5py.File(SNAPSHOT_FILE, 'r') as f:
        # Load grid
        grid_r = f['grid']['r'][:]
        print(f"\nGrid points: {len(grid_r)}")
        print(f"r range: [{grid_r[0]:.4f}, {grid_r[-1]:.4f}]")

        # Load snapshots
        snapshots = load_all_snapshots(f, skip=FRAME_SKIP)

    if len(snapshots) == 0:
        print("\nERROR: No snapshots loaded!")
        return

    # Create videos or GIFs
    print("\n" + "="*70)
    if USE_MP4:
        print("Creating Videos (MP4 format)")
        ext = ".mp4"
    else:
        print("Creating Animations (GIF format)")
        ext = ".gif"
    print("="*70)

    # Simple video/animation
    output_file_simple = os.path.join(OUTPUT_DIR, f"tov_density_simple{ext}")
    print(f"\n1. Creating simple density animation...")
    create_density_video(snapshots, grid_r, output_file_simple)

    # Video/animation with zoom
    output_file_zoom = os.path.join(OUTPUT_DIR, f"tov_density_zoom{ext}")
    print(f"\n2. Creating density animation with zoom...")
    create_density_video_with_zoom(snapshots, grid_r, output_file_zoom)

    print("\n" + "="*70)
    print("Done!")
    print("="*70)
    print(f"\nAnimations created:")
    print(f"  1. {output_file_simple}")
    print(f"  2. {output_file_zoom}")

    # Print animation info
    duration = len(snapshots) / FPS
    print(f"\nAnimation duration: {duration:.1f} seconds")
    print(f"Total frames: {len(snapshots)}")
    print(f"Time range: t=0 to t={snapshots[-1]['t']:.4f}")
    print(f"Step range: {snapshots[0]['step']} to {snapshots[-1]['step']}")

    if not USE_MP4:
        print(f"\nNote: Animations saved as GIF format.")
        print(f"To create MP4 videos instead, install ffmpeg:")
        print(f"  conda install -c conda-forge imageio-ffmpeg")
        print(f"  or: pip install imageio[ffmpeg]")
        print(f"Then set USE_MP4=True in the script.")

if __name__ == "__main__":
    main()
