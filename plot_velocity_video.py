#!/usr/bin/env python3
"""
Generate TOV velocity evolution video.
Creates PNG frames with dynamic zoom and MP4 video.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import os
import subprocess
import argparse
from matplotlib.animation import FuncAnimation, PillowWriter


def load_snapshot_data(snapshot_file):
    """Load all snapshots from HDF5 file."""
    with h5py.File(snapshot_file, 'r') as f:
        r = f['grid/r'][:]
        snap_keys = sorted(f['snapshots'].keys(),
                          key=lambda x: int(x.split('_')[1]))

        n_snaps = len(snap_keys)
        n_points = len(r)

        times = np.zeros(n_snaps)
        velocities = np.zeros((n_snaps, n_points))
        densities = np.zeros((n_snaps, n_points))
        pressures = np.zeros((n_snaps, n_points))

        for i, key in enumerate(snap_keys):
            snap = f['snapshots'][key]
            times[i] = snap.attrs['time']
            velocities[i, :] = snap['primitives/vr'][:]
            densities[i, :] = snap['primitives/rho0'][:]
            pressures[i, :] = snap['primitives/p'][:]

        return r, times, velocities, densities, pressures


def create_diagnostic_plots(snapshot_file, output_dir='.'):
    """
    Generate diagnostic plots for velocity analysis.
    """
    print(f"\nGenerating diagnostic plots...")
    r, times, velocities, densities, pressures = load_snapshot_data(snapshot_file)

    NUM_GHOSTS = 3

    # Plot 1: Velocity propagation analysis
    threshold = 1e-6
    r_max_v = []
    t_valid = []

    for i, t in enumerate(times):
        v_abs = np.abs(velocities[i])
        if np.max(v_abs) > threshold:
            idx_max = np.argmax(v_abs)
            r_max_v.append(r[idx_max])
            t_valid.append(t)

    if len(r_max_v) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_valid, r_max_v, 'bo-', linewidth=2, markersize=4, label='Position of max |v^r|')
        ax.set_xlabel('Time [M]', fontsize=12)
        ax.set_ylabel('Radius [M]', fontsize=12)
        ax.set_title('Position of Maximum Velocity vs Time', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'velocity_propagation.png'), dpi=150)
        plt.close()
        print("  ✓ velocity_propagation.png")

    # Plot 2: Velocity magnitude evolution and spacetime
    v_max_time = np.max(np.abs(velocities), axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Growth over time
    ax1.semilogy(times, v_max_time, 'b-', linewidth=2, label='max |v^r|')
    ax1.set_xlabel('Time [M]', fontsize=12)
    ax1.set_ylabel('Max |v^r| [c]', fontsize=12)
    ax1.set_title('Velocity Magnitude Evolution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Spacetime diagram (zoomed on star)
    i_plot_start = NUM_GHOSTS
    i_plot_end = min(len(r), NUM_GHOSTS + 200)
    r_plot = r[i_plot_start:i_plot_end]
    v_plot = velocities[:, i_plot_start:i_plot_end]

    T, R = np.meshgrid(times, r_plot, indexing='ij')
    v_max_plot = np.max(np.abs(v_plot))
    im = ax2.pcolormesh(R, T, v_plot, shading='auto', cmap='RdBu_r',
                        vmin=-v_max_plot, vmax=v_max_plot)
    ax2.set_xlabel('Radius [M]', fontsize=12)
    ax2.set_ylabel('Time [M]', fontsize=12)
    ax2.set_title('Velocity Spacetime Diagram', fontsize=14)
    plt.colorbar(im, ax=ax2, label='v^r [c]')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_magnitude.png'), dpi=150)
    plt.close()
    print("  ✓ velocity_magnitude.png")

    # Plot 3: Velocity at different radii
    # Find surface
    rho_first = densities[0]
    rho_grad = np.abs(np.gradient(rho_first, r))
    i_surface = np.argmax(rho_grad[NUM_GHOSTS:]) + NUM_GHOSTS

    # Sample points
    i_center = NUM_GHOSTS + 5
    i_mid = (i_center + i_surface) // 2
    i_surf = i_surface
    i_ext = min(i_surface + 50, len(r) - NUM_GHOSTS - 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, velocities[:, i_center], 'b-', linewidth=2, label=f'r = {r[i_center]:.3f} M (center)')
    ax.plot(times, velocities[:, i_mid], 'g-', linewidth=2, label=f'r = {r[i_mid]:.3f} M (interior)')
    ax.plot(times, velocities[:, i_surf], 'r-', linewidth=2, label=f'r = {r[i_surf]:.3f} M (surface)')
    ax.plot(times, velocities[:, i_ext], 'm-', linewidth=2, label=f'r = {r[i_ext]:.3f} M (exterior)')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time [M]', fontsize=12)
    ax.set_ylabel('v^r [c]', fontsize=12)
    ax.set_title('Velocity at Different Radii', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_radii.png'), dpi=150)
    plt.close()
    print("  ✓ velocity_radii.png")


def create_velocity_frames(snapshot_file, frame_dir='tov_velocity_frames'):
    """
    Generate individual velocity frame images with dynamic zoom.

    Returns:
        n_snaps: Number of frames generated
    """
    print(f"\nGenerating velocity frames with dynamic zoom...")
    r, times, velocities, densities, pressures = load_snapshot_data(snapshot_file)

    n_snaps = len(times)
    print(f"Found {n_snaps} snapshots, time range: {times[0]:.3f} - {times[-1]:.3f} M")

    # Create frame directory
    os.makedirs(frame_dir, exist_ok=True)

    for i in range(n_snaps):
        # Create figure for this frame
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))

        # Plot velocity
        ax.semilogy(r, velocities[i], 'b-', linewidth=2.5, label='$v^r$')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_xlabel('Radius r [M]', fontsize=14)
        ax.set_ylabel('Radial Velocity $v^r$ [c]', fontsize=14)
        ax.set_xlim(r[0], r[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='upper right')

        # Compute velocity statistics
        v_max_frame = np.max(np.abs(velocities[i]))
        v_min_frame = np.min(velocities[i])
        v_max_pos_frame = np.max(velocities[i])

        # Dynamic zoom: vmax reaches near plot limits (5% margin)
        if v_max_frame > 1e-15:
            margin = 1.05
            v_lim_frame = v_max_frame * margin
            ax.set_ylim(-v_lim_frame, v_lim_frame)

        # Add text box for velocity scale
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        info_text = f'v_max = {v_max_frame:.3e}\nv_min = {v_min_frame:.3e}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=props)

        # Title
        ax.set_title(f'Radial Velocity - t = {times[i]:.3f} M (frame {i+1}/{n_snaps})',
                     fontsize=16, weight='bold')

        plt.tight_layout()

        # Save frame
        frame_file = os.path.join(frame_dir, f'velocity_frame_{i:04d}.png')
        plt.savefig(frame_file, dpi=100, bbox_inches='tight')
        plt.close()

        # Print progress
        if i % 5 == 0 or i == n_snaps - 1:
            print(f"  Frame {i+1}/{n_snaps}: t={times[i]:.3f}, "
                  f"v_range=[{v_min_frame:.3e}, {v_max_pos_frame:.3e}]")

    print(f"✓ {n_snaps} frames saved to {frame_dir}/")
    return n_snaps


def create_video_matplotlib(snapshot_file, output_file='tov_velocity.gif', fps=10):
    """
    Create animated GIF using matplotlib (fallback when ffmpeg not available).

    Args:
        snapshot_file: HDF5 file with snapshots
        output_file: Output GIF filename
        fps: Frames per second
    """
    print(f"\nCreating GIF using matplotlib (ffmpeg not available)...")
    r, times, velocities, densities, pressures = load_snapshot_data(snapshot_file)

    n_snaps = len(times)

    # Setup figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    # Initialize plot
    line_v, = ax.plot([], [], 'b-', linewidth=2.5, label='$v^r$')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Radius r [M]', fontsize=14)
    ax.set_ylabel('Radial Velocity $v^r$ [c]', fontsize=14)
    ax.set_xlim(r[0], r[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')

    # Add text box for velocity scale
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    text_box = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', bbox=props)

    title = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                     ha='center', fontsize=16, weight='bold')

    plt.tight_layout()

    def init():
        line_v.set_data([], [])
        title.set_text('')
        text_box.set_text('')
        return line_v, title, text_box

    def animate(i):
        line_v.set_data(r, velocities[i])
        title.set_text(f'Radial Velocity - t = {times[i]:.3f} M (frame {i+1}/{n_snaps})')

        v_max_frame = np.max(np.abs(velocities[i]))
        v_min_frame = np.min(velocities[i])

        # Dynamic zoom: vmax reaches near plot limits
        if v_max_frame > 1e-15:
            margin = 1.05
            v_lim_frame = v_max_frame * margin
            ax.set_ylim(-v_lim_frame, v_lim_frame)

        info_text = f'v_max = {v_max_frame:.3e}\nv_min = {v_min_frame:.3e}'
        text_box.set_text(info_text)

        return line_v, title, text_box

    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=n_snaps, interval=1000/fps,
                        blit=True, repeat=True)

    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer, dpi=100)

    print(f"✓ GIF created: {output_file}")
    plt.close()
    return True


def create_mp4_from_frames(frame_dir='tov_velocity_frames',
                           output_file='tov_velocity.mp4',
                           fps=10):
    """
    Create MP4 video from PNG frames using ffmpeg.

    Args:
        frame_dir: Directory containing frames
        output_file: Output MP4 filename
        fps: Frames per second
    """
    print(f"\nCreating MP4 video...")

    frame_pattern = os.path.join(frame_dir, 'velocity_frame_%04d.png')

    # ffmpeg command
    cmd = [
        'ffmpeg', '-y',  # Overwrite output
        '-framerate', str(fps),
        '-i', frame_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',  # Quality (lower = better, 18 = visually lossless)
        '-preset', 'slow',  # Encoding speed (slower = better compression)
        output_file
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ MP4 video created: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ FFmpeg failed:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"✗ ffmpeg not found.")
        print(f"   Install with: sudo apt install ffmpeg")
        print(f"   Falling back to GIF generation...")
        return False


def convert_gif_to_mp4(gif_file, mp4_file=None, fps=None):
    """
    Convert GIF to MP4 format.

    First tries using ffmpeg (fast and high quality).
    Falls back to Python libraries if ffmpeg is not available.

    Args:
        gif_file: Input GIF filename
        mp4_file: Output MP4 filename (defaults to same name with .mp4 extension)
        fps: Frames per second (auto-detects from GIF if None)

    Returns:
        bool: True if successful, False otherwise
    """
    if mp4_file is None:
        mp4_file = gif_file.rsplit('.', 1)[0] + '.mp4'

    print(f"\nConverting {gif_file} to {mp4_file}...")

    # Method 1: Try ffmpeg first (fastest and best quality)
    cmd = [
        'ffmpeg', '-y',  # Overwrite output
        '-i', gif_file,
        '-movflags', 'faststart',  # Optimize for streaming
        '-pix_fmt', 'yuv420p',  # Compatible pixel format
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure even dimensions
        mp4_file
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ MP4 created using ffmpeg: {mp4_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ FFmpeg failed: {e.stderr}")
    except FileNotFoundError:
        print("✗ ffmpeg not found, trying Python libraries...")

    # Method 2: Fallback to Python libraries
    try:
        from PIL import Image
        import imageio

        # Read the GIF
        gif = Image.open(gif_file)

        # Extract all frames
        frames = []
        try:
            while True:
                frame = gif.copy()
                frames.append(np.array(frame.convert('RGB')))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        # Get FPS from GIF or use provided value
        if fps is None:
            duration_ms = gif.info.get('duration', 100)
            fps = 1000.0 / duration_ms if duration_ms > 0 else 10

        print(f"  Converting {len(frames)} frames at {fps:.2f} fps...")

        # Write as MP4
        writer = imageio.get_writer(mp4_file, fps=fps, codec='libx264',
                                    pixelformat='yuv420p')
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        print(f"✓ MP4 created using Python libraries: {mp4_file}")
        return True

    except ImportError as e:
        print(f"✗ Required Python libraries not available: {e}")
        print("  Install with: pip install pillow imageio imageio-ffmpeg")
        return False
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate TOV velocity evolution video with dynamic zoom'
    )
    parser.add_argument('--snapshot-file', '-f',
                       default=None,
                       help='HDF5 snapshot file (auto-detects latest if not specified)')
    parser.add_argument('--output', '-o',
                       default='tov_velocity.mp4',
                       help='Output MP4 filename (default: tov_velocity.mp4)')
    parser.add_argument('--frames-dir', '-d',
                       default='tov_velocity_frames',
                       help='Directory for frame output (default: tov_velocity_frames)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for video (default: 10)')
    parser.add_argument('--no-diagnostics', action='store_true',
                       help='Skip diagnostic plots generation')
    parser.add_argument('--clean-frames', action='store_true',
                       help='Delete PNG frames after creating video')
    parser.add_argument('--convert-gif', '-g',
                       default=None,
                       help='Convert existing GIF file to MP4 (skips video generation)')

    args = parser.parse_args()

    # If convert-gif mode, just do the conversion and exit
    if args.convert_gif:
        print("="*60)
        print("GIF to MP4 Converter")
        print("="*60)
        if not os.path.exists(args.convert_gif):
            print(f"Error: File not found: {args.convert_gif}")
            return
        success = convert_gif_to_mp4(args.convert_gif, fps=args.fps)
        if success:
            mp4_file = args.convert_gif.rsplit('.', 1)[0] + '.mp4'
            print("="*60)
            print(f"✓ Conversion complete: {mp4_file}")
            print("="*60)
        return

    print("="*60)
    print("TOV Velocity Evolution Video Generator")
    print("="*60)

    # Auto-detect snapshot file if not specified
    if args.snapshot_file is None:
        files = sorted(glob.glob('tov_evolution_data2/tov_snapshots_*.h5'))
        if not files:
            print("Error: No snapshot files found in tov_evolution_data2/")
            return
        args.snapshot_file = files[-1]
        print(f"Auto-detected: {args.snapshot_file}")

    if not os.path.exists(args.snapshot_file):
        print(f"Error: File not found: {args.snapshot_file}")
        return

    # Generate diagnostic plots
    if not args.no_diagnostics:
        create_diagnostic_plots(args.snapshot_file, output_dir='.')

    # Generate frames
    n_frames = create_velocity_frames(args.snapshot_file, args.frames_dir)

    # Try to create MP4
    success = create_mp4_from_frames(args.frames_dir, args.output, args.fps)

    # If ffmpeg not available, create GIF as fallback
    if not success:
        gif_output = args.output.replace('.mp4', '.gif')
        success = create_video_matplotlib(args.snapshot_file, gif_output, args.fps)
        actual_output = gif_output
    else:
        actual_output = args.output

    # Clean up frames if requested and video was created successfully
    if success and args.clean_frames:
        print(f"\nCleaning up frames...")
        import shutil
        shutil.rmtree(args.frames_dir)
        print(f"✓ Removed {args.frames_dir}/")

    print("\n" + "="*60)
    print("Done! Output:")
    if success:
        print(f"  - {actual_output}")

    # Show frames unless they were cleaned up
    if not args.clean_frames:
        print(f"  - {args.frames_dir}/ ({n_frames} PNG frames)")

    if not args.no_diagnostics:
        print("\nDiagnostic plots:")
        print(f"  - velocity_propagation.png")
        print(f"  - velocity_magnitude.png")
        print(f"  - velocity_radii.png")
    print("="*60)


if __name__ == '__main__':
    main()
