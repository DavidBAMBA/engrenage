#!/usr/bin/env python3
"""
Master automation script for TOV simulation analysis.

Discovers all tov_evolution_data* directories, identifies available resolutions,
and runs appropriate analysis scripts with error handling.
"""

import os
import sys
import subprocess
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Configuration: Scripts and their requirements
ANALYSIS_SCRIPTS = {
    'plot_qnm_analysis.py': {
        'min_res': 1,
        'max_res': None,
        'needs_tov_cache': False,
        'description': 'QNM frequency analysis',
    },
    'rho_central.py': {
        'min_res': 1,
        'max_res': None,
        'needs_tov_cache': False,
        'description': 'Central density evolution',
    },
    'plot_final_profiles.py': {
        'min_res': 1,
        'max_res': None,
        'needs_tov_cache': False,
        'description': 'Final profiles comparison',
    },
    'plot_convergence.py': {
        'min_res': 3,
        'max_res': 3,
        'needs_tov_cache': False,
        'description': 'Convergence analysis with L1 errors',
    },
    'baryon_mass_convergence.py': {
        'min_res': 3,
        'max_res': 3,
        'needs_tov_cache': False,
        'description': 'Baryon mass convergence',
    },
    'convergence_test.py': {
        'min_res': 3,
        'max_res': 3,
        'needs_tov_cache': False,
        'description': 'Q-factor convergence test',
    },
    'rest_mass_density_convergence.py': {
        'min_res': 4,
        'max_res': 4,
        'needs_tov_cache': True,
        'description': 'Density convergence (4 methods)',
    },
}


def find_parent_directories(base_dir: str) -> List[str]:
    """Find all tov_evolution_data* directories."""
    pattern = re.compile(r'^tov_evolution_data')
    parent_dirs = []

    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and pattern.match(entry):
            parent_dirs.append(full_path)

    return sorted(parent_dirs)


def extract_resolution_from_dirname(dirname: str) -> Optional[int]:
    """Extract resolution number from directory name.

    Pattern: tov_star_rhoc1p28em03_N{RES}_K100_G2_cow_mp5
    Regex: [Nn]r?[=_]?(\\d+)
    """
    match = re.search(r'[Nn]r?[=_]?(\d+)', dirname)
    if match:
        return int(match.group(1))
    return None


def validate_resolution_directory(res_dir: str) -> bool:
    """Check if directory contains required files."""
    required_files = [
        'tov_evolution_cow.h5',
        'tov_snapshots_cow.h5',
    ]
    for filename in required_files:
        if not os.path.exists(os.path.join(res_dir, filename)):
            return False
    return True


def find_resolution_directories(parent_dir: str) -> List[Tuple[int, str]]:
    """Find all valid resolution subdirectories.

    Returns: List of (resolution_number, full_path), sorted by resolution
    """
    res_dirs = []

    for entry in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, entry)
        if not os.path.isdir(full_path):
            continue

        # Extract resolution number
        res_num = extract_resolution_from_dirname(entry)
        if res_num is None:
            continue

        # Validate required files exist
        if validate_resolution_directory(full_path):
            res_dirs.append((res_num, full_path))

    return sorted(res_dirs, key=lambda x: x[0])


def find_tov_cache(base_dir: str) -> Optional[str]:
    """Search for TOV analytical solution cache."""
    cache_base = os.path.join(base_dir, 'tov_iso_cache')
    if not os.path.exists(cache_base):
        return None

    # Look for directories starting with TOVSOL_ISO_
    for entry in os.listdir(cache_base):
        entry_path = os.path.join(cache_base, entry)
        if os.path.isdir(entry_path) and entry.startswith('TOVSOL_ISO_'):
            # Verify it has required files
            if os.path.exists(os.path.join(entry_path, 'r_iso.npy')):
                return entry_path

    return None


def select_resolutions_for_script(
    available: List[Tuple[int, str]],
    min_res: int,
    max_res: Optional[int]
) -> Optional[List[str]]:
    """Select appropriate resolutions for a script.

    Strategy for exact-N requirements:
    - N=3: Select [min, median, max] when 5+ available
           Select [0, 1, 3] when 4 available
           Select all when exactly 3 available
    - N=4: Select [min, 1/3, 2/3, max] when 5+ available
           Select all when exactly 4 available

    Returns: List of directory paths, or None if insufficient resolutions
    """
    n_available = len(available)

    # For scripts that accept any number
    if max_res is None:
        return [path for _, path in available]

    # For scripts requiring exactly N resolutions
    if n_available < min_res:
        return None  # Insufficient resolutions

    if n_available == max_res:
        return [path for _, path in available]

    # Need to select subset
    if max_res == 3:
        if n_available == 4:
            # Use indices [0, 1, 3] - skip one middle resolution
            return [available[0][1], available[1][1], available[3][1]]
        else:  # 5+
            # Evenly spaced
            mid = n_available // 2
            return [available[0][1], available[mid][1], available[-1][1]]

    elif max_res == 4:
        if n_available == 5:
            # Use indices [0, 1, 3, 4]
            return [available[0][1], available[1][1], available[3][1], available[4][1]]
        else:  # 6+
            # Evenly spaced across range
            indices = [0, n_available//3, 2*n_available//3, -1]
            return [available[i][1] for i in indices]

    return None


def run_analysis_script(
    script_path: str,
    data_dirs: List[str],
    output_dir: str,
    tov_cache: Optional[str] = None,
    verbose: bool = False
) -> Tuple[bool, Optional[str]]:
    """Run a single analysis script.

    Returns: (success, error_message)
    """
    script_name = os.path.basename(script_path)

    # Build command
    cmd = [sys.executable, script_path]
    cmd.extend(['--data-dirs'] + data_dirs)
    cmd.extend(['--output-dir', output_dir])

    # Add TOV cache if needed
    config = ANALYSIS_SCRIPTS.get(script_name, {})
    if config.get('needs_tov_cache') and tov_cache:
        cmd.extend(['--tov-cache', tov_cache])

    try:
        if verbose:
            print(f"    Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        return (True, None)

    except subprocess.CalledProcessError as e:
        error_msg = f"Script failed with exit code {e.returncode}"
        if e.stderr:
            error_msg += f"\n{e.stderr}"
        return (False, error_msg)
    except Exception as e:
        return (False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description='Automate TOV simulation analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  python run_all_analysis.py                    # Analyze all
  python run_all_analysis.py --dry-run          # Preview
  python run_all_analysis.py --parent-dir NAME  # Specific parent
  python run_all_analysis.py --scripts A,B,C    # Select scripts
'''
    )

    parser.add_argument('--base-dir', default='.',
                       help='Base directory (default: current)')
    parser.add_argument('--parent-dir', default=None,
                       help='Analyze specific parent directory')
    parser.add_argument('--scripts', default=None,
                       help='Comma-separated list of scripts')
    parser.add_argument('--tov-cache', default=None,
                       help='Path to TOV cache')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview without running')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue if script fails')

    args = parser.parse_args()

    # Find parent directories
    if args.parent_dir:
        if os.path.isabs(args.parent_dir):
            parent_dirs = [args.parent_dir]
        else:
            parent_dirs = [os.path.join(args.base_dir, args.parent_dir)]
    else:
        parent_dirs = find_parent_directories(args.base_dir)

    if not parent_dirs:
        print("No tov_evolution_data* directories found")
        return 1

    # Find TOV cache
    tov_cache = args.tov_cache or find_tov_cache(args.base_dir)
    if tov_cache:
        print(f"Found TOV cache: {tov_cache}")
    else:
        print("Warning: No TOV cache found. Scripts requiring it will be skipped.")

    # Select scripts to run
    if args.scripts:
        scripts_to_run = [s.strip() for s in args.scripts.split(',')]
    else:
        scripts_to_run = list(ANALYSIS_SCRIPTS.keys())

    # Get analysis directory
    analysis_dir = os.path.join(args.base_dir, 'analysis')
    if not os.path.exists(analysis_dir):
        print(f"Error: Analysis directory not found: {analysis_dir}")
        return 1

    # Statistics
    total_runs = 0
    successful_runs = 0
    failed_runs = 0
    skipped_runs = 0

    # Process each parent directory
    for parent_dir in parent_dirs:
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(parent_dir)}")
        print(f"{'='*70}")

        # Find resolutions
        res_dirs = find_resolution_directories(parent_dir)
        res_nums = [n for n, _ in res_dirs]
        print(f"Found {len(res_dirs)} resolutions: {res_nums}")

        if len(res_dirs) == 0:
            print("  No valid resolution directories found. Skipping.")
            continue

        # Create output directory
        output_dir = os.path.join(parent_dir, 'plots')
        if not args.dry_run:
            os.makedirs(output_dir, exist_ok=True)

        # Run each script
        for script_name in scripts_to_run:
            if script_name not in ANALYSIS_SCRIPTS:
                print(f"\n  Warning: Unknown script '{script_name}', skipping")
                continue

            config = ANALYSIS_SCRIPTS[script_name]
            script_path = os.path.join(analysis_dir, script_name)

            if not os.path.exists(script_path):
                print(f"\n  Warning: Script not found: {script_path}")
                skipped_runs += 1
                continue

            print(f"\n  Script: {script_name}")
            print(f"    {config['description']}")

            # Check if script needs TOV cache
            if config.get('needs_tov_cache') and not tov_cache:
                print("    ⚠️  SKIPPED: TOV cache not found")
                skipped_runs += 1
                continue

            # Select resolutions
            selected_dirs = select_resolutions_for_script(
                res_dirs, config['min_res'], config['max_res']
            )

            if selected_dirs is None:
                print(f"    ⚠️  SKIPPED: Need {config['min_res']} resolutions, only {len(res_dirs)} available")
                skipped_runs += 1
                continue

            selected_res = [extract_resolution_from_dirname(os.path.basename(d))
                          for d in selected_dirs]
            print(f"    Using resolutions: {selected_res}")

            if args.dry_run:
                print("    [DRY RUN] Would execute")
                total_runs += 1
                continue

            # Run the script
            total_runs += 1
            success, error = run_analysis_script(
                script_path, selected_dirs, output_dir, tov_cache, args.verbose
            )

            if success:
                print("    ✓ SUCCESS")
                successful_runs += 1
            else:
                print(f"    ✗ FAILED: {error}")
                failed_runs += 1
                if not args.continue_on_error:
                    return 1

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Parent directories processed: {len(parent_dirs)}")
    if not args.dry_run:
        print(f"Total analyses run: {total_runs}")
        print(f"  Successful: {successful_runs}")
        print(f"  Failed: {failed_runs}")
        print(f"  Skipped: {skipped_runs}")
    else:
        print("DRY RUN - no scripts were executed")
        print(f"Would have run: {total_runs} analyses")
        print(f"Would have skipped: {skipped_runs} analyses")

    return 0 if failed_runs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
