#!/usr/bin/env python3
"""
Batch comparison script for multiple GVHMR predictions against a reference.

This script finds all hmr4d_results.pt files in subdirectories and compares them
against a reference GVHMR file using the compare_two_hmr_mpjpe_timealigned.py script.

Usage:
    python batch_compare_hmr.py --song Old_Town_Road --person Wontaek
    python batch_compare_hmr.py --song Unstoppable --person Wontaek
    
Or with custom paths:
    python batch_compare_hmr.py --gvhmr_dir /path/to/dir --reference /path/to/ref.pt
"""

import argparse
import pathlib
import subprocess
import sys
from glob import glob

HERE = pathlib.Path(__file__).parent
COMPARE_SCRIPT = HERE / "compare_two_hmr_mpjpe_timealigned.py"

# Default base directories
DEFAULT_HUMAN_BASE = "/home/jkim3662/Videos/Switch4EAI/HumanRecordings_GVHMR"
DEFAULT_REFERENCE_BASE = "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GVHMR"


def find_gvhmr_files(directory):
    """
    Find all hmr4d_results.pt files in subdirectories.
    
    Args:
        directory: Path to search in
        
    Returns:
        List of Path objects to hmr4d_results.pt files
    """
    search_pattern = str(pathlib.Path(directory) / "*" / "hmr4d_results.pt")
    files = [pathlib.Path(f) for f in glob(search_pattern)]
    return sorted(files)


def run_comparison(gvhmr_1, gvhmr_2, python_cmd="python"):
    """
    Run the comparison script for a single pair of files.
    
    Args:
        gvhmr_1: Path to first GVHMR file
        gvhmr_2: Path to reference GVHMR file
        python_cmd: Python command to use
        
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        python_cmd,
        str(COMPARE_SCRIPT),
        "--gvhmr_1", str(gvhmr_1),
        "--gvhmr_2", str(gvhmr_2)
    ]
    
    print(f"\n{'='*80}")
    print(f"Running comparison:")
    print(f"  GVHMR 1: {gvhmr_1}")
    print(f"  GVHMR 2: {gvhmr_2}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Comparison completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Comparison failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch compare multiple GVHMR predictions against a reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all recordings for a specific song and person
  python batch_compare_hmr.py --song Old_Town_Road --person Wontaek
  
  # Compare with custom paths
  python batch_compare_hmr.py --gvhmr_dir /path/to/recordings --reference /path/to/ref.pt
  
  # Use specific Python interpreter (e.g., in conda environment)
  python batch_compare_hmr.py --song Unstoppable --person Wontaek --python "/home/user/miniconda3/bin/python"
        """
    )
    
    # Song-based arguments
    parser.add_argument(
        "--song",
        type=str,
        help="Song name (e.g., 'Old_Town_Road', 'Unstoppable'). Will use default directory structure."
    )
    parser.add_argument(
        "--person",
        type=str,
        help="Person name for human recordings (e.g., 'Wontaek'). Required if --song is used."
    )
    
    # Custom path arguments
    parser.add_argument(
        "--gvhmr_dir",
        type=str,
        help="Directory containing GVHMR files to compare (searches for */hmr4d_results.pt)"
    )
    parser.add_argument(
        "--reference",
        type=str,
        help="Path to reference GVHMR file"
    )
    
    # Other options
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use (default: current Python)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running comparisons"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.song or args.person:
        if not (args.song and args.person):
            parser.error("Both --song and --person are required when using song-based mode")
        
        # Construct paths based on song and person
        gvhmr_dir = f"{DEFAULT_HUMAN_BASE}/{args.person}/{args.song}_cut"
        reference = f"{DEFAULT_REFERENCE_BASE}/{args.song}_cut/hmr4d_results.pt"
        
    elif args.gvhmr_dir and args.reference:
        gvhmr_dir = args.gvhmr_dir
        reference = args.reference
        
    else:
        parser.error("Either use --song and --person, OR use --gvhmr_dir and --reference")
    
    # Convert to Path objects
    gvhmr_dir = pathlib.Path(gvhmr_dir)
    reference = pathlib.Path(reference)
    
    # Check if paths exist
    if not gvhmr_dir.exists():
        print(f"Error: GVHMR directory does not exist: {gvhmr_dir}")
        sys.exit(1)
    
    if not reference.exists():
        print(f"Error: Reference file does not exist: {reference}")
        sys.exit(1)
    
    # Find all GVHMR files
    gvhmr_files = find_gvhmr_files(gvhmr_dir)
    
    if not gvhmr_files:
        print(f"Error: No hmr4d_results.pt files found in {gvhmr_dir}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Batch Comparison Configuration")
    print(f"{'='*80}")
    print(f"GVHMR directory: {gvhmr_dir}")
    print(f"Reference file: {reference}")
    print(f"Found {len(gvhmr_files)} GVHMR file(s) to compare")
    print(f"Python interpreter: {args.python}")
    print(f"{'='*80}")
    
    # List all files to compare
    print(f"\nFiles to compare:")
    for i, gvhmr_file in enumerate(gvhmr_files, 1):
        relative_path = gvhmr_file.relative_to(gvhmr_dir)
        print(f"  {i}. {relative_path.parent}/")
    
    if args.dry_run:
        print("\n[DRY RUN] - No comparisons will be executed")
        sys.exit(0)
    
    # Run comparisons
    print(f"\n{'='*80}")
    print(f"Starting batch comparisons...")
    print(f"{'='*80}")
    
    successful = 0
    failed = 0
    
    for i, gvhmr_file in enumerate(gvhmr_files, 1):
        print(f"\n[{i}/{len(gvhmr_files)}] Processing: {gvhmr_file.parent.name}")
        
        success = run_comparison(gvhmr_file, reference, args.python)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Batch Comparison Summary")
    print(f"{'='*80}")
    print(f"Total files: {len(gvhmr_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*80}")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All comparisons completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
