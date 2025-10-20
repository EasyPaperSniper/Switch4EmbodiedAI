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
DEFAULT_HUMAN_BASE = "/home/jkim3662/Videos/Switch4EAI/HumanRecordings_GVHMR/raw/"
DEFAULT_REFERENCE_BASE = "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GVHMR/cut_mirrored"


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


def run_comparison(gvhmr_1, gvhmr_2, python_cmd="python", csv_output=None, cut_videos=False, video_names=None, compute_dtw=True):
    """
    Run the comparison script for a single pair of files.
    
    Args:
        gvhmr_1: Path to first GVHMR file
        gvhmr_2: Path to reference GVHMR file
        python_cmd: Python command to use
        csv_output: Path to CSV file for metrics (optional)
        cut_videos: Whether to cut videos based on alignment
        video_names: List of video filenames to cut (optional)
        compute_dtw: Whether to compute DTW alignment (default: True)
        
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        python_cmd,
        str(COMPARE_SCRIPT),
        "--gvhmr_1", str(gvhmr_1),
        "--gvhmr_2", str(gvhmr_2)
    ]
    
    if csv_output:
        cmd.extend(["--csv_output", str(csv_output)])
    
    if cut_videos:
        cmd.append("--cut_videos")
        
    if video_names:
        cmd.append("--video_names")
        cmd.extend(video_names)
    
    if not compute_dtw:
        cmd.append("--no_dtw")
    
    print(f"\n{'='*80}")
    print(f"Running comparison:")
    print(f"  GVHMR 1: {gvhmr_1}")
    print(f"  GVHMR 2: {gvhmr_2}")
    if csv_output:
        print(f"  CSV Output: {csv_output}")
    if cut_videos:
        print(f"  Video cutting: Enabled")
    print(f"  DTW computation: {'Enabled' if compute_dtw else 'Disabled'}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Comparison completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Comparison failed with error: {e}")
        return False


def process_song(gvhmr_dir, reference, args):
    """
    Process all GVHMR files in a directory against a reference.
    
    Args:
        gvhmr_dir: Path to directory containing GVHMR files
        reference: Path to reference GVHMR file
        args: Command line arguments
        
    Returns:
        Tuple of (successful_count, failed_count, failed_files_list)
    """
    # Find all GVHMR files
    gvhmr_files = find_gvhmr_files(gvhmr_dir)
    
    if not gvhmr_files:
        print(f"Error: No hmr4d_results.pt files found in {gvhmr_dir}")
        return 0, 0, []
    
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
        return 0, 0, []
    
    # Run comparisons
    print(f"\n{'='*80}")
    print(f"Starting batch comparisons...")
    print(f"{'='*80}")
    
    successful = 0
    failed = 0
    failed_files = []
    
    # Prepare CSV output path if needed
    csv_output = args.csv_output if hasattr(args, 'csv_output') else None
    cut_videos = args.cut_videos if hasattr(args, 'cut_videos') else False
    video_names = args.video_names if hasattr(args, 'video_names') else None
    compute_dtw = args.compute_dtw if hasattr(args, 'compute_dtw') else True
    
    for i, gvhmr_file in enumerate(gvhmr_files, 1):
        print(f"\n[{i}/{len(gvhmr_files)}] Processing: {gvhmr_file.parent.name}")
        
        success = run_comparison(
            gvhmr_file, 
            reference, 
            args.python,
            csv_output=csv_output,
            cut_videos=cut_videos,
            video_names=video_names,
            compute_dtw=compute_dtw
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            failed_files.append(gvhmr_file.parent.name)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Batch Comparison Summary")
    print(f"{'='*80}")
    print(f"Total files: {len(gvhmr_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed comparisons:")
        for failed_name in failed_files:
            print(f"  ✗ {failed_name}")
    
    print(f"{'='*80}")
    
    if failed == 0:
        print("\n✓ All comparisons completed successfully!")
    
    return successful, failed, failed_files


def main():
    parser = argparse.ArgumentParser(
        description="Batch compare multiple GVHMR predictions against a reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all recordings for a single song and person
  python batch_compare_hmr.py --song Old_Town_Road --person Wontaek
  
  # Compare all recordings for multiple songs
  python batch_compare_hmr.py --song Old_Town_Road Unstoppable bad_guy --person Wontaek
  
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
        nargs='+',
        help="Song name(s) (e.g., 'Old_Town_Road' or 'Old_Town_Road Unstoppable bad_guy'). Will use default directory structure."
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
    parser.add_argument(
        "--csv_output",
        type=str,
        help="Path to CSV file for saving all metrics (will be created if doesn't exist)"
    )
    parser.add_argument(
        "--cut_videos",
        action="store_true",
        help="Cut videos based on optimal alignment"
    )
    parser.add_argument(
        "--video_names",
        type=str,
        nargs='+',
        default=['0_input_video.mp4', '1_incam.mp4'],
        help="Names of video files to cut (default: 0_input_video.mp4 1_incam.mp4)"
    )
    parser.add_argument(
        "--compute_dtw",
        action="store_true",
        default=True,
        help="Compute DTW alignment on the optimal cut segment (default: True)"
    )
    parser.add_argument(
        "--no_dtw",
        action="store_false",
        dest="compute_dtw",
        help="Skip DTW computation"
    )
    parser.add_argument(
        "--pause_iterations",
        action="store_true",
        help="Pause after each comparison and ask to continue"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.song or args.person:
        if not (args.song and args.person):
            parser.error("Both --song and --person are required when using song-based mode")
        
        # Process multiple songs
        songs = args.song if isinstance(args.song, list) else [args.song]
        
        all_successful = 0
        all_failed = 0
        all_failed_files = []
        
        for song_idx, song in enumerate(songs, 1):
            if len(songs) > 1:
                print(f"\n{'#'*80}")
                print(f"# SONG {song_idx}/{len(songs)}: {song}")
                print(f"{'#'*80}\n")
            
            # Construct paths based on song and person
            gvhmr_dir = pathlib.Path(f"{DEFAULT_HUMAN_BASE}/{args.person}/{song}")
            reference = pathlib.Path(f"{DEFAULT_REFERENCE_BASE}/{song}_cut/hmr4d_results.pt")
            
            # Check if paths exist
            if not gvhmr_dir.exists():
                print(f"Warning: GVHMR directory does not exist: {gvhmr_dir}")
                print(f"Skipping song: {song}\n")
                continue
            
            if not reference.exists():
                print(f"Warning: Reference file does not exist: {reference}")
                print(f"Skipping song: {song}\n")
                continue
            
            # Process this song
            successful, failed, failed_files = process_song(gvhmr_dir, reference, args)
            all_successful += successful
            all_failed += failed
            
            # Track failed files with song name
            for failed_name in failed_files:
                all_failed_files.append(f"{song}: {failed_name}")
        
        # Overall summary for multiple songs
        if len(songs) > 1:
            print(f"\n{'#'*80}")
            print(f"# OVERALL SUMMARY FOR ALL SONGS")
            print(f"{'#'*80}")
            print(f"Songs processed: {len(songs)}")
            print(f"Total comparisons successful: {all_successful}")
            print(f"Total comparisons failed: {all_failed}")
            
            if all_failed > 0:
                print(f"\nAll failed comparisons across songs:")
                for failed_info in all_failed_files:
                    print(f"  ✗ {failed_info}")
            
            print(f"{'#'*80}")
        
        sys.exit(1 if all_failed > 0 else 0)
        
    elif args.gvhmr_dir and args.reference:
        gvhmr_dir = pathlib.Path(args.gvhmr_dir)
        reference = pathlib.Path(args.reference)
        
        # Check if paths exist
        if not gvhmr_dir.exists():
            print(f"Error: GVHMR directory does not exist: {gvhmr_dir}")
            sys.exit(1)
        
        if not reference.exists():
            print(f"Error: Reference file does not exist: {reference}")
            sys.exit(1)
        
        # Process single custom path
        successful, failed, failed_files = process_song(gvhmr_dir, reference, args)
        sys.exit(1 if failed > 0 else 0)
        
    else:
        parser.error("Either use --song and --person, OR use --gvhmr_dir and --reference")


if __name__ == "__main__":
    main()
