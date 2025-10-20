#!/usr/bin/env python3
"""
Run batch comparison for ALL songs and ALL persons in the directory structure.

This script automatically discovers all persons and songs in the directory structure
and runs comparisons for each combination.

Usage:
    python batch_compare_all.py
    python batch_compare_all.py --dry-run
    python batch_compare_all.py --persons WT MG
    python batch_compare_all.py --songs Old_Town_Road Unstoppable
"""

import argparse
import pathlib
import subprocess
import sys
from glob import glob

HERE = pathlib.Path(__file__).parent
BATCH_SCRIPT = HERE / "batch_compare_hmr.py"

# Default base directories
DEFAULT_HUMAN_BASE = pathlib.Path("/home/jkim3662/Videos/Switch4EAI/HumanRecordings_GVHMR/raw/")
DEFAULT_REFERENCE_BASE = pathlib.Path("/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GVHMR/cut_mirrored")


def discover_persons(base_dir):
    """
    Discover all person directories.
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        List of person names
    """
    if not base_dir.exists():
        return []
    
    persons = [d.name for d in base_dir.iterdir() if d.is_dir()]
    return sorted(persons)


def discover_songs_for_person(base_dir, person):
    """
    Discover all song directories for a person.
    
    Args:
        base_dir: Base directory
        person: Person name
        
    Returns:
        List of song names
    """
    person_dir = base_dir / person
    if not person_dir.exists():
        return []
    
    songs = [d.name for d in person_dir.iterdir() if d.is_dir()]
    return sorted(songs)


def discover_all_reference_songs(reference_base):
    """
    Discover all available reference songs.
    
    Args:
        reference_base: Reference base directory
        
    Returns:
        List of song names (without _cut suffix)
    """
    if not reference_base.exists():
        return []
    
    songs = []
    for d in reference_base.iterdir():
        if d.is_dir() and d.name.endswith("_cut"):
            # Check if hmr4d_results.pt exists
            if (d / "hmr4d_results.pt").exists():
                # Remove _cut suffix
                song_name = d.name[:-4]  # Remove "_cut"
                songs.append(song_name)
    
    return sorted(songs)


def run_batch_comparison(person, song, python_cmd, csv_output=None, cut_videos=False, video_names=None, compute_dtw=True, dry_run=False):
    """
    Run batch comparison for a person-song combination.
    
    Args:
        person: Person name
        song: Song name
        python_cmd: Python command to use
        csv_output: Path to CSV file for metrics (optional)
        cut_videos: Whether to cut videos based on alignment
        video_names: List of video filenames to cut (optional)
        compute_dtw: Whether to compute DTW alignment (default: True)
        dry_run: If True, only print what would be done
        
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        python_cmd,
        str(BATCH_SCRIPT),
        "--song", song,
        "--person", person
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
    
    if dry_run:
        cmd.append("--dry-run")
    
    print(f"\n{'='*80}")
    print(f"Running batch comparison:")
    print(f"  Person: {person}")
    print(f"  Song: {song}")
    if csv_output:
        print(f"  CSV Output: {csv_output}")
    if cut_videos:
        print(f"  Video cutting: Enabled")
    print(f"  DTW computation: {'Enabled' if compute_dtw else 'Disabled'}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("[DRY RUN] - Would execute the above command")
        return True
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ Batch comparison completed successfully for {person} - {song}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Batch comparison failed for {person} - {song} with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run batch comparisons for ALL songs and ALL persons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all comparisons for all persons and all songs
  python batch_compare_all.py
  
  # Run for specific persons only
  python batch_compare_all.py --persons WT MG
  
  # Run for specific songs only
  python batch_compare_all.py --songs Old_Town_Road Unstoppable
  
  # Dry run to see what would be executed
  python batch_compare_all.py --dry-run
  
  # Combine filters
  python batch_compare_all.py --persons WT --songs Old_Town_Road Unstoppable
        """
    )
    
    parser.add_argument(
        "--persons",
        type=str,
        nargs='+',
        help="Specific person(s) to process (default: all)"
    )
    parser.add_argument(
        "--songs",
        type=str,
        nargs='+',
        help="Specific song(s) to process (default: all)"
    )
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
        "--human-base",
        type=str,
        default=str(DEFAULT_HUMAN_BASE),
        help=f"Base directory for human recordings (default: {DEFAULT_HUMAN_BASE})"
    )
    parser.add_argument(
        "--reference-base",
        type=str,
        default=str(DEFAULT_REFERENCE_BASE),
        help=f"Base directory for reference recordings (default: {DEFAULT_REFERENCE_BASE})"
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        help="Path to CSV file for saving all metrics (will be created/appended to)"
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
   
    args = parser.parse_args()
    
    human_base = pathlib.Path(args.human_base)
    reference_base = pathlib.Path(args.reference_base)
    
    # Discover available data
    print(f"\n{'#'*80}")
    print(f"# DISCOVERING AVAILABLE DATA")
    print(f"{'#'*80}")
    print(f"Human recordings base: {human_base}")
    print(f"Reference base: {reference_base}")
    
    # Get persons
    if args.persons:
        persons = args.persons
        print(f"\nUsing specified persons: {persons}")
    else:
        persons = discover_persons(human_base)
        print(f"\nDiscovered persons: {persons}")
    
    if not persons:
        print("Error: No persons found or specified")
        sys.exit(1)
    
    # Get reference songs
    available_ref_songs = discover_all_reference_songs(reference_base)
    print(f"Available reference songs: {available_ref_songs}")
    
    if not available_ref_songs:
        print("Error: No reference songs found")
        sys.exit(1)
    
    # Build list of (person, song) combinations to process
    combinations = []
    
    for person in persons:
        # Get songs for this person
        person_songs = discover_songs_for_person(human_base, person)
        
        if not person_songs:
            print(f"\nWarning: No songs found for person: {person}")
            continue
        
        print(f"\nPerson '{person}' has songs: {person_songs}")
        
        # Filter by requested songs if specified
        if args.songs:
            songs_to_process = [s for s in person_songs if s in args.songs]
        else:
            songs_to_process = person_songs
        
        # Only include songs that have references
        for song in songs_to_process:
            if song in available_ref_songs:
                combinations.append((person, song))
            else:
                print(f"  Warning: No reference found for song '{song}', skipping")
    
    if not combinations:
        print("\nError: No valid person-song combinations found")
        sys.exit(1)
    
    # Summary
    print(f"\n{'#'*80}")
    print(f"# BATCH COMPARISON PLAN")
    print(f"{'#'*80}")
    print(f"Total combinations to process: {len(combinations)}")
    print("\nCombinations:")
    for i, (person, song) in enumerate(combinations, 1):
        print(f"  {i}. {person} - {song}")
    
    if args.dry_run:
        print("\n[DRY RUN] - No actual comparisons will be executed")
    
    # Confirm if not dry run and many combinations
    if not args.dry_run and len(combinations) > 5:
        print(f"\nThis will run {len(combinations)} batch comparisons (which may take a long time).")
        response = input("Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted by user")
            sys.exit(0)
    
    # Execute comparisons
    print(f"\n{'#'*80}")
    print(f"# STARTING BATCH COMPARISONS")
    print(f"{'#'*80}")
    
    successful = 0
    failed = 0
    failed_combinations = []
    
    for i, (person, song) in enumerate(combinations, 1):
        print(f"\n{'='*80}")
        print(f"COMBINATION {i}/{len(combinations)}: {person} - {song}")
        print(f"{'='*80}")
        
        success = run_batch_comparison(
            person, 
            song, 
            args.python,
            csv_output=args.csv_output,
            cut_videos=args.cut_videos,
            video_names=args.video_names,
            compute_dtw=args.compute_dtw,
            dry_run=args.dry_run
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            failed_combinations.append(f"{person} - {song}")
            
    # Final summary
    print(f"\n{'#'*80}")
    print(f"# FINAL SUMMARY")
    print(f"{'#'*80}")
    print(f"Total combinations processed: {len(combinations)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed combinations:")
        for failed_combo in failed_combinations:
            print(f"  ✗ {failed_combo}")
    else:
        print(f"\n✓ All batch comparisons completed successfully!")
    
    print(f"{'#'*80}")
    
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
