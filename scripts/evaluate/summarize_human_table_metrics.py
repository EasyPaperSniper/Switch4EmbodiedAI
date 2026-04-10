#!/usr/bin/env python3
"""
Summarize normal-only human metrics into a compact table matching the paper-style layout.

This script uses cached per-recording comparison JSON files when available and can
optionally recompute missing metrics. It also aggregates Hand JDS scores into Easy,
Hard, and All splits.

Outputs:
    plots/human/human_table_summary.json
    plots/human/human_table_summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate.data.jds_scores import data as jds_scores


DEFAULT_GMR_BASE = Path("/home/jkim3662/Videos/Switch4EAI/HumanRecordings_GMR/trimmed")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "plots" / "human"
SONG_LEVELS = {
    "Old_Town_Road": 1,
    "Heart_Of_Glass": 2,
    "Unstoppable": 2,
    "Padam_Padam": 3,
    "Pink_Venom": 4,
}
LEADERBOARD_JDS = 13333
_GMR_CONTEXT: dict[str, object] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize normal-only human JDS/MPJPE/DTW/Smoothness metrics."
    )
    parser.add_argument(
        "--gmr-base",
        type=Path,
        default=DEFAULT_GMR_BASE,
        help=f"Base directory containing human GMR recordings (default: {DEFAULT_GMR_BASE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for summary outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--jds-score-type",
        choices=["Hand", "Arm"],
        default="Hand",
        help="Which JDS score channel to aggregate (default: Hand)",
    )
    parser.add_argument(
        "--recompute-missing",
        action="store_true",
        help="Recompute metrics when cached comparison JSON is missing or incomplete.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for per-recording metric collection (default: 1).",
    )
    return parser.parse_args()


def discover_normal_recordings(base_dir: Path) -> list[Path]:
    return sorted(base_dir.glob("*_normal_*/*_poses.pkl"))


def result_json_path(recording_path: Path) -> Path:
    return recording_path.with_suffix(".comparison_results.json")


def extract_song_name(recording_path: Path) -> str:
    recording_name = recording_path.parent.name
    for song in SONG_LEVELS:
        if recording_name.startswith(song):
            return song
    raise ValueError(f"Could not infer song name from {recording_path}")


def compute_jds_summary(score_type: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {"easy": [], "hard": [], "all": []}

    for song, conditions in jds_scores.items():
        bucket = "easy" if SONG_LEVELS[song] <= 2 else "hard"
        for condition, score_dict in conditions.items():
            if not condition.startswith("normal_"):
                continue
            scores = list(score_dict[score_type].values())
            buckets[bucket].extend(scores)
            buckets["all"].extend(scores)

    return {
        "easy": float(np.mean(buckets["easy"])),
        "hard": float(np.mean(buckets["hard"])),
        "all": float(np.mean(buckets["all"])),
    }


def compute_forward_kinematics_cached(
    kinematics_model: object,
    data: dict[str, np.ndarray],
    device: object,
    batch_size: int = 512,
) -> np.ndarray:
    import torch

    root_pos = data["root_pos"]
    root_rot = data["root_rot"]
    dof_pos = data["dof_pos"]

    body_positions_list: list[np.ndarray] = []
    for start_idx in range(0, root_pos.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, root_pos.shape[0])
        root_pos_batch = torch.from_numpy(root_pos[start_idx:end_idx]).to(
            device=device, dtype=torch.float32
        )
        root_rot_batch = torch.from_numpy(root_rot[start_idx:end_idx]).to(
            device=device, dtype=torch.float32
        )
        dof_pos_batch = torch.from_numpy(dof_pos[start_idx:end_idx]).to(
            device=device, dtype=torch.float32
        )
        body_pos_batch, _body_rot_batch = kinematics_model.forward_kinematics(
            root_pos_batch,
            root_rot_batch,
            dof_pos_batch,
        )
        body_positions_list.append(body_pos_batch.detach().cpu().numpy())

    return np.concatenate(body_positions_list, axis=0)


def find_optimal_time_alignment_mpjpe_fast(
    recorded_body_pos: np.ndarray, reference_body_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, range, range, float]:
    evaluate_dir = REPO_ROOT / "scripts" / "evaluate"
    if str(evaluate_dir) not in sys.path:
        sys.path.insert(0, str(evaluate_dir))

    from metrics import compute_mpjpe

    t_recorded = recorded_body_pos.shape[0]
    t_reference = reference_body_pos.shape[0]

    if t_recorded >= t_reference:
        best_error = float("inf")
        best_start_idx = 0
        max_shift = t_recorded - t_reference + 1
        for start_idx in range(max_shift):
            recorded_shifted = recorded_body_pos[start_idx : start_idx + t_reference]
            error = compute_mpjpe(recorded_shifted, reference_body_pos)
            if error < best_error:
                best_error = error
                best_start_idx = start_idx

        recorded_indices = range(best_start_idx, best_start_idx + t_reference)
        reference_indices = range(t_reference)
        return (
            recorded_body_pos[best_start_idx : best_start_idx + t_reference],
            reference_body_pos,
            recorded_indices,
            reference_indices,
            best_error,
        )

    best_error = float("inf")
    best_start_idx = 0
    max_shift = t_reference - t_recorded + 1
    for start_idx in range(max_shift):
        reference_shifted = reference_body_pos[start_idx : start_idx + t_recorded]
        error = compute_mpjpe(recorded_body_pos, reference_shifted)
        if error < best_error:
            best_error = error
            best_start_idx = start_idx

    recorded_indices = range(t_recorded)
    reference_indices = range(best_start_idx, best_start_idx + t_recorded)
    return (
        recorded_body_pos,
        reference_body_pos[best_start_idx : best_start_idx + t_recorded],
        recorded_indices,
        reference_indices,
        best_error,
    )


def get_gmr_context() -> dict[str, object]:
    global _GMR_CONTEXT
    if _GMR_CONTEXT is not None:
        return _GMR_CONTEXT

    evaluate_dir = REPO_ROOT / "scripts" / "evaluate"
    if str(evaluate_dir) not in sys.path:
        sys.path.insert(0, str(evaluate_dir))

    import compare_two_gmr as c
    import torch

    device = torch.device("cpu")
    kinematics_model = c.KinematicsModel(str(c.ROBOT_XML), device=device)

    reference_cache: dict[str, dict[str, object]] = {}
    for song, reference_path in c.REFERENCE_GMR_MAP_OFFLINE.items():
        reference_data = c.align_to_zero(c.load_gmr_data(reference_path))
        reference_body_pos = compute_forward_kinematics_cached(
            kinematics_model,
            reference_data,
            device,
        )
        reference_cache[song] = {
            "reference_path": reference_path,
            "reference_body_pos": reference_body_pos,
        }

    _GMR_CONTEXT = {
        "compare_module": c,
        "device": device,
        "kinematics_model": kinematics_model,
        "reference_cache": reference_cache,
    }
    return _GMR_CONTEXT


def load_cached_metrics(json_path: Path) -> dict[str, float]:
    data = json.loads(json_path.read_text())
    required = (
        "mpjpe_mm",
        "mpjpe_padded_mm",
        "dtw_mm",
        "recorded_smoothness",
        "recorded_mean_acceleration",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing fields in {json_path}: {missing}")
    return {
        "mpjpe_active_mm": float(data["mpjpe_mm"]),
        "mpjpe_all_mm": float(data["mpjpe_padded_mm"]),
        "dtw_mm": float(data["dtw_mm"]),
        "jerk_rad_per_s3": float(data["recorded_smoothness"]),
        "acc_rad_per_s2": float(data["recorded_mean_acceleration"]),
    }


def recompute_metrics(recording_path: Path) -> dict[str, float]:
    evaluate_dir = REPO_ROOT / "scripts" / "evaluate"
    if str(evaluate_dir) not in sys.path:
        sys.path.insert(0, str(evaluate_dir))

    from metrics import compute_dtw, compute_joint_smoothness, compute_mpjpe

    context = get_gmr_context()
    c = context["compare_module"]

    song = extract_song_name(recording_path)
    reference_entry = context["reference_cache"][song]

    recorded_data = c.load_gmr_data(str(recording_path))
    recorded_data = c.align_to_zero(recorded_data)

    recorded_body_pos = compute_forward_kinematics_cached(
        context["kinematics_model"],
        recorded_data,
        context["device"],
    )
    reference_body_pos = reference_entry["reference_body_pos"]
    (
        recorded_body_pos_aligned,
        reference_body_pos_aligned,
        recorded_aligned_indices,
        _reference_aligned_indices,
        _best_error,
    ) = find_optimal_time_alignment_mpjpe_fast(recorded_body_pos, reference_body_pos)

    mpjpe_active_mm = float(
        compute_mpjpe(
            recorded_body_pos_aligned[:, c.GMR_G1_MPJPE_FULL_BODY_INDICES],
            reference_body_pos_aligned[:, c.GMR_G1_MPJPE_FULL_BODY_INDICES],
        )
        * 1000.0
    )
    dtw_mm = float(
        compute_dtw(
            recorded_body_pos_aligned[:, c.GMR_G1_MPJPE_FULL_BODY_INDICES],
            reference_body_pos_aligned[:, c.GMR_G1_MPJPE_FULL_BODY_INDICES],
        )
        * 1000.0
    )
    jerk_rad_per_s3, _vel_disc, _mean_vel, acc_rad_per_s2 = compute_joint_smoothness(
        recorded_data["dof_pos"][recorded_aligned_indices],
        recorded_data["fps"],
    )

    metrics = {
        "mpjpe_active_mm": mpjpe_active_mm,
        "mpjpe_all_mm": mpjpe_active_mm,
        "dtw_mm": dtw_mm,
        "jerk_rad_per_s3": float(jerk_rad_per_s3),
        "acc_rad_per_s2": float(acc_rad_per_s2),
    }
    return metrics


def collect_single_recording(
    recording_path_str: str, recompute_missing: bool
) -> tuple[dict[str, object] | None, dict[str, str] | None]:
    recording_path = Path(recording_path_str)
    try:
        json_path = result_json_path(recording_path)
        if json_path.exists():
            try:
                metrics = load_cached_metrics(json_path)
            except KeyError:
                if not recompute_missing:
                    raise
                metrics = recompute_metrics(recording_path)
        elif recompute_missing:
            metrics = recompute_metrics(recording_path)
        else:
            raise FileNotFoundError(f"Missing cached results: {json_path}")

        return (
            {
                "recording": recording_path.parent.name,
                "song": extract_song_name(recording_path),
                "path": str(recording_path),
                **metrics,
            },
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return None, {"path": str(recording_path), "error": str(exc)}


def collect_motion_metrics(
    recording_paths: list[Path], recompute_missing: bool, workers: int
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    if workers <= 1:
        for recording_path in recording_paths:
            row, failure = collect_single_recording(str(recording_path), recompute_missing)
            if row is not None:
                rows.append(row)
            if failure is not None:
                failures.append(failure)
        return rows, failures

    max_workers = min(workers, len(recording_paths), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(collect_single_recording, str(recording_path), recompute_missing)
            for recording_path in recording_paths
        ]
        for future in as_completed(futures):
            row, failure = future.result()
            if row is not None:
                rows.append(row)
            if failure is not None:
                failures.append(failure)

    rows.sort(key=lambda row: str(row["path"]))
    failures.sort(key=lambda failure: failure["path"])

    return rows, failures


def build_summary(
    rows: list[dict[str, object]],
    expected_count: int,
    jds_summary: dict[str, float],
    score_type: str,
) -> dict[str, object]:
    if not rows:
        raise ValueError("No motion metrics were collected.")

    mpjpe_active = np.array([row["mpjpe_active_mm"] for row in rows], dtype=float)
    mpjpe_all = np.array([row["mpjpe_all_mm"] for row in rows], dtype=float)
    dtw = np.array([row["dtw_mm"] for row in rows], dtype=float)
    jerk = np.array([row["jerk_rad_per_s3"] for row in rows], dtype=float)
    acc = np.array([row["acc_rad_per_s2"] for row in rows], dtype=float)
    success_rate = 100.0 * len(rows) / expected_count if expected_count else 0.0

    human_row = {
        "player_setting": "Human",
        "jds_type": score_type,
        "jds_easy": jds_summary["easy"],
        "jds_hard": jds_summary["hard"],
        "jds_all": jds_summary["all"],
        "mpjpe_active_mm": float(np.mean(mpjpe_active)),
        "mpjpe_all_mm": float(np.mean(mpjpe_all)),
        "dtw_mm": float(np.mean(dtw)),
        "sr_percent": success_rate,
        "smoothness_jerk_rad_per_s3": float(np.mean(jerk)),
        "smoothness_acc_rad_per_s2": float(np.mean(acc)),
        "num_trials": len(rows),
        "expected_trials": expected_count,
    }

    leaderboard_row = {
        "player_setting": "Human-leaderBoard",
        "jds_type": score_type,
        "jds_easy": float(LEADERBOARD_JDS),
        "jds_hard": float(LEADERBOARD_JDS),
        "jds_all": float(LEADERBOARD_JDS),
        "mpjpe_active_mm": None,
        "mpjpe_all_mm": None,
        "dtw_mm": None,
        "sr_percent": 100.0,
        "smoothness_jerk_rad_per_s3": None,
        "smoothness_acc_rad_per_s2": None,
        "num_trials": None,
        "expected_trials": None,
    }

    return {
        "metadata": {
            "source_base": str(DEFAULT_GMR_BASE),
            "score_type": score_type,
            "expected_normal_recordings": expected_count,
            "successful_normal_recordings": len(rows),
            "recording_representation": "GMR-retargeted human recordings (*.pkl)",
            "reference_representation": "GMR offline reference motions",
            "run_filter": "normal only",
            "comparison_space": "GMR forward-kinematics full-body joint positions (38 joints)",
            "temporal_alignment": "best-shift alignment by non-PA MPJPE",
            "dtw_cost": "DTW over the same per-frame MPJPE cost after best-shift alignment",
        },
        "rows": [leaderboard_row, human_row],
        "per_recording_metrics": rows,
    }


def format_table_row(row: dict[str, object]) -> str:
    def fmt_jds(value: float | None) -> str:
        return "-" if value is None else f"{value:.0f}"

    def fmt_metric(value: float | None) -> str:
        return "-" if value is None else f"{value:.1f}"

    return (
        f"| {row['player_setting']} | {fmt_jds(row['jds_easy'])} | {fmt_jds(row['jds_hard'])} | "
        f"{fmt_jds(row['jds_all'])} | {fmt_metric(row['mpjpe_active_mm'])} | "
        f"{fmt_metric(row['mpjpe_all_mm'])} | {fmt_metric(row['dtw_mm'])} | "
        f"{fmt_metric(row['sr_percent'])} | "
        f"{fmt_metric(row['smoothness_jerk_rad_per_s3'])} | "
        f"{fmt_metric(row['smoothness_acc_rad_per_s2'])} |"
    )


def write_outputs(summary: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "human_table_summary.json"
    md_path = output_dir / "human_table_summary.md"

    json_path.write_text(json.dumps(summary, indent=2))

    lines = [
        "# Human Table Summary",
        "",
        "| Player-Setting | JDS Easy | JDS Hard | JDS All | MPJPE Active (mm) | MPJPE All (mm) | DTW (mm) | SR (%) | Smoothness Jerk (rad/s^3) | Smoothness Acc (rad/s^2) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(format_table_row(row) for row in summary["rows"])
    lines.extend(
        [
            "",
            "## Metadata",
            "",
            f"- Source base: `{summary['metadata']['source_base']}`",
            f"- JDS type: `{summary['metadata']['score_type']}`",
            f"- Successful normal recordings: {summary['metadata']['successful_normal_recordings']} / {summary['metadata']['expected_normal_recordings']}",
            f"- Recording representation: {summary['metadata']['recording_representation']}",
            f"- Reference representation: {summary['metadata']['reference_representation']}",
            f"- Run filter: {summary['metadata']['run_filter']}",
            f"- Comparison space: {summary['metadata']['comparison_space']}",
            f"- Temporal alignment: {summary['metadata']['temporal_alignment']}",
            f"- DTW cost: {summary['metadata']['dtw_cost']}",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")

    return json_path, md_path


def main() -> None:
    args = parse_args()

    recording_paths = discover_normal_recordings(args.gmr_base)
    if not recording_paths:
        raise SystemExit(f"No normal recordings found under {args.gmr_base}")

    print(
        f"Collecting metrics for {len(recording_paths)} normal recordings "
        f"with {args.workers} worker(s)..."
    )
    rows, failures = collect_motion_metrics(
        recording_paths,
        args.recompute_missing,
        args.workers,
    )
    jds_summary = compute_jds_summary(args.jds_score_type)
    summary = build_summary(rows, len(recording_paths), jds_summary, args.jds_score_type)
    if failures:
        summary["failures"] = failures

    json_path, md_path = write_outputs(summary, args.output_dir)

    human_row = summary["rows"][1]
    print(f"Saved JSON summary to: {json_path}")
    print(f"Saved Markdown summary to: {md_path}")
    print(
        "Basis: GMR-retargeted human recordings vs GMR offline references, "
        "normal runs only."
    )
    print(
        "Metric definition: non-PA MPJPE on 38 GMR FK joints; DTW uses the same "
        "per-frame MPJPE cost after best-shift alignment."
    )
    print(
        "Human row: "
        f"JDS Easy={human_row['jds_easy']:.1f}, "
        f"JDS Hard={human_row['jds_hard']:.1f}, "
        f"JDS All={human_row['jds_all']:.1f}, "
        f"MPJPE Active={human_row['mpjpe_active_mm']:.1f} mm, "
        f"MPJPE All={human_row['mpjpe_all_mm']:.1f} mm, "
        f"DTW={human_row['dtw_mm']:.1f} mm, "
        f"SR={human_row['sr_percent']:.1f}%, "
        f"Jerk={human_row['smoothness_jerk_rad_per_s3']:.1f} rad/s^3, "
        f"Acc={human_row['smoothness_acc_rad_per_s2']:.1f} rad/s^2"
    )
    if failures:
        print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
