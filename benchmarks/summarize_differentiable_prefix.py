"""Three-start log-ratio intervals for the full fixed-trace update scope."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def summarize(directory: Path, baseline: str, expected_rows: int) -> None:
    rows = [
        json.loads(line) for path in sorted(directory.glob("seed*.jsonl")) for line in path.read_text().splitlines()
    ]
    assert len(rows) == expected_rows
    keys = sorted({(row["family"], row["P"], row["D"], row["G"]) for row in rows})
    results = []
    print("| Family | P / D / G | Baseline ms | Prefix ms | Baseline / prefix [95% CI] |")
    print("|---|---:|---:|---:|---:|")
    for family, prompt, response, group_size in keys:
        group = [
            row
            for row in rows
            if (row["family"], row["P"], row["D"], row["G"]) == (family, prompt, response, group_size)
        ]
        logs = []
        for seed in (31, 32, 33):
            times = {
                arm: [row["actor_update_seconds"] for row in group if row["seed"] == seed and row["arm"] == arm]
                for arm in (baseline, "shared-prefix")
            }
            assert len(times[baseline]) == len(times["shared-prefix"]) > 0
            logs.append(
                math.log(
                    statistics.geometric_mean(times[baseline]) / statistics.geometric_mean(times["shared-prefix"])
                )
            )
        mean = statistics.mean(logs)
        radius = 4.302652729696142 * statistics.stdev(logs) / math.sqrt(3)
        ratio, lower, upper = math.exp(mean), math.exp(mean - radius), math.exp(mean + radius)
        times = {
            arm: statistics.geometric_mean(row["actor_update_seconds"] for row in group if row["arm"] == arm)
            for arm in (baseline, "shared-prefix")
        }
        results.append(
            {
                "family": family,
                "P": prompt,
                "D": response,
                "G": group_size,
                "speed_ratio": ratio,
                "ci95_t_log": [lower, upper],
                "per_start_ratio": [math.exp(v) for v in logs],
                "actor_update_seconds_geomean": times,
                "max_gradient_relative_l2": max(row["gradient_relative_l2"] for row in group),
            }
        )
        print(
            f"| {family} | {prompt} / {response} / {group_size} | {times[baseline] * 1000:.2f} | "
            f"{times['shared-prefix'] * 1000:.2f} | {ratio:.3f} [{lower:.3f}, {upper:.3f}] |"
        )
    (directory / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (directory / "sha256.json").write_text(
        json.dumps(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(directory.glob("*.json*"))
                if path.name != "sha256.json"
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--rows", required=True, type=int)
    args = parser.parse_args()
    summarize(args.directory, args.baseline, args.rows)
