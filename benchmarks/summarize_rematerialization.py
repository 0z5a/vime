"""Recompute tables from independent process cohorts, preserving negative results."""

import hashlib
import json
import math
import statistics
from pathlib import Path


def interval(values):
    mean = statistics.mean(values)
    radius = 4.302652729696142 * statistics.stdev(values) / math.sqrt(3)
    return math.exp(mean), [math.exp(mean - radius), math.exp(mean + radius)]


def summarize(directory, baseline, expected):
    rows = [
        json.loads(line) for path in sorted(directory.glob("seed-*.jsonl")) for line in path.read_text().splitlines()
    ]
    assert len(rows) == expected
    def key(row):
        plan = tuple(row["plan"][name] for name in ("loop_interval", "layer_interval", "token_chunk")) if "plan" in row else ()
        return row["family"], plan
    results = []
    for family, plan in sorted({key(row) for row in rows}):
        group = [row for row in rows if key(row) == (family, plan)]
        logs = []
        for seed in (31, 32, 33):
            arms = {
                arm: [row["actor_update_seconds"] for row in group if row["seed"] == seed and row["arm"] == arm]
                for arm in (baseline, "remat")
            }
            assert all(arms.values()) and len(arms[baseline]) == len(arms["remat"])
            logs.append(math.log(statistics.geometric_mean(arms[baseline]) / statistics.geometric_mean(arms["remat"])))
        ratio, ci = interval(logs)
        result = {
            "family": family,
            "plan": plan,
            "speed_ratio": ratio,
            "ci95_t_log": ci,
            "per_start_ratios": [math.exp(value) for value in logs],
            "actor_update_seconds_geomean": {
                arm: statistics.geometric_mean(row["actor_update_seconds"] for row in group if row["arm"] == arm)
                for arm in (baseline, "remat")
            },
            "max_gradient_relative_l2": max(row["gradient_relative_l2"] for row in group),
        }
        if plan:
            storage = {
                arm: {row["forward_observed_saved_storage_bytes"] for row in group if row["arm"] == arm}
                for arm in (baseline, "remat")
            }
            assert all(len(values) == 1 for values in storage.values()), storage
            result["saved_storage_bytes"] = {arm: next(iter(values)) for arm, values in storage.items()}
            result["saved_storage_reduction"] = 1 - next(iter(storage["remat"])) / next(iter(storage[baseline]))
        results.append(result)
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
    return results


if __name__ == "__main__":
    root = Path(__file__).parent / "results"
    ablation = summarize(root / "rematerialization-cpu", "layer-baseline", 864)
    clean = summarize(root / "rematerialization-clean-cpu", "parent", 36)
    print(
        "| Family | Loop / layer / query chunk | Speed ratio [95% CI] | Observed saved bytes, baseline → candidate | Change |"
    )
    print("|---|---:|---:|---:|---:|")
    for row in ablation:
        lower, upper = row["ci95_t_log"]
        sizes = row["saved_storage_bytes"]
        change = row["saved_storage_reduction"]
        direction = "lower" if change >= 0 else "higher"
        print(
            f"| {row['family']} | {' / '.join(map(str, row['plan']))} | {row['speed_ratio']:.3f} [{lower:.3f}, {upper:.3f}] | "
            f"{sizes['layer-baseline']:,} → {sizes['remat']:,} | {100 * abs(change):.2f}% {direction} |"
        )
    print("\n| Family | Parent ms | Candidate ms | Clean-parent speed ratio [95% CI] |")
    print("|---|---:|---:|---:|")
    for row in clean:
        times = row["actor_update_seconds_geomean"]
        lower, upper = row["ci95_t_log"]
        print(
            f"| {row['family']} | {1000 * times['parent']:.2f} | {1000 * times['remat']:.2f} | "
            f"{row['speed_ratio']:.3f} [{lower:.3f}, {upper:.3f}] |"
        )
