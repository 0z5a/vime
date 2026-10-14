"""CPU tests for tools/bench_spec_rollout.py (plan, dry run and summary)."""

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

NUM_GPUS = 0

_PATH = Path(__file__).resolve().parents[1] / "tools" / "bench_spec_rollout.py"
_SPEC = importlib.util.spec_from_file_location("bench_spec_rollout", _PATH)
bench = importlib.util.module_from_spec(_SPEC)
sys.modules["bench_spec_rollout"] = bench
_SPEC.loader.exec_module(bench)


def _plan(tmp_path, **overrides):
    values = dict(
        out=str(tmp_path),
        num_rollout=5,
        seed=7,
        modes=list(bench.MODES),
        layouts=list(bench.LAYOUTS),
        env=["MODEL_DIR=/data/Qwen3-4B"],
        extra_args=[],
    )
    values.update(overrides)
    return bench.plan(Namespace(**values))


@pytest.mark.unit
def test_plan_covers_every_mode_and_layout_with_the_same_seed(tmp_path):
    result = _plan(tmp_path)

    assert [arm["name"] for arm in result["arms"]] == [
        f"{layout}-{mode}" for layout in bench.LAYOUTS for mode in bench.MODES
    ]
    for arm in result["arms"]:
        env = arm["env"]
        assert env["SPEC_MODE"] == arm["mode"]
        # Every arm trains the policy, so acceptance is measured under the same drift.
        assert env["DSPARK_FREEZE_POLICY"] == "0"
        assert env["NUM_ROLLOUT"] == "5"
        assert "--seed 7" in env["EXTRA_ARGS"] and "--rollout-seed 7" in env["EXTRA_ARGS"]
        assert env["TENSORBOARD_DIR"] == str(Path(arm["dir"]) / "tensorboard")
        assert env["MODEL_DIR"] == "/data/Qwen3-4B"
        assert Path(arm["script"]).exists()
    assert json.loads((tmp_path / "plan.json").read_text()) == result


@pytest.mark.unit
def test_dry_run_runs_nothing(tmp_path, capsys):
    _plan(tmp_path, layouts=["colocate"])

    code = bench.run(Namespace(out=str(tmp_path), arms=["colocate-off"], dry_run=True))

    assert code == 0
    assert 'SPEC_MODE="off"' in capsys.readouterr().out
    assert not (tmp_path / "colocate-off" / "run.json").exists()


@pytest.mark.unit
def test_summary_skips_warmup_and_splits_acceptance_into_thirds():
    scalars = {
        "perf/rollout_time": {0: 100.0, 1: 10.0, 2: 12.0, 3: 14.0},
        "rollout/spec_accept_rate": {0: 0.9, 1: 0.6, 2: 0.5, 3: 0.4},
        # The token-weighted key wins over the per-sample mean when both exist.
        "rollout/spec_accept_length/token_weighted": {0: 9.0, 1: 3.0, 2: 3.0, 3: 3.0},
        "rollout/spec_accept_length": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
    }

    summary = bench.summarize_arm(scalars, warmup=1)

    assert summary["rollout_time"] == pytest.approx(12.0)
    assert summary["rollout_time_steps"] == 3
    assert summary["spec_accept_rate_first_third"] == pytest.approx(0.6)
    assert summary["spec_accept_rate_last_third"] == pytest.approx(0.4)
    assert summary["spec_accept_length"] == pytest.approx(3.0)
    assert "logprob_abs_diff" not in summary


@pytest.mark.unit
def test_speedup_is_against_off_arm_of_the_same_layout():
    arms = [
        {"name": f"{layout}-{mode}", "layout": layout, "mode": mode}
        for layout in bench.LAYOUTS
        for mode in ("off", "online")
    ]
    summaries = {
        "colocate-off": {"rollout_time": 30.0},
        "colocate-online": {"rollout_time": 20.0},
        "non-colocate-off": {"rollout_time": 40.0},
        "non-colocate-online": {"rollout_time": 10.0},
    }

    rows = {row["arm"]: row for row in bench.compare(summaries, arms)}

    assert rows["colocate-online"]["rollout_time_speedup_vs_off"] == pytest.approx(1.5)
    assert rows["non-colocate-online"]["rollout_time_speedup_vs_off"] == pytest.approx(4.0)
    assert rows["colocate-off"]["rollout_time_speedup_vs_off"] == pytest.approx(1.0)


@pytest.mark.unit
def test_summarize_reads_tensorboard_written_by_several_processes(tmp_path):
    pytest.importorskip("tensorboard")
    from torch.utils.tensorboard import SummaryWriter

    _plan(tmp_path, layouts=["colocate"], modes=["off", "online"])
    for name, rollout_times in (("colocate-off", [50.0, 30.0, 30.0]), ("colocate-online", [50.0, 20.0, 20.0])):
        tb_dir = tmp_path / name / "tensorboard"
        # The rollout manager and the trainer write separate event files.
        rollout_writer, train_writer = SummaryWriter(str(tb_dir)), SummaryWriter(str(tb_dir))
        for step, value in enumerate(rollout_times):
            rollout_writer.add_scalar("perf/rollout_time", value, step)
            train_writer.add_scalar("train/train_rollout_logprob_abs_diff", 0.01, step)
        rollout_writer.close()
        train_writer.close()

    rows = {row["arm"]: row for row in bench.summarize(Namespace(out=str(tmp_path), warmup=1))}

    assert rows["colocate-online"]["rollout_time_speedup_vs_off"] == pytest.approx(1.5)
    assert rows["colocate-online"]["logprob_abs_diff"] == pytest.approx(0.01)
    assert "| colocate-online |" in (tmp_path / "summary.md").read_text()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
