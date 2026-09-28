import pytest
import torch

from vime.utils.tensor_backper import TensorBackuper


def test_live_actor_reacquires_storage_after_completed_load():
    model = torch.nn.Linear(3, 2)
    backuper = TensorBackuper.create(model.named_parameters, single_tag="actor")
    previous = backuper.get("actor")
    restored = {name: value.detach().clone() + 1 for name, value in model.state_dict().items()}
    model.load_state_dict(restored, assign=True)
    current = backuper.get("actor")
    for name, parameter in model.named_parameters():
        assert current[name].data_ptr() == parameter.data_ptr()
        assert current[name].data_ptr() != previous[name].data_ptr()
        torch.testing.assert_close(current[name], restored[name], rtol=0, atol=0)


def test_live_actor_tracks_optimizer_updates_without_snapshot_or_sync(monkeypatch):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    backuper = TensorBackuper.create(model.named_parameters, single_tag="actor")

    def unexpected_sync():
        pytest.fail("live actor snapshots must not synchronize CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_sync)
    before = model.weight.detach().clone()
    for _ in range(3):
        backuper.restore("actor")
        optimizer.zero_grad()
        model(torch.ones(2, 3)).square().mean().backward()
        optimizer.step()
        backuper.backup("actor")
        published = backuper.get("actor")
        for name, parameter in model.named_parameters():
            assert published[name].data_ptr() == parameter.data_ptr()
            assert not published[name].requires_grad
            torch.testing.assert_close(published[name], parameter, rtol=0, atol=0)
    assert not torch.equal(before, model.weight)
    for operation in (backuper.get, backuper.backup, backuper.restore):
        with pytest.raises(ValueError, match="only available for actor"):
            operation("ref")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="snapshot parity requires CUDA")
def test_live_and_snapshot_actor_have_identical_training_and_published_weights():
    torch.manual_seed(42)
    baseline = torch.nn.Linear(3, 2, device="cuda")
    resident = torch.nn.Linear(3, 2, device="cuda")
    resident.load_state_dict(baseline.state_dict())
    models = (baseline, resident)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=0.01) for model in models]
    backupers = [
        TensorBackuper.create(baseline.named_parameters, single_tag=None),
        TensorBackuper.create(resident.named_parameters, single_tag="actor"),
    ]
    for backuper in backupers:
        backuper.backup("actor")
    inputs = torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3)
    for _ in range(3):
        for model, optimizer, backuper in zip(models, optimizers, backupers, strict=True):
            backuper.restore("actor")
            optimizer.zero_grad()
            model(inputs).square().mean().backward()
            optimizer.step()
            backuper.backup("actor")
        left, right = [backuper.get("actor") for backuper in backupers]
        for name in left:
            torch.testing.assert_close(left[name], right[name].cpu(), rtol=0, atol=0)
        for left_state, right_state in zip(optimizers[0].state.values(), optimizers[1].state.values(), strict=True):
            for key in left_state:
                torch.testing.assert_close(left_state[key], right_state[key], rtol=0, atol=0)
