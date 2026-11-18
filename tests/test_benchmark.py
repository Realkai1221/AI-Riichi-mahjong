import pytest
import torch

from riichi_ai.benchmark import benchmark_compute


def test_benchmark_checks_real_optimizer_updates():
    report = benchmark_compute("cpu", [2], iterations=1, warmup=0, width=4, blocks=1)
    result = report["results"][0]
    assert result["initial_update_validation"]["parameter_change_norm"] > 0
    assert result["final_update_validation"]["finite_parameters"]
    assert result["inference"]["examples_per_second"] > 0


def test_corrupt_optimizer_update_fails_instead_of_reporting_throughput(monkeypatch):
    original_step = torch.optim.AdamW.step
    calls = 0

    def corrupt_final_step(optimizer, *args, **kwargs):
        nonlocal calls
        result = original_step(optimizer, *args, **kwargs)
        calls += 1
        if calls == 2:
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].fill_(float("nan"))
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", corrupt_final_step)
    with pytest.raises(RuntimeError, match="Non-finite"):
        benchmark_compute("cpu", [2], iterations=1, warmup=0, width=4, blocks=1)
