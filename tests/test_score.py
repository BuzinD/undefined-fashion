"""Tests for the scoring script. They use tiny fake data, so they run in seconds."""

import json
import time
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from fashion.score import (
    Result,
    SubmissionError,
    accuracy,
    count_parameters,
    load_train_function,
    save,
    train_with_time_limit,
)

SUBMISSIONS = sorted(p for p in Path("submissions").iterdir() if p.is_dir())


def fake_images(count: int = 20) -> TensorDataset:
    images = torch.rand(count, 1, 28, 28)
    labels = torch.randint(0, 10, (count,))
    return TensorDataset(images, labels)


def write_submission(folder: Path, code: str) -> Path:
    folder.mkdir()
    (folder / "model.py").write_text(code)
    return folder


@pytest.mark.parametrize("folder", SUBMISSIONS, ids=lambda p: p.name)
def test_every_submission_has_a_train_function(folder: Path) -> None:
    assert callable(load_train_function(folder))


def test_missing_model_file(tmp_path: Path) -> None:
    with pytest.raises(SubmissionError, match="model.py"):
        load_train_function(tmp_path)


def test_missing_train_function(tmp_path: Path) -> None:
    folder = write_submission(tmp_path / "alice", "x = 1\n")
    with pytest.raises(SubmissionError, match="def train"):
        load_train_function(folder)


def test_train_must_return_a_model() -> None:
    with pytest.raises(SubmissionError, match="nn.Module"):
        train_with_time_limit(lambda train_set: "not a model", fake_images())


def test_training_that_runs_too_long_is_stopped() -> None:
    def slow_train(train_set):
        time.sleep(5)
        return nn.Linear(1, 1)

    start = time.perf_counter()
    with pytest.raises(SubmissionError, match="longer than"):
        train_with_time_limit(slow_train, fake_images(), limit=0.5)
    assert time.perf_counter() - start < 2


def test_accuracy_and_size() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    assert count_parameters(model) == 784 * 10 + 10
    assert 0 <= accuracy(model, fake_images()) <= 1


def test_perfect_model_scores_one() -> None:
    test_set = fake_images()
    answers = torch.eye(10)[test_set.tensors[1]]

    class AlwaysRight(nn.Module):
        def forward(self, images):
            return answers[: len(images)]

    assert accuracy(AlwaysRight(), test_set) == 1.0


def test_save_writes_one_json_file(tmp_path: Path) -> None:
    result = Result("alice", 0.9, 1000, 12.3, "2026-01-02T03:04:05+00:00")
    path = save(result, tmp_path)
    assert path.name == "2026-01-02T03-04-05_alice.json"
    assert json.loads(path.read_text())["accuracy"] == 0.9
