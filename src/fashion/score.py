"""Score a submission: train it, then test it on images it has never seen.

Run it like this:

    uv run score submissions/<your-github-name>

GitHub runs the same command on pull requests and after merges, with `--save`
to keep the results for the leaderboard.
"""

import argparse
import importlib.util
import json
import os
import signal
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fashion import data

TIME_LIMIT_SECONDS = 10 * 60

type TrainFunction = Callable[[Dataset], nn.Module]


class SubmissionError(Exception):
    """Something is wrong with a submission. The message says what, in plain words."""


@dataclass
class Result:
    name: str
    accuracy: float
    parameters: int
    train_seconds: float
    date: str
    code_url: str | None = None


def load_train_function(folder: Path) -> TrainFunction:
    """Import model.py from a submission folder and return its train() function."""
    path = folder / "model.py"
    if not path.is_file():
        raise SubmissionError(f"{path} not found. Each submission needs a model.py file.")

    spec = importlib.util.spec_from_file_location(f"submission_{folder.name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    train = getattr(module, "train", None)
    if not callable(train):
        raise SubmissionError(f"{path} must define: def train(train_set) -> nn.Module")
    return train


@contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Stop the code inside if it runs longer than `seconds`.

    This uses an alarm signal, which exists on Linux and macOS. On Windows the
    code is not stopped early, but train_with_time_limit() still rejects it.
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def stop(signum, frame):
        raise SubmissionError(f"Training took longer than the limit of {seconds:.0f} seconds.")

    previous = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def train_with_time_limit(
    train: TrainFunction, train_set: Dataset, limit: float = TIME_LIMIT_SECONDS
) -> tuple[nn.Module, float]:
    """Run the student's train() and return the model and how many seconds it took."""
    torch.manual_seed(0)  # same starting point for everyone, so runs are repeatable
    start = time.perf_counter()
    with time_limit(limit):
        model = train(train_set)
    seconds = time.perf_counter() - start

    if seconds > limit:
        raise SubmissionError(f"Training took {seconds:.0f} seconds. The limit is {limit:.0f}.")
    if not isinstance(model, nn.Module):
        raise SubmissionError("train() must return a PyTorch model (an nn.Module).")
    return model, seconds


def count_parameters(model: nn.Module) -> int:
    """How many numbers the model learns by gradient descent (its parameters).

    Buffers, such as stored class averages, are not counted. A model can have 0.
    """
    return sum(p.numel() for p in model.parameters())


def accuracy(model: nn.Module, test_set: Dataset) -> float:
    """The share of test images the model labels correctly, from 0 to 1."""
    model = model.to("cpu").eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in DataLoader(test_set, batch_size=1000):
            outputs = model(images)
            if outputs.ndim != 2 or len(outputs) != len(labels):
                raise SubmissionError(
                    f"For a batch of {len(labels)} images the model returned shape "
                    f"{tuple(outputs.shape)}. It must return one row of class scores per "
                    f"image, shape ({len(labels)}, 10)."
                )
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return correct / total


def code_url(name: str) -> str | None:
    """A link to the submission's code on GitHub, when running inside GitHub Actions."""
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    sha = os.environ.get("GITHUB_SHA")
    if not (server and repo and sha):
        return None
    return f"{server}/{repo}/blob/{sha}/submissions/{name}/model.py"


def score(folder: Path, limit: float = TIME_LIMIT_SECONDS) -> Result:
    """Train one submission and measure it."""
    train = load_train_function(folder)
    # Load both sets before training. Downloading the first time is not counted as
    # training time, and the test images are already in memory before student code runs.
    train_set = data.train_set()
    test_set = data.test_set()
    model, seconds = train_with_time_limit(train, train_set, limit)
    return Result(
        name=folder.name,
        accuracy=round(accuracy(model, test_set), 4),
        parameters=count_parameters(model),
        train_seconds=round(seconds, 1),
        date=datetime.now(UTC).isoformat(timespec="seconds"),
        code_url=code_url(folder.name),
    )


def save(result: Result, results_dir: Path) -> Path:
    """Write the result as a small JSON file, one file per scored submission."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.date.replace(":", "-").removesuffix("+00-00")
    path = results_dir / f"{stamp}_{result.name}.json"
    path.write_text(json.dumps(asdict(result), indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train submissions and print their scores.")
    parser.add_argument("folders", type=Path, nargs="+", help="for example: submissions/baseline")
    parser.add_argument("--save", type=Path, metavar="DIR", help="also save the results here")
    args = parser.parse_args()

    failed = []
    for folder in args.folders:
        try:
            result = score(folder)
        except SubmissionError as error:
            print(f"Error in {folder}: {error}", file=sys.stderr)
            failed.append(folder.name)
            continue

        print()
        print(f"Submission:     {result.name}")
        print(f"Test accuracy:  {result.accuracy:.2%}")
        print(f"Parameters:     {result.parameters:,}")
        print(f"Training time:  {result.train_seconds:.0f} seconds")
        if args.save:
            print(f"Saved to:       {save(result, args.save)}")

    if failed:
        sys.exit(f"Could not score: {', '.join(failed)}")
