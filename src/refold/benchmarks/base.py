"""What a benchmark provides: which instances, how they are run, how they are scored.

    name           the --benchmark value
    runner_module  module the launcher executes with `python -m` (see runners/)
    config         the benchmark's prompt/environment YAML, passed first with -c
    data_flags()   flags telling the runner where its data is
    provenance()   which data exactly: the dataset revision, or the task directory's commit
    load()         the instance ids to run (validated against the dataset)
    score()        the benchmark's own official harness -- never a local reimplementation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

CONFIGS = Path(__file__).parent / "configs"
SETS = Path(__file__).parent / "sets"

# The Hugging Face datasets, pinned to the revision of every reported number. A dataset
# can change under its name: SWE-bench Pro's main branch now holds 642 of the 731 tasks
# and lacks 6 of the 50.
HF_DATASETS = {  # flavor -> (dataset, split, revision)
    "verified": ("princeton-nlp/SWE-Bench_Verified", "test", "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"),
    "pro": ("ScaleAI/SWE-bench_Pro", "test", "7ab5114912baf22bb098818e604c02fe7ad2c11f"),
    "msb": ("ByteDance-Seed/Multi-SWE-bench-flash", "train", "b0485dbebaf8a1317ebf140e80e6fc6c02d3502b"),
}


def load_hf(flavor: str):
    """The pinned split of one of HF_DATASETS."""
    from datasets import load_dataset

    name, split, revision = HF_DATASETS[flavor]
    return load_dataset(name, split=split, revision=revision)


@dataclass
class ScoreReport:
    """resolved / submitted, and which. The ids are kept because an instance that never
    produced a trajectory silently leaves the denominator, and the rate goes up."""

    submitted: int
    resolved: int
    empty_patch: int = 0
    resolved_ids: tuple[str, ...] = ()
    report_path: Path | None = None

    @property
    def rate(self) -> float:
        return self.resolved / self.submitted if self.submitted else float("nan")


class Benchmark(Protocol):
    name: str
    runner_module: str
    config: Path

    def data_flags(self) -> list[str]: ...

    def provenance(self) -> dict: ...

    def load(self, subset: str | None = None, limit: int | None = None) -> list[str]: ...

    def score(self, out_dir: Path) -> ScoreReport: ...


def subset_ids(sets_file: str, subset: str | None) -> list[str] | None:
    """Instance ids of a named subset in sets/<sets_file>; None means the whole split."""
    if not subset:
        return None
    sets = json.loads((SETS / sets_file).read_text())
    if not isinstance(sets.get(subset), list):
        raise KeyError(f"unknown subset {subset!r} in {sets_file}. "
                       f"Available: {', '.join(k for k, v in sets.items() if isinstance(v, list))}")
    return list(sets[subset])


def pick(available_ids: list[str], sets_file: str, subset: str | None, limit: int | None) -> list[str]:
    """The ids of a named subset (every one of which must be available), or the whole split."""
    wanted = subset_ids(sets_file, subset)
    if wanted is None:
        ids = list(available_ids)
    else:
        missing = [i for i in wanted if i not in set(available_ids)]
        if missing:
            raise KeyError(f"{len(missing)} of the {len(wanted)} tasks of subset {subset!r} are not in the data "
                           f"(first: {missing[0]}): the dataset or the task directory is not the one the set was "
                           "drawn from")
        ids = wanted
    return ids[:limit] if limit else ids
