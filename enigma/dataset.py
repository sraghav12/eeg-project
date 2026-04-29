"""Thin re-export of sharmar/atm/dataset.py so ENIGMA uses the same pipeline.

Rationale: for the report table to be an apples-to-apples comparison of
architectures, ENIGMA must see the *identical* preprocessing, split, and
augmentation as ATM. Rather than duplicate 230 lines of code, we load the
ATM dataset module by absolute file path (under the name ``atm_dataset`` to
avoid colliding with this file's own ``dataset`` module name) and re-export
the public symbols.

If you change preprocessing, change it in sharmar/atm/dataset.py and both
models pick it up.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_ATM_DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "atm", "dataset.py")
)

_spec = importlib.util.spec_from_file_location("atm_dataset", _ATM_DATASET_PATH)
_atm_dataset = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass can resolve cls.__module__ lookups.
sys.modules["atm_dataset"] = _atm_dataset
_spec.loader.exec_module(_atm_dataset)

ArtifactPaths = _atm_dataset.ArtifactPaths
EEGDataset = _atm_dataset.EEGDataset
build_datasets = _atm_dataset.build_datasets
build_subject_index = _atm_dataset.build_subject_index
collate = _atm_dataset.collate
load_artifacts = _atm_dataset.load_artifacts


if __name__ == "__main__":
    tr, va, te, sub_to_idx, n_classes = build_datasets(augment_train=False)
    print(f"train {len(tr):>6} | val {len(va):>6} | test {len(te):>6}")
    print(f"subjects ({len(sub_to_idx)}):", sub_to_idx)
    print(f"n_classes: {n_classes}")
    x, y, s = tr[0]
    print("sample eeg", tuple(x.shape), x.dtype, "label", y, "subject_id", s)
