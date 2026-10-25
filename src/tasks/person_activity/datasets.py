# Person-activity dataset loader (UCI "Localization Data for Person
# Activity", ConfLongDemo_JSI.txt).
#
# Parsing logic adapted from classification/irregular_sampled_datasets.py
# (ODE-LSTM authors, https://github.com/mlech26l/ode-lstms), with these changes:
#   - the data path is configurable and defaults to
#     <repo>/data/person/ConfLongDemo_JSI.txt resolved relative to the repo
#     root (the original hardcodes a CWD-relative "../data/..." path);
#   - a missing file raises FileNotFoundError instead of sys.exit(-1);
#   - all 25 recordings are kept (the original drops the last one) and the
#     few negative elapsed times (out-of-order timestamps) are clamped to 0.
# The windowing (length seq_len, stride seq_len // 2, inside one recording) is
# unchanged. split='grouped' (default) assigns whole recordings to
# train/val/test; split='random_window' is the original protocol (seed 98841,
# 20% of the windows as test, no validation set), kept as literature anchor:
# overlapping windows of the same recording land on both sides there.

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

# datasets.py -> person_activity -> tasks -> src -> <repo root>
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_DATA_PATH = os.path.join(_REPO_ROOT, "data", "person",
                                 "ConfLongDemo_JSI.txt")

NUM_CLASSES = 7
SPLITS = ("grouped", "random_window")

# Activity label -> class id (several raw labels share a class).
CLASS_MAP = {
    "lying down": 0,
    "lying": 0,
    "sitting down": 1,
    "sitting": 1,
    "standing up from lying": 2,
    "standing up from sitting": 2,
    "standing up from sitting on the ground": 2,
    "walking": 3,
    "falling": 4,
    "on all fours": 5,
    "sitting on the ground": 6,
}

# Tag id -> one-hot slot (4 body-worn sensors).
SENSOR_IDS = {
    "010-000-024-033": 0,
    "010-000-030-096": 1,
    "020-000-033-111": 2,
    "020-000-032-221": 3,
}


@dataclass
class PersonActivityData:
    """Train/val/test arrays plus metadata for the person-activity task.

    Shapes (N = number of sequences):
        *_x:   (N, seq_len, feature_size)  float32 features
        *_t:   (N, seq_len, 1)             float32 elapsed time between samples
        *_y:   (N, seq_len)                int32 class labels
        *_rec: (N,)                        recording id of each window ('A01')
    split='random_window' has no validation set: val_* are the test_* arrays.
    recordings: {'train'|'val'|'test': [recording ids]} for split='grouped',
    None for 'random_window' (every recording feeds both sides).
    """
    train_x: np.ndarray
    train_t: np.ndarray
    train_y: np.ndarray
    train_rec: np.ndarray
    val_x: np.ndarray
    val_t: np.ndarray
    val_y: np.ndarray
    val_rec: np.ndarray
    test_x: np.ndarray
    test_t: np.ndarray
    test_y: np.ndarray
    test_rec: np.ndarray
    feature_size: int
    num_classes: int
    seq_len: int
    split: str
    split_seed: Optional[int]
    recordings: Optional[dict]
    n_recordings: int
    n_negative_dt: int


def _parse_raw(data_path):
    """Parse the raw CSV into per-recording (features, elapsed, labels) series.

    Row format: recording_id, sensor_tag, millis, date_string, x, y, z, label.
    The recording id is person letter + recording number ('A01' .. 'E05').
    Feature vector = one-hot(sensor, 4) ++ (x, y, z) -> feature_size 7.
    Elapsed time is the gap to the previous row in units of 100 ms
    (100 ms -> 1.0), first row of each recording fixed at 0.05; negative gaps
    are clamped to 0.

    Returns:
        (names, all_x, all_t, all_y, n_negative_dt), one list entry per
        recording in file order.
    """
    names, all_x, all_t, all_y = [], [], [], []
    series_x, series_t, series_y = [], [], []
    n_negative_dt = 0

    def enqueue(name):
        names.append(name)
        all_x.append(np.stack(series_x, axis=0))
        all_t.append(np.stack(series_t, axis=0))
        all_y.append(np.array(series_y, dtype=np.int32))

    last_millis = None
    with open(data_path, "r") as f:
        current_rec = None
        for line in f:
            arr = line.split(",")
            if len(arr) < 6:
                break
            if current_rec is not None and arr[0] != current_rec:
                # Enqueue the finished recording's series and reset.
                enqueue(current_rec)
                last_millis = None
                series_x, series_t, series_y = [], [], []

            millis = np.int64(arr[2]) / (100 * 1000)
            # 100 ms is normalized to 1.0
            millis_mapped_to_1 = 10.0
            if last_millis is None:
                elapsed_sec = 0.05
            else:
                elapsed_sec = float(millis - last_millis) / 1000.0
            elapsed = elapsed_sec * 1000 / millis_mapped_to_1
            if elapsed < 0:
                # Out-of-order timestamps in the raw file.
                n_negative_dt += 1
                elapsed = 0.0

            last_millis = millis
            current_rec = arr[0]
            sensor_onehot = np.zeros(4, dtype=np.float32)
            sensor_onehot[SENSOR_IDS[arr[1]]] = 1
            xyz = np.array(arr[4:7], dtype=np.float32)

            series_x.append(np.concatenate([sensor_onehot, xyz]))
            series_t.append(elapsed)
            series_y.append(CLASS_MAP[arr[7].replace("\n", "")])
    if series_x:
        enqueue(current_rec)

    return names, all_x, all_t, all_y, n_negative_dt


def _cut_in_sequences(all_x, all_t, all_y, seq_len, inc):
    """Slice each series into overlapping windows of length seq_len.

    Windows never cross a series boundary; the fourth return value holds the
    index of the source series of every window.
    """
    sequences_x, sequences_t, sequences_y, source = [], [], [], []
    for i, (x, t, y) in enumerate(zip(all_x, all_t, all_y)):
        for start in range(0, x.shape[0] - seq_len, inc):
            end = start + seq_len
            sequences_x.append(x[start:end])
            sequences_t.append(t[start:end])
            sequences_y.append(y[start:end])
            source.append(i)
    return (
        np.stack(sequences_x, axis=0),
        np.stack(sequences_t, axis=0).reshape([-1, seq_len, 1]),
        np.stack(sequences_y, axis=0),
        np.array(source),
    )


def _assign_recordings(names, split_seed):
    """Grouped split: whole recordings to train/val/test, balanced per person.

    A seeded permutation of each person's recordings puts the last one into
    test, the one before into val and the rest into train (5 recordings per
    person -> 3/1/1, 15/5/5 overall).
    """
    rng = np.random.default_rng(split_seed)
    parts = {"train": [], "val": [], "test": []}
    for person in sorted({n[0] for n in names}):
        recs = rng.permutation(sorted(n for n in names if n[0] == person))
        parts["train"] += recs[:-2].tolist()
        parts["val"].append(str(recs[-2]))
        parts["test"].append(str(recs[-1]))
    return parts


def load_person_activity(seq_len=32, data_path=None, split="grouped",
                         split_seed=0):
    """Load the person-activity dataset as windowed train/val/test splits.

    Args:
        seq_len:    window length in timesteps (default 32; windows overlap
                    with stride seq_len // 2, as in the original runner).
        data_path:  path to ConfLongDemo_JSI.txt; defaults to
                    <repo>/data/person/ConfLongDemo_JSI.txt.
        split:      'grouped' (whole recordings per side, see
                    _assign_recordings) or 'random_window' (original
                    protocol, literature anchor only: windows permuted with
                    seed 98841, 20% test, val = test).
        split_seed: seeds the grouped recording assignment (fixed for the
                    whole suite); ignored for 'random_window'.

    Returns:
        PersonActivityData with train/val/test x, t, y, rec and metadata.

    Raises:
        FileNotFoundError: if the data file is missing (run
        download_dataset.sh in the repo root to fetch it).
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    data_path = data_path or DEFAULT_DATA_PATH
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Person-activity data file not found: {data_path}. "
            "Run 'source download_dataset.sh' in the repo root to fetch it.")

    names, all_x, all_t, all_y, n_negative_dt = _parse_raw(data_path)
    all_x, all_t, all_y, source = _cut_in_sequences(all_x, all_t, all_y,
                                                    seq_len=seq_len,
                                                    inc=seq_len // 2)
    all_rec = np.array(names)[source]

    if split == "grouped":
        recordings = _assign_recordings(names, split_seed)
        idx = {part: np.flatnonzero(np.isin(all_rec, recs))
               for part, recs in recordings.items()}
    else:
        # Fixed permutation and 20% test share of the original PersonData.
        recordings, split_seed = None, None
        total_seqs = all_x.shape[0]
        permutation = np.random.RandomState(98841).permutation(total_seqs)
        test_size = int(0.2 * total_seqs)
        idx = {"train": permutation[test_size:],
               "val": permutation[:test_size],
               "test": permutation[:test_size]}

    arrays = dict(x=all_x, t=all_t, y=all_y, rec=all_rec)
    return PersonActivityData(
        **{f"{part}_{k}": a[i] for part, i in idx.items()
           for k, a in arrays.items()},
        feature_size=int(all_x.shape[-1]),
        num_classes=NUM_CLASSES,
        seq_len=seq_len,
        split=split,
        split_seed=split_seed,
        recordings=recordings,
        n_recordings=len(names),
        n_negative_dt=n_negative_dt,
    )
