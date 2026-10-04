# Person-activity dataset loader (UCI "Localization Data for Person
# Activity", ConfLongDemo_JSI.txt).
#
# Parsing logic adapted from classification/irregular_sampled_datasets.py
# (ODE-LSTM authors, https://github.com/mlech26l/ode-lstms), with two changes:
#   - the data path is configurable and defaults to
#     <repo>/data/person/ConfLongDemo_JSI.txt resolved relative to the repo
#     root (the original hardcodes a CWD-relative "../data/..." path);
#   - a missing file raises FileNotFoundError instead of sys.exit(-1).
# The split (seed 98841, 20% test) and the windowing (stride seq_len // 2)
# are kept identical so results stay comparable to the original runner.

import os
from dataclasses import dataclass

import numpy as np

# datasets.py -> person_activity -> tasks -> src -> <repo root>
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_DATA_PATH = os.path.join(_REPO_ROOT, "data", "person",
                                 "ConfLongDemo_JSI.txt")

NUM_CLASSES = 7

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
    """Train/test arrays plus metadata for the person-activity task.

    Shapes (N = number of sequences):
        *_x: (N, seq_len, feature_size)  float32 features
        *_t: (N, seq_len, 1)             float32 elapsed time between samples
        *_y: (N, seq_len)                int32 class labels
    """
    train_x: np.ndarray
    train_t: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_t: np.ndarray
    test_y: np.ndarray
    feature_size: int
    num_classes: int
    seq_len: int


def _parse_raw(data_path):
    """Parse the raw CSV into per-person (features, elapsed, labels) series.

    Row format: person_id, sensor_tag, millis, date_string, x, y, z, label.
    Feature vector = one-hot(sensor, 4) ++ (x, y, z) -> feature_size 7.
    Elapsed time is the gap to the previous row in units of 100 ms
    (100 ms -> 1.0), first row of each person fixed at 0.05.
    """
    all_x, all_t, all_y = [], [], []
    series_x, series_t, series_y = [], [], []

    last_millis = None
    with open(data_path, "r") as f:
        current_person = "A01"
        for line in f:
            arr = line.split(",")
            if len(arr) < 6:
                break
            if arr[0] != current_person:
                # Enqueue the finished person's series and reset.
                all_x.append(np.stack(series_x, axis=0))
                all_t.append(np.stack(series_t, axis=0))
                all_y.append(np.array(series_y, dtype=np.int32))
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

            last_millis = millis
            current_person = arr[0]
            sensor_onehot = np.zeros(4, dtype=np.float32)
            sensor_onehot[SENSOR_IDS[arr[1]]] = 1
            xyz = np.array(arr[4:7], dtype=np.float32)

            series_x.append(np.concatenate([sensor_onehot, xyz]))
            series_t.append(elapsed)
            series_y.append(CLASS_MAP[arr[7].replace("\n", "")])

    return all_x, all_t, all_y


def _cut_in_sequences(all_x, all_t, all_y, seq_len, inc):
    """Slice per-person series into overlapping windows of length seq_len."""
    sequences_x, sequences_t, sequences_y = [], [], []
    for x, t, y in zip(all_x, all_t, all_y):
        for start in range(0, x.shape[0] - seq_len, inc):
            end = start + seq_len
            sequences_x.append(x[start:end])
            sequences_t.append(t[start:end])
            sequences_y.append(y[start:end])
    return (
        np.stack(sequences_x, axis=0),
        np.stack(sequences_t, axis=0).reshape([-1, seq_len, 1]),
        np.stack(sequences_y, axis=0),
    )


def load_person_activity(seq_len=32, data_path=None):
    """Load the person-activity dataset as windowed train/test splits.

    Args:
        seq_len:   window length in timesteps (default 32; windows overlap
                   with stride seq_len // 2, as in the original runner).
        data_path: path to ConfLongDemo_JSI.txt; defaults to
                   <repo>/data/person/ConfLongDemo_JSI.txt.

    Returns:
        PersonActivityData with train/test x, t, y and metadata.

    Raises:
        FileNotFoundError: if the data file is missing (run
        download_dataset.sh in the repo root to fetch it).
    """
    data_path = data_path or DEFAULT_DATA_PATH
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Person-activity data file not found: {data_path}. "
            "Run 'source download_dataset.sh' in the repo root to fetch it.")

    all_x, all_t, all_y = _parse_raw(data_path)
    all_x, all_t, all_y = _cut_in_sequences(all_x, all_t, all_y,
                                            seq_len=seq_len,
                                            inc=seq_len // 2)

    # Fixed permutation and 20% test share, identical to the original
    # PersonData so splits (and thus reported accuracies) stay comparable.
    total_seqs = all_x.shape[0]
    permutation = np.random.RandomState(98841).permutation(total_seqs)
    test_size = int(0.2 * total_seqs)

    return PersonActivityData(
        train_x=all_x[permutation[test_size:]],
        train_t=all_t[permutation[test_size:]],
        train_y=all_y[permutation[test_size:]],
        test_x=all_x[permutation[:test_size]],
        test_t=all_t[permutation[:test_size]],
        test_y=all_y[permutation[:test_size]],
        feature_size=int(all_x.shape[-1]),
        num_classes=NUM_CLASSES,
        seq_len=seq_len,
    )
