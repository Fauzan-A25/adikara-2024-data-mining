"""Load dan susun data Penyisihan Data Mining Adikara 2024 (indeks harga pangan bulanan 22 negara)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def find_data_dir(marker: str = "train_adikara2024.csv", local: str | Path = "../data/raw") -> Path:
    """Cari folder data di Kaggle (/kaggle/input, termasuk symlink); jika tidak ada, pakai folder lokal."""
    for root, _, files in os.walk("/kaggle/input", followlinks=True):
        if marker in files:
            return Path(root)
    return Path(local)


def load_raw(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    out = {
        "train": pd.read_csv(d / "train_adikara2024.csv"),
        "test": pd.read_csv(d / "test_adikara2024_unlabeled.csv"),
        "sample": pd.read_csv(d / "sample_submission_adikara2024.csv"),
    }
    for k in ("train", "test"):
        out[k]["t"] = out[k].Year * 12 + out[k].Month - 1  # indeks bulan kontinu
        out[k]["date"] = pd.to_datetime(dict(year=out[k].Year, month=out[k].Month, day=1))
    return out


def to_panel(train: pd.DataFrame) -> pd.DataFrame:
    """Tabel lebar: baris = bulan (t), kolom = negara, nilai = FoodPriceIndex mentah."""
    return train.pivot(index="t", columns="Country", values="FoodPriceIndex").sort_index()


def clean_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Nilai 0 (seri belum dimulai) menjadi NaN; NaN di tengah seri diisi interpolasi linear."""
    return panel.replace(0, np.nan).interpolate(limit_area="inside")


def t_to_date(t) -> pd.DatetimeIndex:
    t = np.asarray(t)
    return pd.to_datetime(dict(year=t // 12, month=t % 12 + 1, day=1))
