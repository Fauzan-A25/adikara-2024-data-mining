"""Generate notebook riset Penyisihan Data Mining Adikara 2024 dan script runner untuk Kaggle.

Output:
- kaggle-notebook/adikara-2024-data-mining.ipynb  (notebook bersih untuk di-push sebagai notebook Kaggle)
- kaggle/run_on_kaggle.py                         (script kernel terpisah: mengeksekusi notebook di Kaggle dan
                                                   menyimpan versi ber-output untuk folder notebooks/)
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent
NB_NAME = "adikara-2024-data-mining.ipynb"
FINDINGS = json.loads((ROOT / "_findings.json").read_text()) if (ROOT / "_findings.json").exists() else {}


def T(key: str) -> str:
    return FINDINGS.get(key, "_(temuan diisi setelah run pertama)_")


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ════════════════════════════════════════════════════════════════════ Header
md(
    f"""
# Meramal Indeks Harga Pangan 22 Negara Sepuluh Tahun ke Depan

| | |
| --- | --- |
| **Kompetisi** | [Penyisihan Data Mining Adikara 2024](https://www.kaggle.com/competitions/data-mining-adikara) (Kaggle, Desember 2024) |
| **Tujuan** | Meramal `FoodPriceIndex` bulanan 22 negara untuk 2014–2023 dari data 1970–2013 (metrik SMAPE) |
| **Data** | 11.616 baris latih (22 negara × 528 bulan), 2.640 baris uji (22 negara × 120 bulan) |
| **Penulis** | Fauzan Ahsanudin Alfikri · [Kaggle: fauzan2](https://www.kaggle.com/fauzan2) |
| **Tanggal** | Disusun ulang November 2026 (notebook asli 2024 hilang) |
| **Lingkungan** | Dijalankan penuh di Kaggle (CPU), ±2 menit |

**Pertanyaan penelitian.** Dengan horizon sepanjang 10 tahun, apakah lebih baik meramal setiap negara sendiri-sendiri,
atau meminjam informasi dari negara lain?

**Ringkasan temuan.**
{T("summary")}

**Alur:** 1. Setup → 2. Memahami data → 3. EDA → 4. Preprocessing → 5. Desain validasi → 6. Baseline →
7. Model per negara → 8. Model gabungan lintas negara → 9. Evaluasi → 10. Ramalan akhir & submission → 11. Kesimpulan
"""
)

# ════════════════════════════════════════════════════════════════════ 1. Setup
md(
    """
## 1. Setup

Semua import, parameter, dan path ada di sel ini. Model peramalan dan backtest ada di `src/models/forecasters.py`
agar notebook fokus pada cerita dan hasil.
"""
)

code(
    """
import os
import shutil
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Folder src/ ada di root proyek (lokal) atau di dataset Kaggle "adikara-2024-data-mining-src"
SRC_ROOT = Path.cwd().parent
for r, _, f in os.walk("/kaggle/input", followlinks=True):
    if Path(r).name == "models" and "forecasters.py" in f:
        pkg = Path(r).parent
        if pkg.name == "src":
            SRC_ROOT = pkg.parent
        else:  # isi dataset langsung di root: salin agar bisa diimpor sebagai paket `src`
            SRC_ROOT = Path("/tmp/project")
            shutil.copytree(pkg, SRC_ROOT / "src", dirs_exist_ok=True)
        break
sys.path.insert(0, str(SRC_ROOT))
from src.data.load_data import clean_panel, find_data_dir, load_raw, t_to_date, to_panel
from src.models.forecasters import (backtest, context, fc_damped, fc_damped_seasonal, fc_last, fc_mean12,
                                    log_growth, make_drift, make_ensemble, make_pooled_drift, smape)
from src.visualization.plots import PALETTE, pct_axis, ref_line, save_fig, set_style

# ── Parameter eksperimen
HORIZON = 120                    # 10 tahun × 12 bulan, sama dengan panjang data uji
ORIGINS = [1993, 1998, 2003]     # tahun terakhir data latih pada setiap backtest
POOL_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]

# ── Path (relatif; di Kaggle otomatis menunjuk /kaggle/input dan /kaggle/working)
DATA_DIR = find_data_dir(local="../data/raw")
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("../reports")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
set_style()
C1, C2, C3, CN = PALETTE["normal"], PALETTE["fraud"], PALETTE["accent"], PALETTE["neutral"]
T0 = time.time()
print("DATA_DIR:", DATA_DIR, "| OUT_DIR:", OUT_DIR)
"""
)

# ════════════════════════════════════════════════════════════════════ 2. Memahami data
md(
    """
## 2. Memahami data

Setiap baris adalah satu negara pada satu bulan: `Country`, `Year`, `Month`, dan target `FoodPriceIndex`
(indeks harga pangan, tanpa satuan). Data latih mencakup Januari 1970 sampai Desember 2013; data uji meminta
ramalan **Januari 2014 sampai Desember 2023** untuk 22 negara yang sama. Jadi ini bukan regresi biasa, melainkan
**peramalan deret waktu dengan horizon 120 bulan**.
"""
)

code(
    """
raw = load_raw(DATA_DIR)
train, test, sample = raw["train"], raw["test"], raw["sample"]
summary = pd.DataFrame({
    "baris": [len(train), len(test)],
    "negara": [train.Country.nunique(), test.Country.nunique()],
    "periode": [f"{train.date.min():%b %Y} – {train.date.max():%b %Y}", f"{test.date.min():%b %Y} – {test.date.max():%b %Y}"],
    "bulan per negara": [train.groupby("Country").size().iloc[0], test.groupby("Country").size().iloc[0]],
}, index=["train", "test"])
summary.style.format({"baris": "{:,}"})
"""
)

code(
    """
panel_raw = to_panel(train)
print("Negara:", ", ".join(panel_raw.columns))
print("Negara di test yang tidak ada di train:", set(test.Country) - set(train.Country) or "tidak ada")
"""
)

# ════════════════════════════════════════════════════════════════════ 3. EDA
md(
    """
## 3. Exploratory Data Analysis

### 3.1 Nilai kosong dan nilai nol: apa artinya?
"""
)

code(
    """
missing = pd.DataFrame({
    "NaN": panel_raw.isna().sum(),
    "nilai 0": (panel_raw == 0).sum(),
    "nilai 0 terakhir": [t_to_date([panel_raw.index[(panel_raw[c] == 0).values].max()])[0].strftime("%b %Y")
                         if (panel_raw[c] == 0).any() else "-" for c in panel_raw],
})
"""
)

code(
    """
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
status = np.where(panel_raw.isna(), 1, np.where(panel_raw == 0, 2, 0)).T
cmap = plt.matplotlib.colors.ListedColormap(["#E8EEF4", C2, "#000000"])
ax[0].imshow(status, aspect="auto", cmap=cmap, interpolation="nearest")
yrs = np.arange(1970, 2014, 5)
ax[0].set_xticks([(y - 1970) * 12 for y in yrs]); ax[0].set_xticklabels(yrs)
ax[0].set_yticks(range(len(panel_raw.columns))); ax[0].set_yticklabels(panel_raw.columns, fontsize=8)
ax[0].set_title("Peta data: terisi (biru muda), NaN (oranye), nol (hitam)")
nan_by_year = panel_raw.isna().groupby(panel_raw.index // 12).mean().mean(1)
ax[1].bar(nan_by_year.index, nan_by_year.values, color=C2)
pct_axis(ax[1]); ax[1].set(title="Proporsi NaN per tahun", xlabel="tahun", ylabel="NaN")
save_fig(FIG_DIR, "01_missing_map")
missing[missing["nilai 0"] > 0]
"""
)

md("**Temuan.** " + T("missing"))

md("### 3.2 Seperti apa perjalanan indeks di setiap negara?")

code(
    """
y = clean_panel(panel_raw)
dates = t_to_date(y.index)
fig, axes = plt.subplots(4, 6, figsize=(18, 10), sharex=True)
for ax, c in zip(axes.flat, y.columns):
    ax.semilogy(dates, y[c], color=C1, lw=1)
    ax.set_title(c, fontsize=9); ax.tick_params(labelsize=7)
    ax.axhline(100, color=CN, lw=0.6, ls=":")
for ax in axes.flat[len(y.columns):]:
    ax.axis("off")
fig.suptitle("FoodPriceIndex per negara, 1970–2013 (skala log; garis titik = 100)", fontweight="bold")
save_fig(FIG_DIR, "02_series_grid")
"""
)

code(
    """
level_2013 = y[y.index >= 2013 * 12].mean().sort_values()
level_2013.round(1).to_frame("rata-rata 2013").T
"""
)

md("**Temuan.** " + T("series"))

md(
    """
### 3.3 Apakah laju kenaikan harga stabil dari dekade ke dekade?

Laju pertumbuhan tahunan dihitung dari rata-rata indeks per tahun, lalu dirata-rata per dekade.
"""
)

code(
    """
annual = y.groupby(y.index // 12).mean()
growth = annual.pct_change() * 100
decade = growth.groupby((growth.index // 10) * 10).median().T
decade.columns = [f"{d}-an" for d in decade.columns]
"""
)

code(
    """
plt.figure(figsize=(9, 8))
sns.heatmap(decade.clip(upper=30), annot=decade.round(1), fmt="", cmap="cividis", cbar_kws={"label": "% per tahun (dipotong di 30)"},
            annot_kws={"size": 8})
plt.title("Median pertumbuhan tahunan indeks per dekade")
save_fig(FIG_DIR, "03_growth_by_decade")
"""
)

md("**Temuan.** " + T("decade"))

md(
    """
### 3.4 Apakah ada pola musiman?

Untuk melihat musiman, setiap nilai dibandingkan dengan rata-rata bergerak 12 bulan di sekitarnya. Rata-rata deviasi
per bulan kalender menunjukkan pola yang berulang setiap tahun.
"""
)

code(
    """
recent = np.log(y[y.index >= 1995 * 12])
dev = (recent - recent.rolling(12, center=True).mean()) * 100
seas = dev.groupby(dev.index % 12 + 1).mean()
"""
)

code(
    """
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
for c in seas.columns:
    ax[0].plot(seas.index, seas[c], color=CN, lw=0.7, alpha=0.6)
ax[0].plot(seas.index, seas.median(1), color=C2, lw=2.5, label="median 22 negara")
ax[0].axhline(0, color="black", lw=0.6)
ax[0].set(xticks=range(1, 13), xlabel="bulan", ylabel="deviasi dari tren (%)", title="Profil musiman 1995–2013"); ax[0].legend()
amp = (seas.max() - seas.min()).sort_values()
ax[1].barh(amp.index, amp.values, color=C1)
ax[1].set(xlabel="amplitudo musiman (poin %)", title="Kekuatan musiman per negara")
ax[1].tick_params(axis="y", labelsize=8)
save_fig(FIG_DIR, "04_seasonality")
"""
)

md("**Temuan.** " + T("season"))

md(
    """
### 3.5 Apakah negara-negara bergerak bersama?

Jika pertumbuhan harga pangan di satu negara mirip dengan negara lain, informasi lintas negara bisa dipakai untuk
menstabilkan ramalan. Diukur dengan korelasi pertumbuhan tahunan antar negara pada periode 1995–2013.
"""
)

code(
    """
g_recent = growth.loc[1995:2013]
corr = g_recent.corr()
off_diag = corr.where(~np.eye(len(corr), dtype=bool)).stack()
common = g_recent.median(1)
"""
)

code(
    """
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax[0], xticklabels=True, yticklabels=True)
ax[0].tick_params(labelsize=7); ax[0].set_title("Korelasi pertumbuhan tahunan antar negara, 1995–2013")
for c in g_recent.columns:
    ax[1].plot(g_recent.index, g_recent[c], color=CN, lw=0.6, alpha=0.5)
ax[1].plot(common.index, common.values, color=C2, lw=2.5, label="median 22 negara")
ax[1].set(ylim=(-5, 15), xlabel="tahun", ylabel="pertumbuhan tahunan (%)", title="Pertumbuhan tahunan: tiap negara vs median")
ax[1].legend()
save_fig(FIG_DIR, "05_comovement")
print(f"Median korelasi antar pasangan negara: {off_diag.median():.2f}")
"""
)

md("**Temuan.** " + T("comove"))

# ════════════════════════════════════════════════════════════════════ 4. Preprocessing
md(
    """
## 4. Preprocessing

| Temuan EDA | Keputusan |
| --- | --- |
| Nilai 0 menandai seri yang belum dimulai (3.1) | 0 diubah menjadi NaN dan tidak dipakai untuk melatih maupun mengevaluasi |
| NaN tersebar acak di tengah seri (3.1) | diisi interpolasi linear untuk melatih; tidak diisi saat mengevaluasi |
| Pertumbuhan bersifat multiplikatif dan berbeda jauh antar era (3.2, 3.3) | model bekerja pada skala log, dan hanya memakai 3–15 tahun terakhir |
| Musiman kecil tetapi ada (3.4) | diuji sebagai komponen opsional |
| Negara bergerak bersama (3.5) | diuji model yang meminjam laju pertumbuhan dari negara lain |
"""
)

code(
    """
panel = clean_panel(panel_raw)
print("NaN tersisa setelah interpolasi (hanya awal seri yang belum dimulai):", int(panel.isna().sum().sum()))
"""
)

# ════════════════════════════════════════════════════════════════════ 5. Validasi
md(
    """
## 5. Desain validasi: backtest rolling-origin

Data uji adalah 10 tahun **setelah** data latih berakhir, jadi validasi harus meniru situasi itu. K-Fold acak tidak
cocok karena akan memakai masa depan untuk meramal masa lalu. Digunakan **tiga backtest**: model dilatih dengan data
sampai akhir 1993, 1998, dan 2003, lalu meramal 120 bulan berikutnya. Skor akhir adalah rata-rata SMAPE ketiganya.
"""
)

code(
    """
fig, ax = plt.subplots(figsize=(11, 2.4))
for i, o in enumerate(ORIGINS + [2013]):
    ax.barh(i, (o + 1) - 1970, left=1970, color=C1, height=0.5)
    ax.barh(i, 10, left=o + 1, color=C2 if o != 2013 else C3, height=0.5)
ax.set_yticks(range(len(ORIGINS) + 1)); ax.set_yticklabels([f"backtest {o}" for o in ORIGINS] + ["ramalan akhir"])
ax.set(xlabel="tahun", title="Latih (biru) dan ramal 10 tahun (oranye: dievaluasi, hijau: data uji kompetisi)")
save_fig(FIG_DIR, "06_backtest_design")
"""
)

# ════════════════════════════════════════════════════════════════════ 6. Baseline
md(
    """
## 6. Baseline

Dua baseline paling sederhana: **nilai terakhir** dan **rata-rata 12 bulan terakhir**, yaitu menganggap harga
tidak berubah selama 10 tahun.
"""
)

code(
    """
MODELS = {"nilai terakhir": fc_last, "rata-rata 12 bulan": fc_mean12}
bt_base = backtest(panel, panel_raw, MODELS, ORIGINS, HORIZON)
bt_base.pivot_table(index="model", columns="origin", values="smape").assign(rata_rata=lambda d: d.mean(1)).style.format("{:.2f}")
"""
)

md("**Temuan.** " + T("baseline"))

# ════════════════════════════════════════════════════════════════════ 7. Model per negara
md(
    """
## 7. Model per negara

Tiga keluarga model, masing-masing dilatih terpisah untuk setiap negara pada skala log:

- **Drift**: melanjutkan tren log linear dari 3, 5, atau 10 tahun terakhir.
- **Damped trend (Holt)**: tren yang melemah perlahan, cocok untuk horizon panjang.
- **Damped trend + musiman**: sama, dengan komponen musiman 12 bulan (dari temuan 3.4).
"""
)

code(
    """
MODELS_LOCAL = {
    "drift 3 tahun": make_drift(3), "drift 5 tahun": make_drift(5), "drift 10 tahun": make_drift(10),
    "damped trend": fc_damped, "damped trend + musiman": fc_damped_seasonal,
}
bt_local = backtest(panel, panel_raw, MODELS_LOCAL, ORIGINS, HORIZON)
bt_local.pivot_table(index="model", columns="origin", values="smape").assign(rata_rata=lambda d: d.mean(1)) \\
    .sort_values("rata_rata").style.format("{:.2f}").highlight_min(subset=["rata_rata"], color="#CDE7F0")
"""
)

md("**Temuan.** " + T("local"))

# ════════════════════════════════════════════════════════════════════ 8. Model gabungan
md(
    """
## 8. Model gabungan lintas negara

Temuan 3.5 menunjukkan negara-negara bergerak bersama. Laju pertumbuhan satu negara yang diestimasi dari 5 tahun
saja rentan noise, sehingga laju itu **disusutkan** ke median laju semua negara:

$$g_{\\text{akhir}} = (1-w)\\cdot g_{\\text{negara}} + w\\cdot g_{\\text{median}}$$

$w = 0$ berarti setiap negara berdiri sendiri, $w = 1$ berarti semua negara memakai laju yang sama.
Median dihitung hanya dari data sebelum origin, jadi tidak ada kebocoran.
"""
)

code(
    """
MODELS_POOL = {f"pooled w={w:.2f}": make_pooled_drift(w) for w in POOL_WEIGHTS}
bt_pool = backtest(panel, panel_raw, MODELS_POOL, ORIGINS, HORIZON)
pool_tab = bt_pool.groupby("model").smape.mean()
pool_tab.to_frame("SMAPE rata-rata").T.style.format("{:.2f}")
"""
)

code(
    """
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(POOL_WEIGHTS, pool_tab.values, marker="o", color=C2)
best_w = POOL_WEIGHTS[int(np.argmin(pool_tab.values))]
ax.axvline(best_w, color=CN, ls="--", label=f"terbaik: w = {best_w}")
ax.set(xlabel="bobot penyusutan w (0 = per negara, 1 = sama untuk semua)", ylabel="SMAPE backtest", title="Seberapa banyak meminjam dari negara lain?")
ax.legend()
save_fig(FIG_DIR, "07_pooling_weight")
"""
)

md("**Temuan.** " + T("pool"))

md(
    """
### 8.1 Ensemble

Model gabungan (tren yang distabilkan) dan damped trend + musiman (menangkap pola musiman dan pelemahan tren)
membuat kesalahan yang berbeda, sehingga rata-rata keduanya diuji sebagai model akhir.
"""
)

code(
    """
POOL_BEST = make_pooled_drift(best_w)
MODELS_ENS = {
    "ensemble: pooled + damped musiman": make_ensemble(POOL_BEST, fc_damped_seasonal),
    "ensemble: pooled + damped": make_ensemble(POOL_BEST, fc_damped),
    "ensemble: pooled + damped musiman + drift 3 tahun": make_ensemble(POOL_BEST, fc_damped_seasonal, make_drift(3)),
}
bt_ens = backtest(panel, panel_raw, MODELS_ENS, ORIGINS, HORIZON)
bt_all = pd.concat([bt_base, bt_local, bt_pool[bt_pool.model == f"pooled w={best_w:.2f}"], bt_ens])
leader = bt_all.pivot_table(index="model", columns="origin", values="smape").assign(rata_rata=lambda d: d.mean(1)).sort_values("rata_rata")
leader.style.format("{:.2f}").background_gradient(subset=["rata_rata"], cmap="cividis_r")
"""
)

md("**Temuan.** " + T("ensemble"))

# ════════════════════════════════════════════════════════════════════ 9. Evaluasi
md(
    """
## 9. Evaluasi

### 9.1 Bagaimana error berkembang sepanjang horizon?
"""
)

code(
    """
FINAL_NAME = leader.index[0]
show = [FINAL_NAME, f"pooled w={best_w:.2f}", "damped trend + musiman", "drift 5 tahun", "nilai terakhir"]
by_year = (bt_all[bt_all.model.isin(show)]
           .assign(y=lambda d: d.smape_by_year.apply(pd.Series).values.tolist())
           .pipe(lambda d: pd.DataFrame([{"model": m, "tahun ke-": k + 1, "smape": v}
                                          for m, yrs in zip(d.model, d.smape_by_year) for k, v in yrs.items()]))
           .groupby(["model", "tahun ke-"]).smape.mean().unstack(0))
"""
)

code(
    """
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
colors = [C2, C3, C1, "#CC79A7", CN]
for m, col in zip(show, colors):
    ax[0].plot(by_year.index, by_year[m], marker="o", ms=3, label=m, color=col, lw=2.5 if m == FINAL_NAME else 1.3)
ax[0].set(xlabel="tahun ke- setelah data latih", ylabel="SMAPE", title="Error menurut jarak ramalan"); ax[0].legend(fontsize=8)
per_c = bt_all[bt_all.model == FINAL_NAME].groupby("country").smape.mean().sort_values()
ax[1].barh(per_c.index, per_c.values, color=C1); ax[1].tick_params(axis="y", labelsize=8)
ax[1].set(xlabel="SMAPE backtest", title="Error model akhir per negara")
save_fig(FIG_DIR, "08_error_by_horizon")
"""
)

md("**Temuan.** " + T("horizon"))

md("### 9.2 Contoh ramalan backtest: apa yang terlihat di lapangan?")

code(
    """
cut = (ORIGINS[-1] + 1) * 12
ctx = context(panel, cut)
examples = [per_c.index[0], per_c.index[len(per_c) // 2], per_c.index[-1]]
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, c in zip(axes, examples):
    s = ctx["hist"][c]; idx = range(cut, cut + HORIZON)
    ax.plot(t_to_date(s.index[-180:]), s.iloc[-180:], color=CN, label="latih")
    act = panel_raw[c].reindex(idx).replace(0, np.nan)
    ax.plot(t_to_date(list(idx)), act.values, color="black", lw=1, label="aktual")
    ax.plot(t_to_date(list(idx)), MODELS_ENS[FINAL_NAME](s, HORIZON, ctx) if FINAL_NAME in MODELS_ENS else POOL_BEST(s, HORIZON, ctx),
            color=C2, lw=2, label="ramalan model akhir")
    ax.plot(t_to_date(list(idx)), fc_last(s, HORIZON), color=C1, ls="--", label="nilai terakhir")
    ax.set_title(f"{c} (backtest 2003)"); ax.legend(fontsize=7)
save_fig(FIG_DIR, "09_backtest_examples")
"""
)

md("**Temuan.** " + T("examples"))

# ════════════════════════════════════════════════════════════════════ 10. Ramalan akhir
md(
    """
## 10. Ramalan akhir dan submission

Model akhir dilatih ulang dengan seluruh data 1970–2013, lalu meramal Januari 2014 sampai Desember 2023.
"""
)

code(
    """
final_model = MODELS_ENS.get(FINAL_NAME) or {**MODELS_LOCAL, **MODELS_POOL}.get(FINAL_NAME, POOL_BEST)
cut = 2014 * 12
ctx = context(panel, cut)
fc_rows = []
for c in panel.columns:
    f = final_model(panel[c].dropna(), HORIZON, ctx)
    fc_rows += [{"Country": c, "t": cut + i, "FoodPriceIndex": v} for i, v in enumerate(f)]
forecast = pd.DataFrame(fc_rows)
sub = test.merge(forecast, on=["Country", "t"], how="left")[["id", "FoodPriceIndex"]]
assert sub.FoodPriceIndex.notna().all() and len(sub) == len(sample)
sub = sample[["id"]].merge(sub, on="id", how="left")
sub.to_csv(OUT_DIR / "submission.csv", index=False)
print("Model akhir:", FINAL_NAME)
sub.FoodPriceIndex.describe().to_frame("ramalan").T.style.format("{:.2f}")
"""
)

code(
    """
fc_panel = forecast.pivot(index="t", columns="Country", values="FoodPriceIndex")
fig, axes = plt.subplots(4, 6, figsize=(18, 10), sharex=True)
for ax, c in zip(axes.flat, panel.columns):
    hist = panel[c][panel.index >= 2000 * 12]
    ax.plot(t_to_date(hist.index), hist, color=C1, lw=1)
    ax.plot(t_to_date(fc_panel.index), fc_panel[c], color=C2, lw=1.5)
    ax.set_title(c, fontsize=9); ax.tick_params(labelsize=7)
for ax in axes.flat[len(panel.columns):]:
    ax.axis("off")
fig.suptitle("Data 2000–2013 (biru) dan ramalan 2014–2023 (oranye)", fontweight="bold")
save_fig(FIG_DIR, "10_final_forecast")
leader.to_csv(OUT_DIR / "backtest_leaderboard.csv")
by_year.to_csv(OUT_DIR / "smape_by_horizon_year.csv")
"""
)

md("**Temuan.** " + T("final"))

# ════════════════════════════════════════════════════════════════════ 11. Kesimpulan
md("## 11. Kesimpulan\n\n" + T("conclusion") + "\n\n### Jejak lingkungan")

code(
    """
import platform
pkgs = ["numpy", "pandas", "statsmodels", "matplotlib", "seaborn"]
print(" | ".join([f"python {platform.python_version()}"] + [f"{p} {version(p)}" for p in pkgs]))
print(f"Waktu eksekusi total: {(time.time() - T0) / 60:.1f} menit")
"""
)

nb["cells"] = cells
nb_path = ROOT / "kaggle-notebook" / NB_NAME
nb_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, nb_path)
print("ditulis:", nb_path)

# ════════════════════════════════════════════════════════════════════ Kaggle runner
src_files = {str(p.relative_to(ROOT)): p.read_text() for p in sorted((ROOT / "src").rglob("*.py"))}
runner = f'''"""Runner Kaggle untuk {NB_NAME} (di-generate oleh _build_notebook.py; jangan diedit manual)."""
import json, shutil, subprocess, sys
from pathlib import Path

SRC_FILES = {src_files!r}
NOTEBOOK = json.loads({nbf.writes(nb)!r})

proj = Path("/kaggle/working/project")
for rel, text in SRC_FILES.items():
    p = proj / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
(proj / "notebooks").mkdir(parents=True, exist_ok=True)
nb_in = proj / "notebooks" / "{NB_NAME}"
nb_in.write_text(json.dumps(NOTEBOOK))
cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", str(nb_in),
       "--output", "{NB_NAME}", "--output-dir", "/kaggle/working", "--ExecutePreprocessor.timeout=-1"]
ret = subprocess.run(cmd, cwd=proj / "notebooks")
shutil.rmtree(proj, ignore_errors=True)
sys.exit(ret.returncode)
'''
(ROOT / "kaggle" / "run_on_kaggle.py").write_text(runner)
print("ditulis:", ROOT / "kaggle" / "run_on_kaggle.py")
