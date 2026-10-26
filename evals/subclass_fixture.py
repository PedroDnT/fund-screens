"""A clean universe in which some classes report per SUBCLASS, as CVM does.

Real June 2026 informe (measured): 221 CNPJs (0.9% of funds) have 2-6 rows per day; for them the class has NO
class-level row (10,873 of 10,876 rows carry an ID_SUBCLASSE) and each subclass has its own quota level
(e.g. 1.2, 0.12, 0.0083), PL, flows and NR_COTST. evals/fixtures.py models none of this (no ID_SUBCLASSE column,
exactly one row per fund-day), so no eval can see a detector fed alternating series.

`write_cvm_informe_csv` writes the clean universe in the raw CVM shape (real headers, formatted CNPJ, subclass rows);
reading it back through DataProcessor.read_informe_diario exercises the reader exactly as production does, so the
test flips from red to green only when the reader (or something upstream of the detectors) handles subclasses.
Nothing is fraudulent in it: any flag on a split fund is a false positive by construction.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

CVM_HEADER = ["TP_FUNDO_CLASSE", "CNPJ_FUNDO_CLASSE", "ID_SUBCLASSE", "DT_COMPTC", "VL_TOTAL",
              "VL_QUOTA", "VL_PATRIM_LIQ", "CAPTC_DIA", "RESG_DIA", "NR_COTST"]


def _fmt_cnpj(d: str) -> str:
    return f"{d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:]}"


def split_into_subclasses(informe: pd.DataFrame, share: float = 0.25, seed: int = 11,
                          kmin: int = 2, kmax: int = 5) -> tuple[pd.DataFrame, set[str]]:
    """Return (raw CVM-shaped informe with subclass rows, CNPJs that were split)."""
    rng = np.random.default_rng(seed)
    cnpjs = sorted(informe["CNPJ_FUNDO"].unique())
    n_split = max(1, int(round(share * len(cnpjs))))
    split = set(rng.choice(cnpjs, size=n_split, replace=False))
    out = []
    for cnpj, g in informe.sort_values(["CNPJ_FUNDO", "DT_COMPTC"]).groupby("CNPJ_FUNDO", sort=True):
        g = g.reset_index(drop=True)
        n = len(g)
        if cnpj not in split:
            out.append(pd.DataFrame({
                "TP_FUNDO_CLASSE": "CLASSES - FIF", "CNPJ_FUNDO_CLASSE": _fmt_cnpj(cnpj), "ID_SUBCLASSE": "",
                "DT_COMPTC": g["DT_COMPTC"].dt.strftime("%Y-%m-%d"), "VL_TOTAL": g["VL_PATRIM_LIQ"],
                "VL_QUOTA": g["VL_QUOTA"], "VL_PATRIM_LIQ": g["VL_PATRIM_LIQ"], "CAPTC_DIA": g["CAPTC_DIA"],
                "RESG_DIA": g["RESG_DIA"], "NR_COTST": g["NR_COTST"], "_o": np.arange(n), "_k": 0}))
            continue
        k = int(rng.integers(kmin, kmax + 1))
        # stable PL weights with slow drift; quota levels span orders of magnitude, as in the real file
        w0 = rng.dirichlet(np.full(k, 1.5))
        drift = np.cumsum(rng.normal(0, 0.002, (n, k)), axis=0)
        w = w0 * np.exp(drift)
        w /= w.sum(axis=1, keepdims=True)
        level = np.exp(rng.normal(0, 2.0, k))
        fee = (1 - rng.uniform(0, 3e-5, k)) ** np.arange(n)[:, None]
        for j in range(k):
            share_c = rng.dirichlet(np.full(k, 1.0), size=n)[:, j]      # flows: independent per subclass, sum preserved
            share_r = rng.dirichlet(np.full(k, 1.0), size=n)[:, j]
            pl_j = g["VL_PATRIM_LIQ"].to_numpy() * w[:, j]
            out.append(pd.DataFrame({
                "TP_FUNDO_CLASSE": "CLASSES - FIF", "CNPJ_FUNDO_CLASSE": _fmt_cnpj(cnpj),
                "ID_SUBCLASSE": f"SUB{cnpj[-6:]}{j}", "DT_COMPTC": g["DT_COMPTC"].dt.strftime("%Y-%m-%d"),
                "VL_TOTAL": pl_j, "VL_QUOTA": g["VL_QUOTA"].to_numpy() * level[j] * fee[:, j],
                "VL_PATRIM_LIQ": pl_j, "CAPTC_DIA": g["CAPTC_DIA"].to_numpy() * share_c,
                "RESG_DIA": g["RESG_DIA"].to_numpy() * share_r,
                "NR_COTST": np.maximum(1, np.floor(g["NR_COTST"].to_numpy() * w[:, j])).astype(int),
                "_o": np.arange(n), "_k": j}))
    raw = pd.concat(out, ignore_index=True)
    # real file order inside a CNPJ: date, then subclass
    raw = raw.sort_values(["CNPJ_FUNDO_CLASSE", "_o", "_k"], kind="stable").drop(columns=["_o", "_k"])
    return raw[CVM_HEADER], split


def write_cvm_informe_csv(raw: pd.DataFrame, path: Path) -> Path:
    raw.to_csv(path, sep=";", index=False, encoding="latin1")
    return path
