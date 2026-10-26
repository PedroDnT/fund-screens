"""Regression guard: a clean fund that reports per SUBCLASS must not be flagged more than one that does not.

evals/fixtures.py gives every fund exactly one row per day, so it cannot see this failure. Real CVM informe
(June 2026) has 221 CNPJs (0.9% of funds) with 2-6 subclass rows per day and no class-level row; each subclass
has its own quota, PL, flows and NR_COTST. A reader that treats those rows as one fund's series makes PL, quota
and holder counts alternate between subclass series, which is what produced the 'Priority review' flags on funds
such as Kinea's. Nothing is injected here, so every flag is a false positive by construction.

The informe goes through DataProcessor.read_informe_diario (a CVM-shaped CSV with ID_SUBCLASSE), because the
fix belongs in or before the reader; a fixture that bypassed the reader could never turn green.

    pytest -q -m eval tests/test_eval_subclass.py
"""
import pytest
from types import SimpleNamespace

from evals import fixtures, subclass_fixture as sf
from evals.run_eval import DETECTORS
from src.processors.data_processor import DataProcessor

pytestmark = pytest.mark.eval

# One-month window (the dashboard's), 80 funds, a quarter of them split into 2-5 subclasses.
N_FUNDS, N_DAYS, SHARE = 80, 21, 0.25
SLACK = 0.15          # a split fund may be flagged this much more often than an unsplit one (finite sample)
# Detectors that read the informe series and are meaningful on a clean synthetic universe.
SIGNALS = ["pl_drop", "quotaholder_exodus", "valuation_smoothing", "redemption_run", "window_dressing"]


@pytest.fixture(scope="module")
def universes(tmp_path_factory):
    clean = fixtures.make_clean_universe(n_funds=N_FUNDS, n_days=N_DAYS, seed=42)
    raw, split = sf.split_into_subclasses(clean.informe, share=SHARE, seed=7)
    path = sf.write_cvm_informe_csv(raw, tmp_path_factory.mktemp("sub") / "inf_diario_fi_202606.csv")
    proc = DataProcessor()
    informe = proc.calculate_net_flow(proc.read_informe_diario(path))
    return clean, SimpleNamespace(informe=informe, cda=clean.cda, raw=raw), split


def test_fixture_really_has_subclass_rows(universes):
    """Guard the guard: the CSV handed to the reader must carry several rows per day for the split funds."""
    clean, split_u, split = universes
    per_day = split_u.raw.groupby(["CNPJ_FUNDO_CLASSE", "DT_COMPTC"]).size()
    assert len(split) >= 10
    assert per_day.max() >= 2 and (split_u.raw["ID_SUBCLASSE"] != "").sum() > len(clean.informe) * SHARE * N_DAYS / N_FUNDS


@pytest.mark.parametrize("signal", SIGNALS)
def test_split_funds_are_not_flagged_more_than_unsplit(signal, universes):
    clean, split_u, split = universes
    flagged = DETECTORS[signal](split_u)
    all_funds = set(clean.informe["CNPJ_FUNDO"].unique())
    rate_split = len(flagged & split) / len(split)
    rate_plain = len(flagged & (all_funds - split)) / len(all_funds - split)
    assert rate_split <= rate_plain + SLACK, (
        f"{signal}: {rate_split:.0%} of clean subclass-reporting funds flagged vs {rate_plain:.0%} of single-class "
        f"funds. Subclass rows are being read as one series."
    )
