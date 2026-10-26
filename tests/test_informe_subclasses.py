"""A class with subclasses must read as ONE series per day.

Since RCVM 175 the CVM informe diario carries a row per subclass per day under
the class CNPJ, each with its own quota, PL and holder count. The pipeline groups
by CNPJ_FUNDO, so left alone it read those rows as one interleaved series: quota
"returns" were the gaps between subclasses, PL saw-toothed, and Benford counted
every figure once per subclass. On the June 2026 informe that was 221 classes
(0.9%), 30% of the published Priority-review queue.

The fixtures keep the real CVM header and invented numbers.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from scripts.run_investigation import _read_informe_any
from src.processors.data_processor import DataProcessor

HEADER = "TP_FUNDO_CLASSE;CNPJ_FUNDO_CLASSE;ID_SUBCLASSE;DT_COMPTC;VL_TOTAL;VL_QUOTA;VL_PATRIM_LIQ;CAPTC_DIA;RESG_DIA;NR_COTST"
TWO_SUB = "11.111.111/0001-11"   # two subclasses, quotas 2.0 and 1.0
CLASS_ONLY = "22.222.222/0001-22"  # reports at class level, no subclass
ONE_SUB = "33.333.333/0001-33"   # a single subclass


def _write(path, lines):
    path.write_text("\n".join([HEADER, *lines]) + "\n", encoding="latin1")
    return path


@pytest.fixture
def informe(tmp_path):
    rows = [
        # TWO_SUB: big subclass BIG (quota ~2.0), small subclass SMALL (quota ~1.0).
        f"CLASSES - FIF;{TWO_SUB};BIG;2026-06-01;1000.5;2.0000;1000.0;10.0;4.0;100",
        f"CLASSES - FIF;{TWO_SUB};SMALL;2026-06-01;100.5;1.0000;100.0;1.0;0.5;7",
        f"CLASSES - FIF;{TWO_SUB};BIG;2026-06-02;1010.5;2.0200;1010.0;20.0;5.0;101",
        f"CLASSES - FIF;{TWO_SUB};SMALL;2026-06-02;101.5;1.0100;101.0;2.0;0.5;8",
        # day 3: BIG did not report, SMALL did. Its 1.02 must not stand in for BIG.
        f"CLASSES - FIF;{TWO_SUB};SMALL;2026-06-03;102.5;1.0200;102.0;3.0;1.0;9",
        # CLASS_ONLY: no subclass id.
        f"CLASSES - FIF;{CLASS_ONLY};;2026-06-01;500.0;1.5000;500.0;5.0;1.0;40",
        f"CLASSES - FIF;{CLASS_ONLY};;2026-06-02;510.0;1.5100;510.0;6.0;2.0;41",
        # ONE_SUB: a single subclass, one row a day.
        f"CLASSES - FIF;{ONE_SUB};ONLY;2026-06-01;300.0;3.0000;300.0;0.0;0.0;5",
        f"CLASSES - FIF;{ONE_SUB};ONLY;2026-06-02;301.0;3.0100;301.0;0.0;0.0;5",
    ]
    return _write(tmp_path / "inf_diario_fi_202606.csv", rows)


def _fund(df, cnpj_digits):
    return df[df["CNPJ_FUNDO"] == cnpj_digits].reset_index(drop=True)


TWO = "11111111000111"
CLS = "22222222000122"
ONE = "33333333000133"


def test_one_row_per_class_per_day(informe):
    df = DataProcessor().read_informe_diario(informe)
    assert not df.duplicated(["CNPJ_FUNDO", "DT_COMPTC"]).any()
    assert len(_fund(df, TWO)) == 3
    assert len(df) == 3 + 2 + 2


def test_amounts_are_summed_across_subclasses(informe):
    two = _fund(DataProcessor().read_informe_diario(informe), TWO)
    day1 = two.iloc[0]
    assert day1["VL_PATRIM_LIQ"] == pytest.approx(1100.0)
    assert day1["VL_TOTAL"] == pytest.approx(1101.0)
    assert day1["CAPTC_DIA"] == pytest.approx(11.0)
    assert day1["RESG_DIA"] == pytest.approx(4.5)
    assert day1["NR_COTST"] == 107  # upper bound on distinct holders


def test_nothing_is_lost_or_invented_in_the_totals(informe):
    processor = DataProcessor()
    raw = processor.read_informe_diario(informe, consolidate_subclasses=False)
    folded = processor.read_informe_diario(informe)
    for col in ("VL_TOTAL", "VL_PATRIM_LIQ", "CAPTC_DIA", "RESG_DIA", "NR_COTST"):
        assert folded[col].sum() == pytest.approx(raw[col].sum())


def test_quota_follows_one_reported_subclass_never_a_blend(informe):
    two = _fund(DataProcessor().read_informe_diario(informe), TWO)
    assert two["VL_QUOTA"].iloc[0] == pytest.approx(2.0)
    assert two["VL_QUOTA"].iloc[1] == pytest.approx(2.02)
    # BIG is the dominant subclass (most PL). On day 3 only SMALL reported: the
    # class has no quota that day rather than a 1.02 that would read as a -49% day.
    assert np.isnan(two["VL_QUOTA"].iloc[2])
    assert set(two["ID_SUBCLASSE_QUOTA"].dropna()) == {"BIG"}
    returns = two["VL_QUOTA"].pct_change().dropna()
    assert (returns.abs() < 0.05).all()


def test_the_raw_rows_really_do_produce_the_phantom_returns(informe):
    """Guards the premise: without folding, the interleaved quotas are +-50% 'returns'."""
    raw = DataProcessor().read_informe_diario(informe, consolidate_subclasses=False)
    returns = _fund(raw, TWO)["VL_QUOTA"].pct_change().dropna()
    assert (returns.abs() > 0.4).any()


def test_provenance_columns_say_what_was_folded(informe):
    df = DataProcessor().read_informe_diario(informe)
    two = _fund(df, TWO)
    assert list(two["N_SUBCLASS_ROWS"]) == [2, 2, 1]
    assert two["ID_SUBCLASSE"].isna().all()
    assert (_fund(df, CLS)["N_SUBCLASS_ROWS"] == 1).all()


def test_other_classes_pass_through_untouched_and_in_file_order(informe):
    processor = DataProcessor()
    raw = processor.read_informe_diario(informe, consolidate_subclasses=False)
    folded = processor.read_informe_diario(informe)
    for cnpj in (CLS, ONE):
        before = _fund(raw, cnpj)
        after = _fund(folded, cnpj)
        for col in ("DT_COMPTC", "VL_QUOTA", "VL_PATRIM_LIQ", "NR_COTST"):
            assert list(after[col]) == list(before[col])
    # the folded class keeps the place of its first row: it still comes first
    assert folded["CNPJ_FUNDO"].iloc[0] == TWO


def test_a_single_subclass_keeps_its_own_quota(informe):
    one = _fund(DataProcessor().read_informe_diario(informe), ONE)
    assert list(one["VL_QUOTA"]) == [3.0, 3.01]
    assert list(one["ID_SUBCLASSE"]) == ["ONLY", "ONLY"]


def test_frames_without_subclass_column_are_returned_as_they_are():
    df = pd.DataFrame({
        "CNPJ_FUNDO": ["F1", "F1"],
        "DT_COMPTC": pd.to_datetime(["2026-06-01", "2026-06-01"]),
        "VL_QUOTA": [1.0, 2.0],
        "VL_PATRIM_LIQ": [10.0, 20.0],
    })
    assert DataProcessor().consolidate_subclasses(df) is df


def test_empty_and_blank_rows_do_not_crash(tmp_path):
    processor = DataProcessor()
    assert processor.consolidate_subclasses(pd.DataFrame()).empty
    blank = _write(tmp_path / "inf_diario_fi_blank.csv", [";" * (len(HEADER.split(";")) - 1)])
    assert len(processor.read_informe_diario(blank)) == 1


def test_class_level_row_beside_subclass_rows_is_not_counted_twice(tmp_path, caplog):
    rows = [
        f"CLASSES - FIF;{TWO_SUB};;2026-06-01;1100.0;2.0;1100.0;10.0;4.0;107",
        f"CLASSES - FIF;{TWO_SUB};BIG;2026-06-01;1000.0;2.0;1000.0;9.0;4.0;100",
        f"CLASSES - FIF;{TWO_SUB};SMALL;2026-06-01;100.0;1.0;100.0;1.0;0.0;7",
    ]
    path = _write(tmp_path / "inf_diario_fi_202606.csv", rows)
    with caplog.at_level(logging.WARNING):
        df = DataProcessor().read_informe_diario(path)
    assert len(df) == 1
    assert df["VL_PATRIM_LIQ"].iloc[0] == pytest.approx(1100.0)
    assert "dropped 2 subclass row(s)" in caplog.text


def test_months_are_folded_together_so_the_quota_series_does_not_step(tmp_path):
    """The dominant subclass is chosen over the whole window, not month by month.

    In month 1 SMALL holds more PL than BIG; over both months BIG does. Folding
    each month on its own would follow SMALL in month 1 and BIG in month 2, and
    the quota would step from ~1.0 to ~2.0 at the boundary.
    """
    month1 = [
        f"CLASSES - FIF;{TWO_SUB};BIG;2026-05-29;100.0;2.0;100.0;0;0;1",
        f"CLASSES - FIF;{TWO_SUB};SMALL;2026-05-29;200.0;1.0;200.0;0;0;1",
    ]
    month2 = [
        f"CLASSES - FIF;{TWO_SUB};BIG;2026-06-01;900.0;2.02;900.0;0;0;1",
        f"CLASSES - FIF;{TWO_SUB};SMALL;2026-06-01;200.0;1.01;200.0;0;0;1",
    ]
    _write(tmp_path / "inf_diario_fi_202605.csv", month1)
    _write(tmp_path / "inf_diario_fi_202606.csv", month2)
    df = _read_informe_any(tmp_path, processor=DataProcessor())
    assert list(df["VL_QUOTA"]) == [2.0, 2.02]
    assert list(df["VL_PATRIM_LIQ"]) == [300.0, 1100.0]


def test_a_repeated_class_level_row_is_not_a_subclass_and_is_left_alone(tmp_path):
    row = f"CLASSES - FIF;{CLASS_ONLY};;2026-06-01;500.0;1.5;500.0;5.0;1.0;40"
    path = _write(tmp_path / "inf_diario_fi_202606.csv", [row, row])
    assert len(DataProcessor().read_informe_diario(path)) == 2
