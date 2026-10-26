"""Tests for PortfolioReconciliationAnalyzer."""

import pandas as pd
import pytest

from src.analyzers.portfolio_reconciliation import PortfolioReconciliationAnalyzer


@pytest.fixture
def analyzer():
    return PortfolioReconciliationAnalyzer()


class TestPortfolioReconciliation:
    def test_missing_cda_columns_raises(self, analyzer):
        cda = pd.DataFrame({"CNPJ_FUNDO": ["123"]})
        informe = pd.DataFrame({"CNPJ_FUNDO": ["123"], "DT_COMPTC": ["2024-01-01"], "VL_PATRIM_LIQ": [1e6]})
        with pytest.raises(ValueError, match="missing required columns"):
            analyzer.analyze(cda, informe)

    def test_missing_informe_columns_raises(self, analyzer):
        cda = pd.DataFrame({"CNPJ_FUNDO": ["123"], "VL_MERCADO": [1e6]})
        informe = pd.DataFrame({"CNPJ_FUNDO": ["123"]})
        with pytest.raises(ValueError, match="missing required columns"):
            analyzer.analyze(cda, informe)

    def test_empty_dataframes(self, analyzer):
        cda = pd.DataFrame(columns=["CNPJ_FUNDO", "VL_MERCADO"])
        informe = pd.DataFrame(columns=["CNPJ_FUNDO", "DT_COMPTC", "VL_PATRIM_LIQ"])
        result = analyzer.analyze(cda, informe)
        assert result.empty

    def test_static_gap_detected(self, analyzer):
        cda = pd.DataFrame({
            "CNPJ_FUNDO": ["fund1", "fund1"],
            "VL_MERCADO": [300_000, 200_000],  # total = 500k
        })
        informe = pd.DataFrame({
            "CNPJ_FUNDO": ["fund1"],
            "DT_COMPTC": ["2024-01-01"],
            "VL_PATRIM_LIQ": [1_000_000],  # CDA total is 50% of PL
        })
        result = analyzer.analyze(cda, informe)
        gaps = result[result["signal_type"] == "static_reconciliation_gap"]
        assert len(gaps) == 1
        assert gaps.iloc[0]["gap_pct"] == 50.0

    def test_hidden_positions_detected(self, analyzer):
        cda = pd.DataFrame({
            "CNPJ_FUNDO": ["fund1"],
            "VL_MERCADO": [100_000],
        })
        informe = pd.DataFrame({
            "CNPJ_FUNDO": ["fund1", "fund2"],
            "DT_COMPTC": ["2024-01-01", "2024-01-01"],
            "VL_PATRIM_LIQ": [100_000, 10_000_000],  # fund2 not in CDA
        })
        result = analyzer.analyze(cda, informe)
        hidden = result[result["signal_type"] == "hidden_positions"]
        assert len(hidden) == 1
        assert hidden.iloc[0]["CNPJ_FUNDO"] == "fund2"

    def test_no_signals_when_aligned(self, analyzer):
        cda = pd.DataFrame({
            "CNPJ_FUNDO": ["fund1", "fund1"],
            "VL_MERCADO": [500_000, 500_000],  # total = 1M
        })
        informe = pd.DataFrame({
            "CNPJ_FUNDO": ["fund1"],
            "DT_COMPTC": ["2024-01-01"],
            "VL_PATRIM_LIQ": [1_000_000],
        })
        result = analyzer.analyze(cda, informe)
        assert result.empty


class TestLiabilitiesAreNetted:
    """CDA publishes liabilities as positive numbers; the analyzer must subtract them."""

    @staticmethod
    def _informe(pl):
        return pd.DataFrame({
            "CNPJ_FUNDO": ["fund1"], "DT_COMPTC": ["2026-06-30"], "VL_PATRIM_LIQ": [pl],
        })

    def test_liabilities_are_netted(self, analyzer):
        cda = pd.DataFrame({
            "CNPJ_FUNDO": ["fund1"] * 3,
            "TP_APLIC": [
                "Cotas de Fundos",
                "Valores a pagar",
                "Obrigações por ações e outros TVM recebidos em empréstimo",
            ],
            # assets 1.5M, liabilities 0.3M + 0.2M -> net 1.0M == PL
            "VL_MERCADO": [1_500_000, 300_000, 200_000],
        })
        assert analyzer.analyze(cda, self._informe(1_000_000)).empty

    def test_liability_match_ignores_case_and_padding(self, analyzer):
        cda = pd.DataFrame({
            "CNPJ_FUNDO": ["fund1"] * 2,
            "TP_APLIC": ["Cotas de Fundos", "  DIFERENCIAL DE SWAP A PAGAR "],
            "VL_MERCADO": [1_200_000, 200_000],
        })
        assert analyzer.analyze(cda, self._informe(1_000_000)).empty

    def test_without_tp_aplic_the_plain_sum_is_used(self, analyzer):
        # Same amounts with no TP_APLIC column: 2.0M vs 1.0M PL is a 100% gap.
        cda = pd.DataFrame({"CNPJ_FUNDO": ["fund1"] * 3, "VL_MERCADO": [1_500_000, 300_000, 200_000]})
        gaps = analyzer.analyze(cda, self._informe(1_000_000))
        assert gaps.iloc[0]["gap_pct"] == 100.0

    def test_asset_lines_are_not_negated(self, analyzer):
        # "Valores a receber" is an asset; netting it would turn 1.0M into 0.6M.
        cda = pd.DataFrame({
            "CNPJ_FUNDO": ["fund1"] * 2,
            "TP_APLIC": ["Cotas de Fundos", "Valores a receber"],
            "VL_MERCADO": [800_000, 200_000],
        })
        assert analyzer.analyze(cda, self._informe(1_000_000)).empty

    def test_gap_that_survives_netting_is_still_flagged(self, analyzer):
        cda = pd.DataFrame({
            "CNPJ_FUNDO": ["fund1"] * 2,
            "TP_APLIC": ["Cotas de Fundos", "Valores a pagar"],
            "VL_MERCADO": [1_500_000, 100_000],  # net 1.4M vs PL 1.0M
        })
        gaps = analyzer.analyze(cda, self._informe(1_000_000))
        assert gaps.iloc[0]["gap_pct"] == 40.0
