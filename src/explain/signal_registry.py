from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Callable, Mapping, Sequence


Severity = str  # LOW | MEDIUM | HIGH | CRITICAL
Row = Mapping[str, Any]


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _abs_metric(row: Row, column: str) -> float | None:
    value = _to_float(row.get(column))
    return abs(value) if value is not None else None


def _severity_from_named_column(row: Row, column: str = "severity") -> Severity:
    raw = row.get(column)
    if raw is None:
        return "LOW"
    value = str(raw).strip().upper()
    if value in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        return value
    return "LOW"


def _severity_from_thresholds(value: float | None, thresholds: Sequence[tuple[float, Severity]]) -> Severity:
    if value is None:
        return "LOW"
    for minimum, severity in thresholds:
        if value >= minimum:
            return severity
    return "LOW"


def _flow_anomaly_severity(row: Row) -> Severity:
    return _severity_from_thresholds(
        _abs_metric(row, "Z_SCORE_FLOW"),
        thresholds=((5.0, "CRITICAL"), (4.0, "HIGH"), (3.0, "MEDIUM")),
    )


def _pl_drop_severity(row: Row) -> Severity:
    value = _to_float(row.get("PL_VAR_PCT"))
    if value is None:
        return "LOW"
    drop = abs(value) if value < 0 else 0.0
    return _severity_from_thresholds(
        drop,
        thresholds=((40.0, "CRITICAL"), (30.0, "HIGH"), (20.0, "MEDIUM")),
    )


def _run_severity(row: Row) -> Severity:
    value = _to_float(row.get("RUN_LENGTH"))
    return _severity_from_thresholds(
        value,
        thresholds=((20.0, "CRITICAL"), (10.0, "HIGH"), (5.0, "MEDIUM")),
    )


def _shell_network_severity(row: Row) -> Severity:
    count = _to_float(row.get("num_suspicious_issuers"))
    pct = _to_float(row.get("suspicious_pct"))
    # Use whichever is available, biasing toward issuer count.
    if count is not None:
        return _severity_from_thresholds(
            count,
            thresholds=((20.0, "CRITICAL"), (10.0, "HIGH"), (5.0, "MEDIUM")),
        )
    if pct is not None:
        return _severity_from_thresholds(
            pct,
            thresholds=((50.0, "CRITICAL"), (30.0, "HIGH"), (15.0, "MEDIUM")),
        )
    return _severity_from_named_column(row)


def _phantom_asset_severity(row: Row) -> Severity:
    risk = str(row.get("fraud_risk", "")).strip().upper()
    if risk in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}:
        return risk
    return "LOW"


@dataclass(frozen=True)
class SignalDefinition:
    finding_id: str
    title: str
    plain_language_explanation: str
    evidence_fields: tuple[str, ...]
    primary_entities: tuple[str, ...]
    next_steps: tuple[str, ...]
    caveats: tuple[str, ...]
    severity_rule: Callable[[Row], Severity]


SIGNAL_REGISTRY: dict[str, SignalDefinition] = {
    "circular_flow": SignalDefinition(
        finding_id="circular_flow",
        title="Circularity / round-tripping pattern",
        plain_language_explanation=(
            "We detected fund-to-fund links and circular exposures within the same administrator group. "
            "This can be consistent with round-tripping, where money effectively returns to the originating "
            "entity through intermediaries."
        ),
        evidence_fields=(
            "admin_cnpj",
            "CNPJ_FUNDO",
            "counterparty_cnpj",
            "num_circular_connections",
            "cycle_length",
            "total_value",
            "fraud_pattern",
            "severity",
            "confidence",
        ),
        primary_entities=("ADMINISTRATOR", "FUND"),
        next_steps=(
            "List the administrator’s funds involved in the loop and request subscription/redemption ledgers for the window.",
            "Trace cash movements around the dates of the circular exposures (bank transfers, distributor orders).",
            "Check related-party links between administrators, issuers, and any involved counterparties.",
        ),
        caveats=(
            "Fund-to-fund holdings can be legitimate in some structures; corroborate with fund documentation and permitted investments.",
            "Public datasets may omit investor identity and transaction routing; internal records are needed to confirm round-tripping.",
        ),
        severity_rule=lambda row: _severity_from_named_column(row),
    ),
    "layered_funds": SignalDefinition(
        finding_id="layered_funds",
        title="Layered funds (fund-of-funds cascade)",
        plain_language_explanation=(
            "We detected funds investing in other funds within the same administrator group, alongside "
            "unusually high recent returns. Layered structures can amplify valuation or reporting problems."
        ),
        evidence_fields=(
            "admin_cnpj",
            "holder_fund",
            "held_fund",
            "investment_value",
            "holder_avg_daily_return",
            "held_avg_daily_return",
            "severity",
        ),
        primary_entities=("ADMINISTRATOR", "FUND"),
        next_steps=(
            "Identify whether inter-fund investments are permitted and disclosed; review prospectuses/regulations.",
            "Check whether returns are consistent with asset mix and whether pricing sources are independent.",
            "Look for rapid growth in AUM/PL with limited external flows (potential valuation inflation).",
        ),
        caveats=(
            "Fund-of-funds structures can be legitimate; focus on unexplained return patterns and related-party exposure.",
        ),
        severity_rule=lambda row: _severity_from_named_column(row),
    ),
    "flow_anomaly": SignalDefinition(
        finding_id="flow_anomaly",
        title="Flow anomaly (z-score outlier)",
        plain_language_explanation=(
            "We detected days where a fund’s net flow was statistically extreme compared to its own history. "
            "This can indicate coordinated subscriptions/redemptions, liquidity stress, or unusual investor behavior."
        ),
        evidence_fields=("CNPJ_FUNDO", "DT_COMPTC", "FLUXO_LIQ_DIA", "Z_SCORE_FLOW"),
        primary_entities=("FUND",),
        next_steps=(
            "Identify the largest subscribers/redeemers for the flagged dates (distributor/investor level).",
            "Confirm whether flows match public disclosures and internal transaction logs (avoid reporting mismatches).",
            "Check if flows coincide with portfolio changes, pricing events, or corporate actions.",
        ),
        caveats=(
            "Large flows can be legitimate (e.g., institutional reallocations); treat as a lead requiring corroboration.",
        ),
        severity_rule=_flow_anomaly_severity,
    ),
    "pl_drop": SignalDefinition(
        finding_id="pl_drop",
        title="Abrupt AUM/PL drop",
        plain_language_explanation=(
            "We detected a sharp drop in a fund’s reported assets under management (AUM/PL) over a short period. "
            "This can indicate a redemption wave, revaluation event, or reporting correction."
        ),
        evidence_fields=("CNPJ_FUNDO", "DT_COMPTC", "VL_PATRIM_LIQ", "PL_VAR_PCT"),
        primary_entities=("FUND",),
        next_steps=(
            "Validate whether the drop is driven by redemptions, valuation changes, or accounting corrections.",
            "Cross-check pricing sources for key holdings during the window.",
            "Review communications to investors and any exceptional events disclosures.",
        ),
        caveats=(
            "PL can change for non-fraud reasons (market moves, large redemptions); confirm with supporting records.",
        ),
        severity_rule=_pl_drop_severity,
    ),
    "redemption_run": SignalDefinition(
        finding_id="redemption_run",
        title="Redemption run (consecutive negative flows)",
        plain_language_explanation=(
            "We detected consecutive days of net negative flows. Sustained runs can signal investor panic, "
            "liquidity mismatch, gating events, or coordinated activity."
        ),
        evidence_fields=("CNPJ_FUNDO", "DT_COMPTC", "FLUXO_LIQ_DIA", "RUN_LENGTH"),
        primary_entities=("FUND",),
        next_steps=(
            "Identify whether the fund used liquidity management tools (gates, side pockets) during the run.",
            "Check if a small number of investors/distributors account for most redemptions.",
            "Compare portfolio liquidity to redemption pressure to assess forced selling risk.",
        ),
        caveats=(
            "Runs can be market-driven; look for clustering across related funds or administrators for stronger signals.",
        ),
        severity_rule=_run_severity,
    ),
    "shell_network": SignalDefinition(
        finding_id="shell_network",
        title="Shell-like issuer network concentration",
        plain_language_explanation=(
            "We detected multiple issuer names with patterns commonly associated with shell or micro-companies. "
            "A dense network of such issuers concentrated under one administrator can be a systemic red flag."
        ),
        evidence_fields=(
            "admin_cnpj",
            "num_suspicious_issuers",
            "suspicious_value",
            "suspicious_pct",
            "num_funds_affected",
            "severity",
        ),
        primary_entities=("ADMINISTRATOR",),
        next_steps=(
            "Collect issuer registries and verify each issuer’s existence, governance, and economic activity.",
            "Map related-party links between issuers, administrators, and any bank/distributor entities.",
            "Review the due diligence process and valuation methodology for private/illiquid instruments.",
        ),
        caveats=(
            "Name-based heuristics can over-flag; confirm with corporate registry and issuance documentation.",
        ),
        severity_rule=_shell_network_severity,
    ),
    "phantom_asset_exposure": SignalDefinition(
        finding_id="phantom_asset_exposure",
        title="Suspicious/phantom asset exposure",
        plain_language_explanation=(
            "We detected assets that appear inconsistent with public registries or have private-asset red flags. "
            "Public assets missing from registries are especially concerning; private assets require manual verification."
        ),
        evidence_fields=("CNPJ_FUNDO", "asset_code", "asset_type", "status", "fraud_risk", "total_value_fund"),
        primary_entities=("FUND",),
        next_steps=(
            "Validate each flagged position with independent documentation (issuance docs, custodian statements).",
            "Confirm pricing sources and whether the asset should be publicly verifiable (stocks/ETFs/BDRs).",
            "Prioritize positions with high value and repeated red flags across multiple funds.",
        ),
        caveats=(
            "Some private assets are not publicly searchable; treat these as ‘needs verification’ rather than definitive fraud.",
        ),
        severity_rule=_phantom_asset_severity,
    ),
    "quotaholder_anomaly": SignalDefinition(
        finding_id="quotaholder_anomaly",
        title="Quotaholder count anomaly",
        plain_language_explanation=(
            "We detected abnormal changes in the number of quota holders (NR_COTST). "
            "Sudden drops can indicate insider knowledge of problems, phantom participation "
            "suggests wash trading, and per-capita AUM outliers suggest concentrated ownership."
        ),
        evidence_fields=("CNPJ_FUNDO", "DT_COMPTC", "signal_type", "NR_COTST", "pct_change", "per_capita_aum", "severity"),
        primary_entities=("FUND",),
        next_steps=(
            "Identify the investors who redeemed during the mass exodus window.",
            "Cross-reference quotaholder changes with flow data to confirm whether drops match redemption volumes.",
            "Check if coordinated exodus across multiple funds under the same admin suggests insider information.",
        ),
        caveats=(
            "Quotaholder count changes can be legitimate (e.g., fund mergers, institutional reallocations).",
            "NR_COTST is a point-in-time snapshot and may not capture intra-day movements.",
        ),
        severity_rule=lambda row: _severity_from_named_column(row),
    ),
    "cost_basis_anomaly": SignalDefinition(
        finding_id="cost_basis_anomaly",
        title="Cost basis vs market value anomaly",
        plain_language_explanation=(
            "We detected positions where the reported market value diverges significantly from cost basis. "
            "Zero-cost or negative-cost positions, or extreme unrealized gains on illiquid assets, "
            "can indicate phantom assets, data fabrication, or valuation manipulation."
        ),
        evidence_fields=("CNPJ_FUNDO", "CD_ATIVO", "signal_type", "VL_MERCADO", "VL_CUSTO_POS_FINAL", "gain_ratio", "severity"),
        primary_entities=("FUND",),
        next_steps=(
            "Validate the pricing methodology for positions with extreme gain ratios.",
            "Check if zero-cost or negative-cost entries reflect data errors or actual phantom positions.",
            "Compare gain ratios across similar funds to identify systematic overvaluation by a single administrator.",
        ),
        caveats=(
            "Some legitimate positions (e.g., received via corporate actions) may have zero cost basis.",
            "Gain ratios can be extreme for long-held positions in volatile markets.",
        ),
        severity_rule=lambda row: _severity_from_named_column(row),
    ),
    "portfolio_reconciliation_gap": SignalDefinition(
        finding_id="portfolio_reconciliation_gap",
        title="Portfolio vs PL reconciliation gap",
        plain_language_explanation=(
            "We detected a significant discrepancy between the net portfolio positions (CDA assets minus "
            "liabilities) and the reported net asset value (Informe Diario PL). Gaps can indicate hidden positions, "
            "off-book assets, or reporting inconsistencies."
        ),
        evidence_fields=("CNPJ_FUNDO", "DT_COMPTC", "cda_total", "informe_pl", "gap_pct", "signal_type", "severity"),
        primary_entities=("FUND",),
        next_steps=(
            "Request the fund's full position report and reconcile against CDA filings.",
            "Check if the gap is explained by derivatives, receivables, or other off-balance items.",
            "Verify whether the administrator filed CDA data on time and completely.",
        ),
        caveats=(
            "CDA and Informe may have different reporting dates; minor gaps can be timing-related.",
            "Derivative positions and cash balances may not be fully reflected in CDA.",
        ),
        severity_rule=lambda row: _severity_from_named_column(row),
    ),
    "fund_lifecycle_anomaly": SignalDefinition(
        finding_id="fund_lifecycle_anomaly",
        title="Fund lifecycle anomaly",
        plain_language_explanation=(
            "We detected unusual patterns in fund creation or cancellation. Hot starts (rapid AUM growth), "
            "short-lived funds, coordinated creation, or suspicious inflow timing can indicate "
            "pump-and-dump schemes or pre-arranged investment structures."
        ),
        evidence_fields=("CNPJ_FUNDO", "signal_type", "DT_CONST", "DT_CANCEL", "days_active", "max_pl", "severity"),
        primary_entities=("FUND",),
        next_steps=(
            "Check the fund prospectus for pre-arranged subscription commitments.",
            "Identify the initial investors and verify their relationship to the administrator or manager.",
            "For short-lived funds, trace the final disposition of assets and redemption destinations.",
        ),
        caveats=(
            "Some funds are legitimately created for specific short-term purposes (e.g., structured products).",
            "Hot starts can be normal for funds seeded by institutional investors.",
        ),
        severity_rule=lambda row: _severity_from_named_column(row),
    ),
    "manager_network_anomaly": SignalDefinition(
        finding_id="manager_network_anomaly",
        title="Manager network anomaly",
        plain_language_explanation=(
            "We detected unusual patterns in fund manager relationships. A single manager operating "
            "across multiple administrators, holding overlapping illiquid assets, or appearing as an "
            "issuer in their own funds can indicate coordinated manipulation."
        ),
        evidence_fields=("CNPJ_GESTOR", "signal_type", "num_admins", "num_funds", "total_aum", "overlap_assets", "severity"),
        primary_entities=("MANAGER",),
        next_steps=(
            "Map the full network of funds, administrators, and issuers connected to the flagged manager.",
            "Check if overlapping illiquid assets are priced consistently across different funds.",
            "Verify whether the manager's registered activities match the scale of assets under management.",
        ),
        caveats=(
            "Large asset managers legitimately operate across multiple administrators.",
            "Asset overlap can occur in index-tracking or sector-focused strategies.",
        ),
        severity_rule=lambda row: _severity_from_named_column(row),
    ),
    "window_dressing": SignalDefinition(
        finding_id="window_dressing",
        title="Window dressing pattern",
        plain_language_explanation=(
            "We detected temporal manipulation patterns around reporting dates. Month-end return spikes, "
            "quarter-end PL inflation, or flow reversal patterns (large inflows before month-end "
            "followed by outflows after) suggest deliberate performance or AUM window dressing."
        ),
        evidence_fields=("CNPJ_FUNDO", "signal_type", "period", "metric_value", "baseline_value", "deviation_pct", "severity"),
        primary_entities=("FUND",),
        next_steps=(
            "Compare month-end vs mid-month pricing sources for the fund's key holdings.",
            "Check if flow reversals correspond to specific distributors or investor categories.",
            "Verify whether quarter-end PL spikes are explained by legitimate corporate actions or dividends.",
        ),
        caveats=(
            "Some month-end effects are market-wide (e.g., rebalancing flows) and not fund-specific.",
            "Quarter-end reporting deadlines can cause legitimate timing differences.",
        ),
        severity_rule=lambda row: _severity_from_named_column(row),
    ),
    "valuation_smoothing": SignalDefinition(
        finding_id="valuation_smoothing",
        title="Valuation smoothing detected",
        plain_language_explanation=(
            "We detected statistical patterns consistent with return smoothing or stale pricing. "
            "High return autocorrelation, extended periods of zero price change, or suspiciously "
            "low volatility can indicate that a fund's NAV is not reflecting true market values."
        ),
        evidence_fields=("CNPJ_FUNDO", "signal_type", "autocorrelation", "stale_days", "vol_ratio", "severity"),
        primary_entities=("FUND",),
        next_steps=(
            "Request the fund's valuation methodology and pricing sources for illiquid positions.",
            "Check if the fund's returns are consistent with its stated asset class and strategy.",
            "Compare the fund's volatility profile with peer funds holding similar assets.",
        ),
        caveats=(
            "Some asset classes (e.g., real estate, private credit) legitimately have low daily volatility.",
            "Autocorrelation can be caused by illiquid markets rather than deliberate smoothing.",
        ),
        severity_rule=lambda row: _severity_from_named_column(row),
    ),
    "cross_fund_issuer": SignalDefinition(
        finding_id="cross_fund_issuer",
        title="Cross-fund issuer concentration",
        plain_language_explanation=(
            "We detected issuer-level patterns that suggest related-party exposure or captive issuers. "
            "Issuers appearing exclusively in one administrator's funds, near-identical issuer names, "
            "or heavy concentration in a single issuer across multiple funds can indicate manufactured assets."
        ),
        evidence_fields=("EMISSOR", "signal_type", "num_funds", "num_admins", "total_exposure", "captive_admin", "similarity", "severity"),
        primary_entities=("ADMINISTRATOR",),
        next_steps=(
            "Verify each captive issuer's corporate registry, governance structure, and economic activity.",
            "Check if near-identical issuer names share beneficial ownership or directors.",
            "Request independent valuation reports for concentrated issuer exposures.",
        ),
        caveats=(
            "Some issuers legitimately serve a single administrator (e.g., proprietary structured products).",
            "Name similarity may be coincidental for common corporate naming conventions.",
        ),
        severity_rule=lambda row: _severity_from_named_column(row),
    ),
    "cross_fund_price_divergence": SignalDefinition(
        finding_id="cross_fund_price_divergence",
        title="Declared price diverges from peer funds",
        plain_language_explanation=(
            "Several funds held this same asset on this same date, and this fund declared a unit price "
            "materially different from what its peers declared. Funds marking the same instrument on the "
            "same day should agree; a fund that does not is either applying a different valuation "
            "methodology or mismarking the position -- upward to support its NAV, or downward to defer "
            "a loss. This comparison uses only the funds' own CVM filings, with no external price source."
        ),
        evidence_fields=(
            "CD_ATIVO",
            "declared_unit_price",
            "consensus_unit_price",
            "divergence_pct",
            "robust_z",
            "peer_fund_count",
            "position_value",
            "direction",
            "severity",
        ),
        primary_entities=("FUND", "ASSET"),
        next_steps=(
            "Request the fund's valuation methodology and pricing source for this asset on this date.",
            "Compare against the peer funds' own methodologies: a defensible difference should be documented.",
            "For unlisted credit, request the independent valuation report and check who prepared it.",
            "Check whether the divergence is systematic across the fund's portfolio or isolated to this asset.",
        ),
        caveats=(
            "Different valuation methodologies can legitimately produce different marks for illiquid "
            "assets, particularly around credit events or for instruments with different maturities "
            "sharing one asset code.",
            "The consensus is the median of the funds holding the asset; if most holders are related "
            "parties, the consensus itself may be mismarked and an honest fund would appear divergent.",
            "Positions acquired at different times may carry different amortized costs under some "
            "accounting treatments.",
        ),
        severity_rule=lambda row: _severity_from_named_column(row),
    ),
    "benford_violation": SignalDefinition(
        finding_id="benford_violation",
        title="Reported figures deviate from Benford's Law",
        plain_language_explanation=(
            "Leading digits in this fund's reported figures do not follow the logarithmic distribution that "
            "naturally-occurring financial data obeys. People inventing numbers tend to spread first digits "
            "evenly, while genuine values start with 1 about 30% of the time and with 9 about 5%. A deviation "
            "is a hint that some series may be constructed rather than measured."
        ),
        evidence_fields=(
            "fund_cnpj", "pl_mad", "pl_p_value", "pl_fraud_risk", "pl_sample_size",
            "quota_mad", "quota_fraud_risk", "captc_mad", "captc_fraud_risk",
            "resg_mad", "resg_fraud_risk", "overall_fraud_risk",
        ),
        primary_entities=("FUND",),
        next_steps=(
            "Identify which series deviates and reconcile it against the fund's own accounting records.",
            "Check whether the deviation coincides with a change of administrator, auditor or valuation agent.",
            "Re-test over a longer window: short samples deviate by chance.",
        ),
        caveats=(
            "Benford applies to values spanning several orders of magnitude. Series that are bounded, "
            "capped, or heavily rounded deviate for entirely innocent reasons -- quota values clustered "
            "near 1.0 are a common false positive.",
            "A small sample deviates by chance; treat a low sample_size result as inconclusive.",
            "Deviation indicates the data is unusual, not that it was fabricated.",
        ),
        severity_rule=lambda row: _severity_from_named_column(row, "overall_fraud_risk"),
    ),
    "concentration_violation": SignalDefinition(
        finding_id="concentration_violation",
        title="Portfolio concentration beyond regulatory limits",
        plain_language_explanation=(
            "A small number of positions account for an outsized share of this fund's portfolio, exceeding "
            "the diversification limits its category is subject to. Concentration is not fraud in itself, "
            "but it is how a fund becomes exposed to a single issuer's failure, and it is a precondition "
            "for a mismarked position to move the fund's whole NAV."
        ),
        evidence_fields=(
            "CNPJ_FUNDO", "category", "num_positions", "hhi", "top1_pct", "top5_pct",
            "top10_pct", "regulatory_limit_pct", "violates_limit", "largest_position",
            "largest_position_value", "severity", "fraud_flags",
        ),
        primary_entities=("FUND", "ASSET"),
        next_steps=(
            "Check the fund's regulation for the diversification limits it committed to.",
            "Establish who issued the dominant position and whether they are related to the administrator or manager.",
            "Request an independent valuation of the largest position.",
        ),
        caveats=(
            "Some categories are permitted or designed to concentrate -- FIPs, single-asset and exclusive "
            "funds among them -- so exceeding a generic limit may be entirely consistent with the mandate.",
            "Month-end snapshots can catch a portfolio mid-rebalance.",
        ),
        severity_rule=lambda row: _severity_from_named_column(row),
    ),
    "peer_outlier": SignalDefinition(
        finding_id="peer_outlier",
        title="Risk/return profile far from category peers",
        plain_language_explanation=(
            "This fund's returns, volatility or risk-adjusted performance sit far from other funds in the "
            "same category. Returns that are too good, or too smooth, are the signature of a portfolio "
            "marked to something other than the market -- the pattern that makes a Ponzi structure look "
            "attractive right up until redemptions arrive."
        ),
        evidence_fields=(
            "CNPJ_FUNDO", "category", "num_peers", "fund_return", "fund_volatility",
            "fund_sharpe", "peer_avg_return", "peer_avg_volatility", "peer_avg_sharpe",
            "return_zscore", "volatility_zscore", "sharpe_zscore", "fraud_flag",
        ),
        primary_entities=("FUND",),
        next_steps=(
            "Compare the holdings against the peers used for comparison: an unusual profile should have an unusual portfolio behind it.",
            "For suppressed volatility, check whether illiquid positions are being marked at model rather than market.",
            "Ask how the fund achieved its risk-adjusted return and verify the explanation against its holdings.",
        ),
        caveats=(
            "Peer groups come from the CVM category label, which is coarse -- a fund may differ legitimately "
            "from its nominal peers because it runs a different strategy under the same label.",
            "A genuinely skilled manager is also an outlier; this signal cannot separate skill from mismarking.",
            "Short measurement windows make ordinary funds look extreme.",
        ),
        severity_rule=lambda row: _severity_from_thresholds(
            _abs_metric(row, "sharpe_zscore"),
            thresholds=((5.0, "CRITICAL"), (4.0, "HIGH"), (3.0, "MEDIUM")),
        ),
    ),
}


def validate_registry() -> None:
    """Validate registry is internally consistent (unique IDs, required fields)."""
    seen: set[str] = set()
    for finding_id, definition in SIGNAL_REGISTRY.items():
        if finding_id != definition.finding_id:
            raise ValueError(f"Registry key mismatch for {finding_id}")
        if finding_id in seen:
            raise ValueError(f"Duplicate finding_id: {finding_id}")
        seen.add(finding_id)
        if not definition.title:
            raise ValueError(f"Missing title for {finding_id}")
        if not definition.plain_language_explanation:
            raise ValueError(f"Missing explanation for {finding_id}")
        if not definition.primary_entities:
            raise ValueError(f"Missing primary_entities for {finding_id}")
        if not definition.next_steps:
            raise ValueError(f"Missing next_steps for {finding_id}")

