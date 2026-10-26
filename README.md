# funds_screens

Statistical fraud detection for Brazilian investment funds, built on public CVM
filings.

The toolkit ingests the CVM's open datasets — informe diário (daily NAV, flows,
quotaholder counts), CDA (portfolio composition) and cadastro (fund registry) —
and runs a battery of detectors over them, producing per-fund findings with
plain-language briefs explaining what was flagged and what would corroborate it.

Detection patterns are drawn from documented Brazilian fund-industry cases:
circular fund-to-fund flows, phantom assets, valuation smoothing, window
dressing, and cross-fund issuer concentration.

> **Findings are leads, not proof.** Every detector produces statistical red
> flags that require corroboration against records this toolkit cannot see —
> subscription ledgers, bank transfers, counterparty documentation. Nothing here
> establishes wrongdoing on its own.

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Requirements are split so you install only what you need:

| File | Contents |
|---|---|
| `requirements.txt` | Core runtime: pandas, numpy, requests, matplotlib, tqdm |
| `requirements-dev.txt` | pytest, hypothesis, ruff |
| `requirements-optional.txt` | `yfinance`, for the optional market-price validator |
| `requirements-notebooks.txt` | JupyterLab, seaborn, for `notebooks/` |

## Collect data

CVM publishes monthly ZIPs at <https://dados.cvm.gov.br/dados>. `CVMCollector`
downloads and extracts them into `data/raw/`:

```python
from src.collectors.cvm_collector import CVMCollector

c = CVMCollector()
c.download_registro_fundo_classe()   # fund/class registry
c.download_informe_diario(2026, 6)   # daily NAV and flows
c.download_cda(2026, 6)              # portfolio composition
```

Use `registro_fundo_classe`, not the legacy `cad_fi.csv`. Since RCVM 175 the
informe reports by *class*, and almost every record in `cad_fi.csv` has been
cancelled: it joins to 6.6% of the funds that actually report, against 88.9%
for the class registry, which also carries administrator and manager CNPJs.

The monthly CDA arrives as one ZIP of ten to twelve CSVs — eight position
blocks split by asset class, plus summaries. All position blocks are read and
unioned; pass the directory rather than a single file.

**Match the registry to the period.** The registry is a current snapshot while
the informe is historical, so pairing today's registry with an old month drops
funds that had not yet registered. A fund filter that matches nothing is
reported as an error rather than producing an empty report.

**Subclasses are folded.** A class with subclasses reports one informe row per
subclass per day, each with its own quota, PL and holder count. The reader folds
them into one row per class per day: PL, flows and holder counts are summed
(derived), and `VL_QUOTA` is the quota the dominant subclass reported. Read as
separate rows they interleave into one series, and every detector that groups by
CNPJ sees jumps that never happened. `read_informe_diario(consolidate_subclasses=False)`
returns the raw rows.

**The CDA is a balance sheet.** CVM publishes liabilities ("Valores a pagar",
swap payables, ...) as positive numbers in the same blocks as assets. The
reconciliation detector subtracts them before comparing with PL.

## Run an investigation

```bash
python scripts/run_investigation.py \
    --informe  data/raw/inf_diario_fi_202606.csv \
    --cda      data/raw \
    --cadastro data/raw/registro_classe.csv \
    --strict
```

Outputs land in `reports/investigation/<run_id>/`:

- `findings/*.csv` — one file per detector
- `briefs/` — per-entity HTML and Markdown explanations
- `summary.json` — run metadata, finding counts, and which analyzers ran

`--analysis` selects individual detectors; `--help` lists them. There is also a
guided terminal flow at `scripts/investigation_tui.py`.

### Scoping to a subset of funds

By default every fund in the loaded data is investigated. To narrow it:

```bash
# Every fund under one administrator (partial, case-insensitive name match)
--fund-mode administrator --fund-identifier "ACME DTVM" --active-funds-only

# Every fund under one manager
--fund-mode manager --fund-identifier "ACME GESTORA"

# An explicit list
--fund-mode cnpj_list --fund-identifier "12.345.678/0001-90,11222333000181"
```

Names are resolved against the registry, so `--fund-mode administrator` and
`--fund-mode manager` need one loaded. A full month covers every fund in Brazil,
so scoping the first run to one administrator is usually the right move.

**Use `--strict` in automation.** It exits non-zero if any analyzer crashed. A
report with no findings because a detector died looks exactly like a clean
result, and `summary.json`'s `execution` block records which analyzers failed or
were skipped for lack of inputs.

## Detectors

| Area | Modules |
|---|---|
| Flow and NAV | `anomaly_detector` (flow z-scores, PL drops, redemption runs, flow/performance divergence) |
| Fraud schemes | `fraud_schemes` (circular flow, layered funds, asset inflation, shell networks) |
| Portfolio | `enhanced_phantom_assets`, `portfolio_reconciliation`, `concentration`, `cost_basis_analyzer` |
| Valuation | `valuation_smoothing`, `window_dressing`, `price_divergence`, `benford_law` |
| Structure | `cross_fund_issuer`, `manager_network`, `fund_lifecycle`, `quotaholder_analyzer`, `peer_comparison` |
| Market data | `market_data` — optional, requires `yfinance` |

All of these are selectable with `--analysis`; `--help` lists the values.

### Two ways to detect price manipulation

`price_divergence` works entirely offline: for each asset held by two or more
funds on the same date, it compares each fund's declared unit price
(`VL_MERCADO / QT_POS`) against the cross-fund median. It needs no external API
and covers unlisted credit — CRI, CRA, debêntures, cotas — which is where
mismarking is most likely.

`market_data` compares declared prices against real B3 closes via `yfinance`.
It is optional because yfinance scrapes Yahoo's undocumented endpoints and only
covers listed equities and ETFs. Install `requirements-optional.txt` to enable
it.

## Reports

```bash
python scripts/generate_public_report.py --format html --output reports/public.html
```

Produces an anonymized summary (fund CNPJs replaced with opaque IDs) in
Markdown, HTML or JSON.

## Tests and evals

```bash
pytest -q -m "not eval"   # unit, integration and property tests
pytest -q -m eval         # detection-quality evaluations
ruff check .
```

The eval suite is separate from the tests. It builds synthetic fund universes
with *labeled* injected fraud and measures each detector's precision, recall and
— most importantly — its false-positive rate against a clean universe, checked
against committed baselines in `evals/baseline.json`. A detector that fires on
clean data is worse than one that fires on nothing.

Synthetic fixtures have their own failure mode: model a column CVM does not
publish and the eval will happily score a detector that can never run on real
data. `evals/cvm_headers.json` records the real CVM headers, and
`tests/test_eval_fixture_schema.py` drives the actual readers over them to check
that no fixture column is invented and no scored detector is missing an input.

## Layout

```
config/       Settings and detection thresholds
src/
  collectors/ CVM downloads
  processors/ CSV parsing, normalization, flow derivation
  analyzers/  Detectors
  explain/    Findings -> human-readable briefs
  utils/      CNPJ handling, statistics, severity, caching
scripts/      CLI entry points
evals/        Detection-quality harness
notebooks/    Exploratory analysis
docs/         Architecture and guides
```

`src/utils/cnpj_utils.py` is the single source of truth for CNPJ handling. CNPJ
is the join key across every dataset here, so normalization lives in exactly one
place — see the module docstring for the rules.

## License

MIT. See [LICENSE](LICENSE).
