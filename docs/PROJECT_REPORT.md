# FlowRisk

Project report | Harsh Saand | 22 September 2026

## The problem

Short-term volume and volatility change rapidly, so a complex forecast should be tested against an adaptive statistical baseline. FlowRisk reconstructs minute-level features from public trades and evaluates future forecasts chronologically.

## What a user gets

A reader receives timestamped variance and quote-volume forecasts with uncertainty ranges, source provenance and error diagnostics.

## Practical value

The recorded experiment supports retaining EWMA over the learned alternatives in this setting. The value is a testable decision about model complexity and unreliable intervals, not a trading-profit claim.

## Logic and flow

```mermaid
flowchart TD
  N0["Public aggregate-trade archives"]
  N1["Checksum + minute aggregation"]
  N2["Past-only features; purged time splits"]
  N3["Boosted forecasts versus EWMA"]
  N4["Forecast records + interval diagnostics"]
  N0 --> N1
  N1 --> N2
  N2 --> N3
  N3 --> N4
```

<details>
<summary><strong>Dataset at a glance</strong></summary>

The saved run downloads **18 Binance public spot aggregate-trade archives**: BTCUSDT and ETHUSDT, three selected days in each of June, July and August. One raw row is an aggregate-trade event with price, quantity, timestamp and trade-side information; not an order-book snapshot. **15,810,312 events** aggregate into **25,920 symbol-minute observations**, from which past-minute features predict future variance and quote volume.

After warm-up and incomplete-target removal, the chronological partitions contain **8,512 training / 8,512 development / 8,512 test rows**. They use June 1-3, July 1-3 and August 1-3, respectively: nine sampled days, not a continuous quarter. Overlapping future targets are correlated. Archive checksums and download details are in [`outputs/provenance.json`](https://github.com/HarshSaand/flowrisk/blob/369ea8724f5121c8114baa63c9ba1a7e155f3478/outputs/provenance.json).

</details>

<details>
<summary><strong>Technical snapshot</strong></summary>

| Question | Implementation |
|---|---|
| What is predicted? | Next-minute squared log trade-price return sum; next-five-minute USDT quote volume |
| What is trained? | Histogram gradient-boosted positive-mean and 5th/50th/95th quantile regressors |
| Baselines | Historical 60-minute mean and 15-span EWMA |
| Data | Actual BTCUSDT and ETHUSDT spot aggregate trades from Binance's public daily archive |
| Leakage controls | Features use fully closed minutes only; contiguous-block warm-up; future labels; chronological partitions and six-minute label-end purge |
| Evaluation | MAE, RMSE, QLIKE for positive variance forecasts, pinball losses, interval coverage and paired UTC-day block uncertainty |
| Current scope | First three contiguous days of June, July and August 2026; not all three months |

</details>

<details>
<summary><strong>Architecture</strong></summary>

### Data provenance and bounded scope

The [official Binance public-data repository](https://github.com/binance/binance-public-data) documents freely downloadable daily aggregate-trade archives, their columns and `.CHECKSUM` files. This run downloads **18 archives: two symbols × three days × three months**, with a 250 MiB compressed-download cap. Seven days per month were considered in a HEAD-only preflight but required about 603 MiB; the three-day scope was fixed **before training or viewing any forecast results**. The completed selection is June 1-3 for training, July 1-3 for development and August 1-3 for test. It is not a full-August or continuous-quarter benchmark.

All publisher archive checksums are validated. Exact URLs, sizes and SHA-256 values are stored in `outputs/provenance.json`. Raw downloads and minute/prediction data stay local and are excluded from Git. Spot timestamps from 2025 onward are microseconds, per the publisher's schema; this implementation uses that contract explicitly.

### Aggregation and causal feature construction

CSV processing uses 250,000-row chunks. The sum of squared consecutive log aggregate-trade-price returns forms each minute's variance proxy; cross-chunk price continuity is preserved. The first event of each daily archive has no preceding price and contributes zero return. This proxy is affected by trade-price microstructure noise and is **not** an efficient-midprice realized-variance estimate. A maker-buyer flag marks seller-initiated aggregate flow; signed quote volume is a trade-flow proxy, not resting-book imbalance. Intensity counts aggregate-trade events, not individual matched fills.

At the close of minute `t`, features use minute `t` and prior minutes only: 1/5/15/60-minute variance and volume means, EWMA, signed quote-flow ratios, event intensity, UTC time and symbol identity. Targets use variance in `t+1` and quote volume in `t+1…t+5`. Minute timestamps represent interval opens, so the last target closes at timestamp `t+6`. Missing monthly intervals reset warm-up rather than becoming artificial consecutive observations. Five trailing rows lack complete future labels and are removed from each contiguous segment.

### Models and evaluation protocol

Two configurations; 7 versus 15 maximum leaves, 120 iterations, learning rate 0.05; are registered before evaluation. Mean models use Poisson deviance on rescaled nonnegative targets. July alone selects the variance configuration by QLIKE and volume configuration by MAE. Models are not refitted on July; August remains an out-of-time test. This first pilot is a **single chronological train/development/test exercise**, not a multi-fold walk-forward study.

Quantile regressors fit log-transformed targets at 0.05, 0.50 and 0.95 using the selected tree size. Row-wise rearrangement enforces monotonic quantiles; the raw outer-crossing rate is disclosed. A nominal 90% interval is not guaranteed to achieve 90% empirical coverage. No test-based calibration is performed. Paired block bootstrap resamples UTC days jointly across BTC and ETH; with only three held-out days, intervals are exploratory and cannot support strong significance claims.

</details>

<details>
<summary><strong>Measured results</strong></summary>

The completed run processes **15,810,312 real aggregate-trade events** from 18 checksum-verified archives, using **225.4 MiB compressed downloads**. They produce 25,920 symbol-minute observations. Warm-up and incomplete future-target removal leave **8,512 training / 8,512 development / 8,512 test rows**, with 4,256 held-out forecast origins per symbol. Overlapping five-minute targets are not independent examples.

| Model | Next-minute variance QLIKE ↓ | Next-five-minute volume MAE ↓ |
|---|---:|---:|
| Historical 60-minute mean | 1.7208 | 946,670 USDT |
| **15-span EWMA** | **1.5263** | **875,289 USDT** |
| Trained gradient boosting | 1.7888 | 1,856,157 USDT |

**The learned mean models underperform EWMA on both tasks.** July selected the seven-leaf configuration for each task; neither test table was used to change that choice. The learned-minus-EWMA day-block difference is **+0.2625 QLIKE** (exploratory 95% interval -0.0725 to +0.7019) and **+980,868 USDT volume MAE** (+532,802 to +1,438,302). Only three held-out daily blocks support these intervals, so they should not be treated as robust significance estimates.

The learned nominal 90% quantile intervals cover just **69.73% of variance targets** and **55.51% of volume targets**. There were no raw outer-quantile crossings. Pinball losses and widths are reported rather than assuming nominal coverage was achieved. Both point error and severe undercoverage matter for practical risk-aware scheduling.

The sampled periods differ sharply: average one-minute BTC quote volume falls from approximately **1.38 million USDT in the June sample to 0.46 million in August**, and ETH from **0.67 million to 0.18 million**. This is consistent with a substantial distribution shift, but the experiment does not isolate its cause or prove that shift alone explains the model failure. The adaptive baselines use recent fully closed observations while the learned model's parameters remain fixed after June training.



Evidence: [`results.json`](https://github.com/HarshSaand/flowrisk/blob/369ea8724f5121c8114baa63c9ba1a7e155f3478/outputs/results.json), [`all development attempts`](https://github.com/HarshSaand/flowrisk/blob/369ea8724f5121c8114baa63c9ba1a7e155f3478/outputs/experiments.json), [`split manifest`](https://github.com/HarshSaand/flowrisk/blob/369ea8724f5121c8114baa63c9ba1a7e155f3478/outputs/split-manifest.json), [`archive provenance`](https://github.com/HarshSaand/flowrisk/blob/369ea8724f5121c8114baa63c9ba1a7e155f3478/outputs/provenance.json) and [`aggregation counts`](https://github.com/HarshSaand/flowrisk/blob/369ea8724f5121c8114baa63c9ba1a7e155f3478/outputs/aggregation.json). Nine contract tests pass; the suite exercises future-target direction, feature causality, gap resets, purging and metric boundaries using synthetic unit inputs separate from real reported metrics.

</details>

<details>
<summary><strong>Limitations and next experiments</strong></summary>

- Only BTC/ETH at one crypto venue and the first three days of three months. Weekend/calendar effects and regime gaps limit generalization; no equity-market claim follows.
- The final test has only three independent daily blocks. Expand to the complete quarter and multiple rolling-origin folds before drawing robust statistical conclusions.
- Trades do not establish executable prices, spreads, book depth, queue priority, fees, slippage or market impact. Forecast quality is not trading profitability.
- USDT quote volume is not guaranteed equivalent to USD volume; all reported currency quantities remain denominated in USDT.
- A variance proxy from transaction prices contains microstructure noise; zero or tiny targets require a documented positive numerical floor in QLIKE.
- Static hyperparameters and one seed; no regime-specific retuning or alternative target definitions were selected on test results.
- Archive history can be revised; compare stored checksums before rerunning. Publisher data terms remain separate from this repository's original MIT-licensed code.

</details>

<details>
<summary><strong>Using the project</strong></summary>

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python flowrisk.py fetch --days 3 --max-mb 250
python flowrisk.py aggregate
python flowrisk.py train
python -m pytest -q
```

No account, API key, paid data, order submission or trading connection is required. Download concurrency is four network workers; numerical computation is limited to two CPU threads. Model checkpoints are local under `models/`; run training to recreate them. The exact environment is recorded in `requirements-lock.txt`.

</details>

## Evidence and reproduction references

Source revision: 369ea8724f5121c8114baa63c9ba1a7e155f3478

- [README.md](https://github.com/HarshSaand/flowrisk/blob/369ea8724f5121c8114baa63c9ba1a7e155f3478/README.md)
- [docs/output-example.json](https://github.com/HarshSaand/flowrisk/blob/369ea8724f5121c8114baa63c9ba1a7e155f3478/docs/output-example.json)
- [outputs/results.json](https://github.com/HarshSaand/flowrisk/blob/369ea8724f5121c8114baa63c9ba1a7e155f3478/outputs/results.json)

This report describes the source and saved evidence at the revision above. Training and full benchmark runs were not repeated for this documentation release. Dataset, model and dependency licences remain separate from the project documentation.
