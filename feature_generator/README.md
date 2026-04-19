# Feature Generator

Generates ~200 features per session for a time-series stock prediction task using OHLCV candle data and headline template signals.

## Installation

```bash
pip install pandas numpy scipy ta --break-system-packages
```

## Usage

```python
import pandas as pd
from feature_generator import generate_features

# candles: 50-row DataFrame with columns [timestamp, open, high, low, close, volume]
# headlines: DataFrame with columns [candle_idx, template_id, amplitude]

features = generate_features(
    candles=candles_df,
    headlines=headlines_df,
    n_templates=30,
    recency_decay=0.05,
)
# features is a pd.Series with ~200 named float values
```

## Feature categories

| Category              | Count |
|-----------------------|-------|
| Candle price-level    | 32    |
| Candle returns        | 20    |
| Candle trend          | 12    |
| Candle volatility     | 12    |
| Technical indicators  | 14    |
| Per-template          | 90    |
| Global headline       | 10    |
| Temporal integration  | 10    |
| **Total**             | **~200** |

## Running tests

```bash
pytest tests/test_generator.py -v
```

## Demo

```bash
python demo.py
```
