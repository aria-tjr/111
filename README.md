# NEXUS — 7-Layer ML Crypto Signal Engine

NEXUS is a multi-layered machine-learning framework for cryptocurrency market analysis and signal generation. It combines regime detection, deep learning, and multi-timeframe analysis to produce high-confidence trading signals.

## Architecture — 7 Signal Layers

| # | Layer | Description |
|---|---|---|
| 1 | Regime Detection | Market regime classification (trending / ranging / volatile) |
| 2 | Momentum Analysis | Multi-timeframe momentum scoring with EMA/RSI confluence |
| 3 | Volume Profile | Order flow and volume distribution analysis |
| 4 | ML Classifier | PyTorch neural network for pattern classification |
| 5 | AltFins Integration | External signal enrichment from AltFins API |
| 6 | Sentiment Layer | On-chain and social sentiment aggregation |
| 7 | Scoring Engine | Weighted ensemble of all layers → final signal score |

## Features

- **PyTorch deep learning** — custom neural network for price-pattern recognition
- **Binance API integration** — real-time and historical OHLCV data ingestion
- **AltFins signal enrichment** — third-party signal validation layer
- **Backtesting suite** — historical signal validation across multiple market regimes
- **Regime awareness** — signal weights adapt to current market conditions
- **Multi-asset scanning** — parallel scanning across 50+ cryptocurrency pairs
- **Output formats** — JSON signals, CSV exports, visualization charts

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch |
| Data Processing | pandas, numpy |
| Exchange Data | Binance API (python-binance) |
| Technical Analysis | ta-lib, pandas-ta |
| Visualization | matplotlib, plotly |
| Config Management | Python dataclasses |
| Testing | pytest |

## Project Structure

```
nexus/
├── config.py        # Configuration (API keys via environment variables)
├── analyze.py       # Main analysis orchestrator
├── scanner/         # Multi-asset market scanner
├── layers/          # 7 signal layers (each in separate module)
├── ml/              # PyTorch model definitions and training
├── models/          # Saved model checkpoints
├── scoring/         # Ensemble scoring and signal generation
├── data/            # Data fetching and caching
└── output/          # Signal exports and charts
```

## Setup

```bash
git clone https://github.com/aria-tjr/111.git nexus
cd nexus
pip install -r requirements.txt

export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_SECRET="your_binance_secret"
export ALTFINS_API_KEY="your_altfins_key"   # optional

python analyze.py
```

All API keys are loaded from environment variables. See `config.py` for the full configuration schema. **Never hardcode credentials.**

## Signal Output

Each generated signal includes:

- Asset pair (e.g. `BTC/USDT`)
- Direction (`LONG` / `SHORT` / `NEUTRAL`)
- Confidence score (0–100)
- Contributing layer scores
- Recommended entry zone
- Invalidation level

## Disclaimer

This tool is provided for research and educational purposes. Cryptocurrency markets are highly volatile; signals are not financial advice. Never trade more than you can afford to lose.

## License

MIT
