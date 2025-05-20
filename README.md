# Quant Finance Trading Simulator

A Python-based trading simulator designed for testing quantitative finance concepts. This project allows you to simulate stock price movements, implement trading strategies, and evaluate their performance without risking real capital.

## Components

- **sim.py** - The market simulator that generates realistic stock price movements using geometric Brownian motion
- **broker.py** - Handles order execution, position tracking, and portfolio management
- **client.py** - Implements trading strategies and automated trading logic

## Installation

```bash
# Clone the repository
git clone https://github.com/amirzarandi/trading-sim.git
cd trading-sim

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Start the Price Simulator

First, start the WebSocket server that simulates stock prices:

```bash
python sim.py
```

This will start a server on `ws://localhost:8765` with simulated prices for AAPL, MSFT, GOOGL, and AMZN.

### Run the Trading Client

There are two ways to run the trading client:

#### Interactive Mode

```bash
python client.py --interactive
```

This will prompt you to select:
- Which stock to trade
- Buy and sell thresholds
- Number of shares per trade
- Confirmation time
- Trading duration

#### Command Line Mode

```bash
python client.py --symbol AAPL --high 180.5 --low 170.5 --shares 10 --confirm 3 --duration 30
```

Parameters:
- `--symbol`: Stock symbol to trade
- `--high`: High price threshold for selling
- `--low`: Low price threshold for buying
- `--shares`: Shares per trade (default: 10)
- `--confirm`: Confirmation seconds (default: 3)
- `--duration`: Trading duration in minutes (default: 30)

## Features

### Price Simulator (sim.py)
- Supports multiple stocks with different characteristics
- Provides real-time price updates via WebSockets
- Includes Monte Carlo simulation for future price prediction

### Broker (broker.py)
- Order execution (market and limit orders)
- Position tracking with cost basis calculation
- Portfolio valuation and performance metrics
- P&L tracking (realized and unrealized)
- Order history and account reset functionality

### Trading Client (client.py)
- Simple threshold-based trading strategy
- Customizable trading parameters
- Performance tracking and reporting
- Interactive setup mode
- Automated trading for a specified duration

## Advanced Usage

You can modify the `TradingStrategy` class in client.py to implement your own quantitative trading strategies. The system is designed to be modular, allowing you to plug in different strategies while reusing the broker and simulator components.