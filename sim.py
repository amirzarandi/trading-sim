import asyncio
import datetime
import json
import math
import random
import numpy as np
import websockets
from typing import Dict, List, Optional

class StockPriceSimulator:
    """
    Simulates stock price movements using geometric Brownian motion,
    a common model for stock price behavior in financial mathematics.
    """
    def __init__(self, 
                 symbol: str, 
                 initial_price: float, 
                 annual_drift: float = 0.05,  # Expected annual return
                 annual_volatility: float = 0.2,  # Annual volatility
                 trading_days: int = 252):  # Trading days per year
        """
        Initialize the stock price simulator.
        
        Args:
            symbol: Stock ticker symbol
            initial_price: Starting price of the stock
            annual_drift: Expected annual return (e.g., 0.05 for 5%)
            annual_volatility: Annual volatility (e.g., 0.2 for 20%)
            trading_days: Number of trading days per year
        """
        self.symbol = symbol
        self.current_price = initial_price
        self.initial_price = initial_price
        self.annual_drift = annual_drift
        self.annual_volatility = annual_volatility
        self.trading_days = trading_days
        
        # Derived parameters
        self.daily_drift = annual_drift / trading_days
        self.daily_volatility = annual_volatility / math.sqrt(trading_days)
        
        # Price history
        self.price_history = [initial_price]
        self.timestamp_history = [datetime.datetime.now()]
        
        # Connected clients
        self.connected_clients = set()
    
    def get_current_price(self) -> float:
        """Get the current stock price."""
        return self.current_price
    
    def get_price_history(self) -> Dict:
        """Get the price history with timestamps."""
        return {
            "symbol": self.symbol,
            "prices": self.price_history,
            "timestamps": [ts.isoformat() for ts in self.timestamp_history]
        }
    
    def step(self) -> float:
        """
        Generate the next price using geometric Brownian motion.
        
        Returns:
            The new stock price
        """
        # Generate random normal return
        random_return = np.random.normal(
            loc=self.daily_drift, 
            scale=self.daily_volatility
        )
        
        # Apply to current price using log-normal model
        self.current_price *= math.exp(random_return)
        
        # Record new price and timestamp
        self.price_history.append(self.current_price)
        self.timestamp_history.append(datetime.datetime.now())
        
        return self.current_price
    
    def run_simulation(self, days: int) -> List[float]:
        """
        Run the simulation for a specified number of days.
        
        Args:
            days: Number of days to simulate
            
        Returns:
            List of simulated prices
        """
        simulated_prices = [self.current_price]
        current_price = self.current_price
        
        for _ in range(days):
            random_return = np.random.normal(
                loc=self.daily_drift, 
                scale=self.daily_volatility
            )
            current_price *= math.exp(random_return)
            simulated_prices.append(current_price)
        
        return simulated_prices
    
    def monte_carlo_simulation(self, days: int, num_simulations: int = 1000) -> Dict:
        """
        Run multiple simulations to generate potential price paths.
        
        Args:
            days: Number of days to simulate
            num_simulations: Number of Monte Carlo paths to generate
            
        Returns:
            Dictionary with simulation results
        """
        all_simulations = []
        
        for _ in range(num_simulations):
            simulation = self.run_simulation(days)
            all_simulations.append(simulation)
        
        # Convert to numpy array for easier analysis
        simulations_array = np.array(all_simulations)
        
        # Calculate statistics
        mean_path = np.mean(simulations_array, axis=0)
        median_path = np.median(simulations_array, axis=0)
        upper_95 = np.percentile(simulations_array, 95, axis=0)
        lower_5 = np.percentile(simulations_array, 5, axis=0)
        
        return {
            "symbol": self.symbol,
            "days": days,
            "num_simulations": num_simulations,
            "initial_price": self.initial_price,
            "mean_path": mean_path.tolist(),
            "median_path": median_path.tolist(),
            "upper_95": upper_95.tolist(),
            "lower_5": lower_5.tolist(),
            "sample_paths": simulations_array[:5].tolist()  # Include a few sample paths
        }

    async def register_client(self, websocket):
        """Register a new WebSocket client."""
        self.connected_clients.add(websocket)
        
    async def unregister_client(self, websocket):
        """Unregister a WebSocket client."""
        self.connected_clients.remove(websocket)
        
    async def notify_clients(self):
        """Send price updates to all connected clients."""
        if not self.connected_clients:
            return
            
        # Prepare the message
        message = json.dumps({
            "type": "price_update",
            "symbol": self.symbol,
            "price": self.current_price,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Send to all clients
        await asyncio.gather(
            *[client.send(message) for client in self.connected_clients],
            return_exceptions=True
        )


class StockMarketSimulator:
    """
    Manages multiple stock simulators and provides a centralized websocket interface.
    """
    def __init__(self):
        """Initialize the stock market simulator."""
        self.stocks = {}  # Dictionary to store stock simulators by symbol
        self.stock_to_clients = {}  # Maps stock symbols to connected clients
        self.client_to_stocks = {}  # Maps clients to their subscribed stock symbols
    
    def add_stock(self, symbol: str, initial_price: float, 
                  annual_drift: float = 0.05, annual_volatility: float = 0.2):
        """Add a new stock to the simulation."""
        self.stocks[symbol] = StockPriceSimulator(
            symbol=symbol,
            initial_price=initial_price,
            annual_drift=annual_drift,
            annual_volatility=annual_volatility
        )
        self.stock_to_clients[symbol] = set()
        return self.stocks[symbol]
    
    def get_stock(self, symbol: str) -> Optional[StockPriceSimulator]:
        """Get a stock simulator by symbol."""
        return self.stocks.get(symbol)
    
    def get_available_symbols(self) -> List[str]:
        """Get a list of all available stock symbols."""
        return list(self.stocks.keys())
    
    async def register_client(self, websocket, symbol: str):
        """Register a client to receive updates for a specific stock."""
        if symbol not in self.stocks:
            return False
        
        # Add to mappings
        if websocket not in self.client_to_stocks:
            self.client_to_stocks[websocket] = set()
        
        self.stock_to_clients[symbol].add(websocket)
        self.client_to_stocks[websocket].add(symbol)
        return True
    
    async def unregister_client(self, websocket):
        """Unregister a client from all subscribed stocks."""
        if websocket not in self.client_to_stocks:
            return
        
        # Remove from all subscribed stocks
        for symbol in self.client_to_stocks[websocket]:
            self.stock_to_clients[symbol].remove(websocket)
        
        # Remove client from mapping
        del self.client_to_stocks[websocket]
    
    async def update_prices(self):
        """Update prices for all stocks."""
        for symbol, stock in self.stocks.items():
            # Update the price
            stock.step()
            
            # Print to terminal
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {symbol}: ${stock.current_price:.2f}")
            
            # Notify clients subscribed to this stock
            if self.stock_to_clients[symbol]:
                message = json.dumps({
                    "type": "price_update",
                    "symbol": symbol,
                    "price": stock.current_price,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                await asyncio.gather(
                    *[client.send(message) for client in self.stock_to_clients[symbol]],
                    return_exceptions=True
                )


# WebSocket handler for the simulation
async def websocket_handler(websocket, path, market_simulator):
    """Handle WebSocket connections for stock price updates."""
    subscribed_symbol = None
    
    try:
        # Send available symbols first
        await websocket.send(json.dumps({
            "type": "available_symbols",
            "symbols": market_simulator.get_available_symbols()
        }))
        
        # Listen for commands from clients
        async for message in websocket:
            try:
                data = json.loads(message)
                command = data.get("command")
                
                if command == "subscribe":
                    symbol = data.get("symbol")
                    if not symbol:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Missing symbol parameter"
                        }))
                        continue
                    
                    # Try to register client for this stock
                    success = await market_simulator.register_client(websocket, symbol)
                    if success:
                        subscribed_symbol = symbol
                        stock = market_simulator.get_stock(symbol)
                        
                        # Send initial data for the subscribed stock
                        await websocket.send(json.dumps({
                            "type": "initial_data",
                            "symbol": symbol,
                            "price": stock.current_price,
                            "history": stock.get_price_history()
                        }))
                    else:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"Invalid symbol: {symbol}"
                        }))
                
                elif command == "get_price":
                    symbol = data.get("symbol", subscribed_symbol)
                    if not symbol:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "No symbol specified or subscribed"
                        }))
                        continue
                    
                    stock = market_simulator.get_stock(symbol)
                    if not stock:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"Invalid symbol: {symbol}"
                        }))
                        continue
                    
                    await websocket.send(json.dumps({
                        "type": "price_data",
                        "symbol": symbol,
                        "price": stock.current_price
                    }))
                    
                elif command == "get_history":
                    symbol = data.get("symbol", subscribed_symbol)
                    if not symbol:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "No symbol specified or subscribed"
                        }))
                        continue
                    
                    stock = market_simulator.get_stock(symbol)
                    if not stock:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"Invalid symbol: {symbol}"
                        }))
                        continue
                    
                    await websocket.send(json.dumps({
                        "type": "history_data",
                        "data": stock.get_price_history()
                    }))
                    
                elif command == "simulate":
                    symbol = data.get("symbol", subscribed_symbol)
                    if not symbol:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "No symbol specified or subscribed"
                        }))
                        continue
                    
                    stock = market_simulator.get_stock(symbol)
                    if not stock:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"Invalid symbol: {symbol}"
                        }))
                        continue
                    
                    days = data.get("days", 30)
                    num_sims = data.get("num_simulations", 1000)
                    results = stock.monte_carlo_simulation(days, num_sims)
                    
                    await websocket.send(json.dumps({
                        "type": "simulation_results",
                        "data": results
                    }))
                
                elif command == "list_symbols":
                    await websocket.send(json.dumps({
                        "type": "available_symbols",
                        "symbols": market_simulator.get_available_symbols()
                    }))
                    
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Unknown command: {command}"
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
    
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        await market_simulator.unregister_client(websocket)


async def price_updater(market_simulator, interval=1.0):
    """Periodically update all stock prices and notify subscribed clients."""
    while True:
        # Update all prices and notify clients
        await market_simulator.update_prices()
        
        # Wait for the next update
        await asyncio.sleep(interval)


async def main():
    # Create the stock market simulator
    market = StockMarketSimulator()
    
    # Add multiple stocks with different characteristics
    market.add_stock(
        symbol="AAPL",
        initial_price=175.50,
        annual_drift=0.08,  # 8% expected annual return
        annual_volatility=0.25  # 25% annual volatility
    )
    
    market.add_stock(
        symbol="MSFT",
        initial_price=390.25,
        annual_drift=0.10,  # 10% expected annual return
        annual_volatility=0.22  # 22% annual volatility
    )
    
    market.add_stock(
        symbol="GOOGL",
        initial_price=140.75,
        annual_drift=0.09,  # 9% expected annual return
        annual_volatility=0.28  # 28% annual volatility
    )
    
    market.add_stock(
        symbol="AMZN",
        initial_price=178.25,
        annual_drift=0.11,  # 11% expected annual return
        annual_volatility=0.30  # 30% annual volatility
    )
    
    # Set up the WebSocket server
    host = "localhost"
    port = 8765
    
    # Start the WebSocket server
    async with websockets.serve(
        lambda ws, path: websocket_handler(ws, path, market),
        host, port
    ):
        print(f"WebSocket server started at ws://{host}:{port}")
        print(f"Simulating prices for {', '.join(market.get_available_symbols())}")
        
        # Start the price updater
        await price_updater(market)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")