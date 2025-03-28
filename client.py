import asyncio
import json
import argparse
import datetime
import websockets
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Import the broker module
from broker import Broker, Order

class TradingStrategy:
    """
    Trading strategy that buys when price hits a low threshold
    and sells when price hits a high threshold.
    
    The strategy will attempt to build a position by buying at the low threshold
    and then selling at the high threshold to realize profits.
    """
    
    def __init__(self, symbol: str, high_threshold: float, low_threshold: float, 
                 shares_per_trade: float = 10, confirmation_seconds: int = 3):
        """
        Initialize the trading strategy.
        
        Args:
            symbol: Stock symbol to trade
            high_threshold: Price threshold to sell
            low_threshold: Price threshold to buy
            shares_per_trade: Number of shares to trade each time
            confirmation_seconds: Seconds to wait before executing to confirm price movement
        """
        self.symbol = symbol
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.shares_per_trade = shares_per_trade
        self.confirmation_seconds = confirmation_seconds
        
        # Trading state
        self.waiting_high = False
        self.waiting_low = False
        self.high_trigger_time = None
        self.low_trigger_time = None
        
        # Price history for tracking
        self.price_history = []
        self.max_history_length = 1000  # Keep last 1000 prices
        
        # Performance tracking
        self.trades_executed = 0
        self.buys_executed = 0
        self.sells_executed = 0
        
        # Set up logging
        self.logger = logging.getLogger('strategy')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('strategy.log')
        file_handler.setFormatter(log_formatter)
        self.logger.addHandler(file_handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)
    
    def update_price(self, price: float, timestamp: datetime.datetime) -> Tuple[bool, str, float]:
        """
        Process a new price update.
        
        Args:
            price: Current price
            timestamp: Current timestamp
            
        Returns:
            Tuple of (should_trade, order_type, price)
        """
        # Add to price history
        self.price_history.append((timestamp, price))
        if len(self.price_history) > self.max_history_length:
            self.price_history.pop(0)
        
        # Check for high threshold crossing
        if price >= self.high_threshold and not self.waiting_high:
            self.waiting_high = True
            self.high_trigger_time = timestamp
            self.logger.info(f"High threshold {self.high_threshold} crossed at {price}. Waiting for confirmation...")
        
        # Check for low threshold crossing
        if price <= self.low_threshold and not self.waiting_low:
            self.waiting_low = True
            self.low_trigger_time = timestamp
            self.logger.info(f"Low threshold {self.low_threshold} crossed at {price}. Waiting for confirmation...")
        
        # Check if high threshold has been confirmed
        should_trade = False
        order_type = ""
        trade_price = 0.0
        
        if self.waiting_high and (timestamp - self.high_trigger_time).total_seconds() >= self.confirmation_seconds:
            # Execute sell
            self.waiting_high = False
            should_trade = True
            order_type = "SELL"
            trade_price = price
            self.logger.info(f"High threshold confirmed. Selling at {price}")
        
        # Check if low threshold has been confirmed
        if self.waiting_low and (timestamp - self.low_trigger_time).total_seconds() >= self.confirmation_seconds:
            # Execute buy
            self.waiting_low = False
            should_trade = True
            order_type = "BUY"
            trade_price = price
            self.logger.info(f"Low threshold confirmed. Buying at {price}")
        
        return should_trade, order_type, trade_price
    
    def track_execution(self, order_type: str):
        """Track successful order execution."""
        self.trades_executed += 1
        
        if order_type == "BUY":
            self.buys_executed += 1
        elif order_type == "SELL":
            self.sells_executed += 1
    
    def get_statistics(self) -> Dict:
        """Get trading statistics."""
        return {
            "symbol": self.symbol,
            "high_threshold": self.high_threshold,
            "low_threshold": self.low_threshold,
            "shares_per_trade": self.shares_per_trade,
            "confirmation_seconds": self.confirmation_seconds,
            "trades_executed": self.trades_executed,
            "buys_executed": self.buys_executed,
            "sells_executed": self.sells_executed,
            "price_history_length": len(self.price_history),
            "current_price": self.price_history[-1][1] if self.price_history else None,
            "waiting_high": self.waiting_high,
            "waiting_low": self.waiting_low
        }


class AutomatedTradingClient:
    """Client that automatically trades based on a trading strategy."""
    
    def __init__(self, broker: Broker, strategy: TradingStrategy):
        """
        Initialize the trading client.
        
        Args:
            broker: Broker instance to execute trades
            strategy: Trading strategy to use
        """
        self.broker = broker
        self.strategy = strategy
        self.running = False
        self.start_time = None
        self.end_time = None
        
        # Set up logging
        self.logger = logging.getLogger('trading_client')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('trading_client.log')
        file_handler.setFormatter(log_formatter)
        self.logger.addHandler(file_handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)
    
    async def start_trading(self):
        """Start automated trading."""
        if self.running:
            self.logger.warning("Trading already running")
            return
        
        self.running = True
        self.start_time = datetime.datetime.now()
        
        # Connect to the broker
        symbols = await self.broker.connect()
        
        if not symbols:
            self.logger.error("No symbols available")
            self.running = False
            return
        
        # Subscribe to the strategy symbol
        if self.strategy.symbol not in symbols:
            self.logger.error(f"Symbol {self.strategy.symbol} not available")
            self.running = False
            return
        
        success = await self.broker.subscribe(self.strategy.symbol)
        if not success:
            self.logger.error(f"Failed to subscribe to {self.strategy.symbol}")
            self.running = False
            return
        
        self.logger.info(f"Starting automated trading for {self.strategy.symbol}")
        self.logger.info(f"Buy threshold: ${self.strategy.low_threshold:.2f}")
        self.logger.info(f"Sell threshold: ${self.strategy.high_threshold:.2f}")
        self.logger.info(f"Shares per trade: {self.strategy.shares_per_trade}")
        self.logger.info(f"Confirmation time: {self.strategy.confirmation_seconds} seconds")
        
        # Start listening for price updates
        await self.broker.listen_for_updates(self.handle_price_update)
    
    async def handle_price_update(self, data: Dict):
        """
        Handle a price update from the broker.
        
        Args:
            data: Price update data
        """
        if not self.running:
            return
        
        if data["type"] == "price_update" and data["symbol"] == self.strategy.symbol:
            price = data["price"]
            timestamp = datetime.datetime.fromisoformat(data["timestamp"])
            
            # Update strategy with new price
            should_trade, order_type, trade_price = self.strategy.update_price(price, timestamp)
            
            # Execute trade if needed
            if should_trade:
                try:
                    # Check if we have shares before trying to sell
                    if order_type == "SELL":
                        # Get the current position
                        position = self.broker.account.get_position(self.strategy.symbol)
                        if not position or position.quantity < self.strategy.shares_per_trade:
                            self.logger.warning(f"Cannot SELL - Insufficient shares. Have {position.quantity if position else 0}, need {self.strategy.shares_per_trade}")
                            return
                    
                    # Check if we have enough cash before trying to buy
                    if order_type == "BUY":
                        required_cash = self.strategy.shares_per_trade * trade_price
                        if self.broker.account.cash < required_cash:
                            self.logger.warning(f"Cannot BUY - Insufficient cash. Have ${self.broker.account.cash:.2f}, need ${required_cash:.2f}")
                            return
                    
                    # Place the order
                    order = self.broker.place_order(
                        self.strategy.symbol,
                        order_type,
                        self.strategy.shares_per_trade,
                        trade_price
                    )
                    
                    if order:
                        # Execute the order
                        fill_details = self.broker.execute_order(order.order_id, trade_price)
                        if "error" not in fill_details:
                            self.strategy.track_execution(order_type)
                            self.logger.info(f"Executed {order_type} order for {self.strategy.shares_per_trade} shares at ${trade_price:.2f}")
                        else:
                            self.logger.error(f"Failed to execute order: {fill_details['error']}")
                    else:
                        self.logger.error(f"Failed to place {order_type} order")
                
                except Exception as e:
                    self.logger.error(f"Error executing trade: {e}")
    
    async def stop_trading(self):
        """Stop automated trading."""
        if not self.running:
            self.logger.warning("Trading not running")
            return
        
        self.running = False
        self.end_time = datetime.datetime.now()
        
        # Print trading summary
        duration = self.end_time - self.start_time
        
        self.logger.info("Trading stopped")
        self.logger.info(f"Trading duration: {duration}")
        self.logger.info(f"Total trades executed: {self.strategy.trades_executed}")
        self.logger.info(f"Buys executed: {self.strategy.buys_executed}")
        self.logger.info(f"Sells executed: {self.strategy.sells_executed}")
        
        # Print account summary
        self.broker.print_account_summary()
        
        # Reset the account and log
        self.broker.reset_account()
        
        # Disconnect from the broker
        await self.broker.disconnect()


async def run_automated_trading(symbol: str, high_threshold: float, low_threshold: float, 
                                shares_per_trade: float = 10, confirmation_seconds: int = 3,
                                duration_minutes: int = 30):
    """
    Run automated trading for a specified duration.
    
    Args:
        symbol: Stock symbol to trade
        high_threshold: Price threshold for selling
        low_threshold: Price threshold for buying
        shares_per_trade: Number of shares to trade each time
        confirmation_seconds: Seconds to wait before executing to confirm price movement
        duration_minutes: How long to run trading for (in minutes)
    """
    # Create broker
    broker = Broker(account_id="auto_trader")
    
    # Create strategy
    strategy = TradingStrategy(
        symbol=symbol,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        shares_per_trade=shares_per_trade,
        confirmation_seconds=confirmation_seconds
    )
    
    # Create client
    client = AutomatedTradingClient(broker, strategy)
    
    # Start trading
    trading_task = asyncio.create_task(client.start_trading())
    
    try:
        # Run for specified duration
        await asyncio.sleep(duration_minutes * 60)
    finally:
        # Stop trading
        await client.stop_trading()
        
        # Wait for trading task to complete
        try:
            await trading_task
        except asyncio.CancelledError:
            pass


async def interactive_setup():
    """Interactive setup for automated trading."""
    print("=== Automated Trading Client Setup ===")
    
    # Create broker and connect
    broker = Broker()
    symbols = await broker.connect()
    
    if not symbols:
        print("No symbols available for trading")
        await broker.disconnect()
        return
    
    print(f"Available symbols: {', '.join(symbols)}")
    
    # Get trading parameters
    while True:
        symbol = input(f"Enter symbol to trade ({symbols[0]}): ") or symbols[0]
        if symbol in symbols:
            break
        print(f"Invalid symbol. Available symbols: {', '.join(symbols)}")
    
    # Get current price to help set thresholds
    await broker.subscribe(symbol)
    current_price = await broker.get_price(symbol)
    
    print(f"Current price of {symbol}: ${current_price:.2f}")
    
    # Get high and low thresholds
    suggested_low = current_price * 0.99  # 1% below current price
    suggested_high = current_price * 1.01  # 1% above current price
    
    low_threshold = float(input(f"Enter low threshold for buying (${suggested_low:.2f}): ") or suggested_low)
    high_threshold = float(input(f"Enter high threshold for selling (${suggested_high:.2f}): ") or suggested_high)
    
    # Get other parameters
    shares = float(input("Enter shares per trade (10): ") or 10)
    confirmation_seconds = int(input("Enter confirmation seconds (3): ") or 3)
    duration_minutes = int(input("Enter trading duration in minutes (30): ") or 30)
    
    print("\n=== Trading Parameters ===")
    print(f"Symbol: {symbol}")
    print(f"Buy threshold: ${low_threshold:.2f}")
    print(f"Sell threshold: ${high_threshold:.2f}")
    print(f"Shares per trade: {shares}")
    print(f"Confirmation time: {confirmation_seconds} seconds")
    print(f"Duration: {duration_minutes} minutes")
    
    confirm = input("\nConfirm these settings? (y/n): ").lower()
    if confirm != 'y':
        print("Setup cancelled")
        await broker.disconnect()
        return
    
    # Disconnect broker used for setup
    await broker.disconnect()
    
    # Run trading with confirmed parameters
    print("\nStarting automated trading...")
    await run_automated_trading(
        symbol=symbol,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        shares_per_trade=shares,
        confirmation_seconds=confirmation_seconds,
        duration_minutes=duration_minutes
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Automated Stock Trading Client")
    parser.add_argument("--symbol", type=str, help="Stock symbol to trade")
    parser.add_argument("--high", type=float, help="High price threshold for selling")
    parser.add_argument("--low", type=float, help="Low price threshold for buying")
    parser.add_argument("--shares", type=float, default=10, help="Shares per trade (default: 10)")
    parser.add_argument("--confirm", type=int, default=3, help="Confirmation seconds (default: 3)")
    parser.add_argument("--duration", type=int, default=30, help="Trading duration in minutes (default: 30)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    
    try:
        if args.interactive or not (args.symbol and args.high and args.low):
            # Run in interactive mode
            asyncio.run(interactive_setup())
        else:
            # Run with command line arguments
            asyncio.run(run_automated_trading(
                symbol=args.symbol,
                high_threshold=args.high,
                low_threshold=args.low,
                shares_per_trade=args.shares,
                confirmation_seconds=args.confirm,
                duration_minutes=args.duration
            ))
    except KeyboardInterrupt:
        print("\nTrading stopped by user")