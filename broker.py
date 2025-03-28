import asyncio
import json
import logging
import datetime
import websockets
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

class Order:
    """Represents a stock market order."""
    
    def __init__(self, symbol: str, order_type: str, quantity: float, price: float):
        """
        Initialize a new order.
        
        Args:
            symbol: Stock symbol
            order_type: "BUY" or "SELL"
            quantity: Number of shares
            price: Order price
        """
        self.symbol = symbol
        self.order_type = order_type.upper()  # BUY or SELL
        self.quantity = quantity
        self.price = price
        self.order_id = f"{order_type}-{symbol}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.timestamp = datetime.datetime.now()
        self.status = "PENDING"  # PENDING, FILLED, CANCELLED, REJECTED
        self.filled_price = None
        self.filled_timestamp = None
    
    def to_dict(self) -> Dict:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "filled_price": self.filled_price,
            "filled_timestamp": self.filled_timestamp.isoformat() if self.filled_timestamp else None
        }
    
    def __str__(self) -> str:
        """String representation of the order."""
        return (f"Order {self.order_id}: {self.order_type} {self.quantity} {self.symbol} "
                f"@ ${self.price:.2f} - {self.status}")


class Position:
    """Represents a position in a stock."""
    
    def __init__(self, symbol: str):
        """Initialize a new position."""
        self.symbol = symbol
        self.quantity = 0.0
        self.cost_basis = 0.0
        self.transactions = []  # List of dictionaries with transaction details
    
    def update_from_fill(self, order: Order, fill_price: float):
        """Update position based on a filled order."""
        timestamp = datetime.datetime.now()
        
        if order.order_type == "BUY":
            # Calculate new cost basis with weighted average
            old_value = self.quantity * self.cost_basis
            new_value = order.quantity * fill_price
            self.quantity += order.quantity
            
            if self.quantity > 0:
                self.cost_basis = (old_value + new_value) / self.quantity
            
            transaction = {
                "timestamp": timestamp.isoformat(),
                "type": "BUY",
                "quantity": order.quantity,
                "price": fill_price,
                "value": order.quantity * fill_price
            }
        else:  # SELL
            self.quantity -= order.quantity
            
            # Don't update cost basis on sell
            transaction = {
                "timestamp": timestamp.isoformat(),
                "type": "SELL",
                "quantity": order.quantity,
                "price": fill_price,
                "value": order.quantity * fill_price
            }
        
        self.transactions.append(transaction)
        return transaction
    
    def market_value(self, current_price: float) -> float:
        """Calculate the current market value of the position."""
        return self.quantity * current_price
    
    def unrealized_pl(self, current_price: float) -> float:
        """Calculate unrealized profit/loss."""
        return self.quantity * (current_price - self.cost_basis)
    
    def realized_pl(self) -> float:
        """Calculate realized profit/loss from all closed transactions."""
        # Implement FIFO (First In, First Out) method
        buy_queue = []  # (quantity, price) tuples
        realized_pl = 0.0
        
        for tx in self.transactions:
            if tx["type"] == "BUY":
                buy_queue.append((tx["quantity"], tx["price"]))
            elif tx["type"] == "SELL":
                remaining_to_sell = tx["quantity"]
                sell_price = tx["price"]
                
                while remaining_to_sell > 0 and buy_queue:
                    buy_quantity, buy_price = buy_queue[0]
                    
                    if buy_quantity <= remaining_to_sell:
                        # Use entire buy lot
                        realized_pl += buy_quantity * (sell_price - buy_price)
                        remaining_to_sell -= buy_quantity
                        buy_queue.pop(0)
                    else:
                        # Use partial buy lot
                        realized_pl += remaining_to_sell * (sell_price - buy_price)
                        buy_queue[0] = (buy_quantity - remaining_to_sell, buy_price)
                        remaining_to_sell = 0
        
        return realized_pl
    
    def to_dict(self, current_price: Optional[float] = None) -> Dict:
        """Convert position to dictionary."""
        result = {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "cost_basis": self.cost_basis,
            "transactions": self.transactions
        }
        
        if current_price is not None:
            result.update({
                "current_price": current_price,
                "market_value": self.market_value(current_price),
                "unrealized_pl": self.unrealized_pl(current_price),
                "realized_pl": self.realized_pl(),
                "total_pl": self.unrealized_pl(current_price) + self.realized_pl()
            })
        
        return result


class Account:
    """Represents a trading account."""
    
    def __init__(self, account_id: str, initial_cash: float = 100000.0):
        """Initialize a new trading account."""
        self.account_id = account_id
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions = {}  # Map from symbol to Position
        self.orders = []  # List of all orders
        self.open_orders = {}  # Map from order_id to Order for open orders
    
    def place_order(self, symbol: str, order_type: str, quantity: float, price: float) -> Order:
        """
        Place a new order.
        
        Args:
            symbol: Stock symbol
            order_type: "BUY" or "SELL"
            quantity: Number of shares
            price: Order price
            
        Returns:
            The created order
        """
        # Validate order
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if price <= 0:
            raise ValueError("Price must be positive")
        
        order_type = order_type.upper()
        if order_type not in ["BUY", "SELL"]:
            raise ValueError("Order type must be 'BUY' or 'SELL'")
        
        # Check if we have enough cash for buy orders
        if order_type == "BUY" and quantity * price > self.cash:
            raise ValueError(f"Insufficient cash. Need ${quantity * price:.2f}, have ${self.cash:.2f}")
        
        # Check if we have enough shares for sell orders
        if order_type == "SELL":
            position = self.positions.get(symbol)
            if not position or position.quantity < quantity:
                raise ValueError(f"Insufficient shares. Have {position.quantity if position else 0}, want to sell {quantity}")
        
        # Create the order
        order = Order(symbol, order_type, quantity, price)
        self.orders.append(order)
        self.open_orders[order.order_id] = order
        
        return order
    
    def fill_order(self, order_id: str, fill_price: Optional[float] = None) -> Dict:
        """
        Fill an open order.
        
        Args:
            order_id: ID of the order to fill
            fill_price: Price at which to fill the order (defaults to order price)
            
        Returns:
            Dictionary with fill details
        """
        if order_id not in self.open_orders:
            raise ValueError(f"Order {order_id} not found or already filled")
        
        order = self.open_orders[order_id]
        fill_price = fill_price or order.price
        
        # Update order status
        order.status = "FILLED"
        order.filled_price = fill_price
        order.filled_timestamp = datetime.datetime.now()
        
        # Update position
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        transaction = self.positions[symbol].update_from_fill(order, fill_price)
        
        # Update cash
        if order.order_type == "BUY":
            self.cash -= order.quantity * fill_price
        else:  # SELL
            self.cash += order.quantity * fill_price
        
        # Remove from open orders
        del self.open_orders[order_id]
        
        # Create fill details
        fill_details = {
            "order_id": order_id,
            "symbol": order.symbol,
            "type": order.order_type,
            "quantity": order.quantity,
            "price": fill_price,
            "timestamp": order.filled_timestamp.isoformat(),
            "transaction": transaction
        }
        
        return fill_details
    
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an open order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Dictionary with cancellation details
        """
        if order_id not in self.open_orders:
            raise ValueError(f"Order {order_id} not found or already filled/cancelled")
        
        order = self.open_orders[order_id]
        order.status = "CANCELLED"
        
        # Remove from open orders
        del self.open_orders[order_id]
        
        # Create cancellation details
        cancel_details = {
            "order_id": order_id,
            "symbol": order.symbol,
            "type": order.order_type,
            "quantity": order.quantity,
            "price": order.price,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return cancel_details
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get a position by symbol."""
        return self.positions.get(symbol)
    
    def get_account_value(self, price_function) -> Dict:
        """
        Calculate the total account value.
        
        Args:
            price_function: Function that takes a symbol and returns the current price
            
        Returns:
            Dictionary with account value details
        """
        positions_value = 0.0
        positions_data = []
        total_realized_pl = 0.0
        total_unrealized_pl = 0.0
        
        for symbol, position in self.positions.items():
            current_price = price_function(symbol)
            
            if current_price is not None and position.quantity > 0:
                market_value = position.market_value(current_price)
                positions_value += market_value
                realized_pl = position.realized_pl()
                unrealized_pl = position.unrealized_pl(current_price)
                
                total_realized_pl += realized_pl
                total_unrealized_pl += unrealized_pl
                
                positions_data.append({
                    "symbol": symbol,
                    "quantity": position.quantity,
                    "cost_basis": position.cost_basis,
                    "current_price": current_price,
                    "market_value": market_value,
                    "realized_pl": realized_pl,
                    "unrealized_pl": unrealized_pl,
                    "total_pl": realized_pl + unrealized_pl
                })
        
        # Calculate total account value
        total_value = self.cash + positions_value
        
        # Calculate profit/loss
        total_pl = total_value - self.initial_cash
        pl_percent = (total_pl / self.initial_cash) * 100 if self.initial_cash > 0 else 0
        
        return {
            "account_id": self.account_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "cash": self.cash,
            "positions_value": positions_value,
            "total_value": total_value,
            "initial_cash": self.initial_cash,
            "total_pl": total_pl,
            "pl_percent": pl_percent,
            "realized_pl": total_realized_pl,
            "unrealized_pl": total_unrealized_pl,
            "positions": positions_data,
            "open_orders": len(self.open_orders),
            "total_orders": len(self.orders)
        }
    
    def reset_account(self) -> Dict:
        """
        Reset the account to initial state.
        
        Returns:
            Dictionary with reset details
        """
        # Store final account state for return
        final_state = {
            "account_id": self.account_id,
            "final_cash": self.cash,
            "initial_cash": self.initial_cash,
            "profit_loss": self.cash - self.initial_cash,
            "positions": [pos.to_dict() for pos in self.positions.values()],
            "orders": [order.to_dict() for order in self.orders],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Reset account
        self.cash = self.initial_cash
        self.positions = {}
        self.orders = []
        self.open_orders = {}
        
        return final_state


class Broker:
    """Broker class for handling orders and account management."""
    
    def __init__(self, uri: str = "ws://localhost:8765", 
                 account_id: str = "default", 
                 initial_cash: float = 100000.0,
                 log_file: str = "broker_log.json"):
        """
        Initialize the broker.
        
        Args:
            uri: WebSocket server URI
            account_id: Account identifier
            initial_cash: Initial cash amount
            log_file: File to log trading activity
        """
        self.uri = uri
        self.account = Account(account_id, initial_cash)
        self.websocket = None
        self.connected = False
        self.available_symbols = []
        self.current_prices = {}  # Map from symbol to current price
        self.log_file = log_file
        
        # Set up logging
        self.logger = logging.getLogger('broker')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('broker.log')
        file_handler.setFormatter(log_formatter)
        self.logger.addHandler(file_handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)
    
    async def connect(self):
        """Connect to the WebSocket server."""
        self.websocket = await websockets.connect(
            self.uri,
            ping_interval=20,
            ping_timeout=30
        )
        self.connected = True
        self.logger.info(f"Connected to {self.uri}")
        
        # First message should be available symbols
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if data["type"] == "available_symbols":
            self.available_symbols = data["symbols"]
            self.logger.info(f"Available symbols: {', '.join(self.available_symbols)}")
        
        return self.available_symbols
    
    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            self.logger.info("Disconnected from server")
    
    async def subscribe(self, symbol: str):
        """Subscribe to a specific stock symbol."""
        if not self.connected:
            self.logger.error("Not connected to server")
            return False
        
        if symbol not in self.available_symbols:
            self.logger.error(f"Symbol {symbol} not available")
            return False
        
        # Send subscription request
        await self.websocket.send(json.dumps({
            "command": "subscribe",
            "symbol": symbol
        }))
        
        # Wait for response
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if data["type"] == "error":
            self.logger.error(f"Error subscribing to {symbol}: {data['message']}")
            return False
        
        if data["type"] == "initial_data":
            self.current_prices[symbol] = data["price"]
            self.logger.info(f"Subscribed to {symbol} at ${data['price']:.2f}")
            return True
        
        return False
    
    async def get_price(self, symbol: str) -> Optional[float]:
        """Get the current price for a stock."""
        if not self.connected:
            self.logger.error("Not connected to server")
            return None
        
        # If we already have a price, return it
        if symbol in self.current_prices:
            return self.current_prices[symbol]
        
        # Otherwise, get the price from the server
        await self.websocket.send(json.dumps({
            "command": "get_price",
            "symbol": symbol
        }))
        
        # Wait for response
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if data["type"] == "error":
            self.logger.error(f"Error getting price for {symbol}: {data['message']}")
            return None
        
        if data["type"] == "price_data":
            self.current_prices[symbol] = data["price"]
            return data["price"]
        
        return None
    
    def place_order(self, symbol: str, order_type: str, quantity: float, price: float) -> Optional[Order]:
        """Place a new order."""
        try:
            order = self.account.place_order(symbol, order_type, quantity, price)
            self.logger.info(f"Placed order: {order}")
            return order
        except ValueError as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def execute_order(self, order_id: str, fill_price: Optional[float] = None) -> Dict:
        """Execute an open order."""
        try:
            fill_details = self.account.fill_order(order_id, fill_price)
            self.logger.info(f"Executed order: {fill_details}")
            return fill_details
        except ValueError as e:
            self.logger.error(f"Error executing order: {e}")
            return {"error": str(e)}
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an open order."""
        try:
            cancel_details = self.account.cancel_order(order_id)
            self.logger.info(f"Cancelled order: {cancel_details}")
            return cancel_details
        except ValueError as e:
            self.logger.error(f"Error cancelling order: {e}")
            return {"error": str(e)}
    
    def get_account_value(self) -> Dict:
        """Get the current account value."""
        def price_func(symbol):
            return self.current_prices.get(symbol)
        
        account_value = self.account.get_account_value(price_func)
        return account_value
    
    def print_account_summary(self):
        """Print a summary of the account to the console."""
        account_value = self.get_account_value()
        
        print("\n===== ACCOUNT SUMMARY =====")
        print(f"Account: {account_value['account_id']}")
        print(f"Cash: ${account_value['cash']:.2f}")
        print(f"Portfolio Value: ${account_value['positions_value']:.2f}")
        print(f"Total Value: ${account_value['total_value']:.2f}")
        print(f"Initial Investment: ${account_value['initial_cash']:.2f}")
        print(f"Total P/L: ${account_value['total_pl']:.2f} ({account_value['pl_percent']:.2f}%)")
        print(f"Realized P/L: ${account_value['realized_pl']:.2f}")
        print(f"Unrealized P/L: ${account_value['unrealized_pl']:.2f}")
        
        if account_value['positions']:
            print("\n----- POSITIONS -----")
            for pos in account_value['positions']:
                print(f"{pos['symbol']}: {pos['quantity']} shares @ ${pos['cost_basis']:.2f} (Current: ${pos['current_price']:.2f})")
                print(f"  Market Value: ${pos['market_value']:.2f}")
                print(f"  Unrealized P/L: ${pos['unrealized_pl']:.2f}")
                print(f"  Realized P/L: ${pos['realized_pl']:.2f}")
                print(f"  Total P/L: ${pos['total_pl']:.2f}")
                print()
        
        print(f"Open Orders: {account_value['open_orders']}")
        print(f"Total Orders: {account_value['total_orders']}")
        print("============================\n")
    
    def reset_account(self) -> None:
        """Reset the account and log the final state."""
        final_state = self.account.reset_account()
        
        # Log to file
        log_path = Path(self.log_file)
        
        # If file exists, read it first to append
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                log_data = []
        else:
            log_data = []
        
        # Append new entry
        log_data.append(final_state)
        
        # Write back to file
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"Account reset. Final state logged to {self.log_file}")
        
        # Print final summary
        print("\n===== FINAL ACCOUNT SUMMARY =====")
        print(f"Account: {final_state['account_id']}")
        print(f"Final Cash: ${final_state['final_cash']:.2f}")
        print(f"Initial Cash: ${final_state['initial_cash']:.2f}")
        print(f"Profit/Loss: ${final_state['profit_loss']:.2f}")
        print(f"Positions: {len(final_state['positions'])}")
        print(f"Orders: {len(final_state['orders'])}")
        print("================================\n")
    
    async def process_price_update(self, data: Dict) -> None:
        """Process a price update message."""
        if data["type"] == "price_update":
            symbol = data["symbol"]
            price = data["price"]
            self.current_prices[symbol] = price
    
    async def listen_for_updates(self, handler_function=None):
        """Listen for real-time price updates and pass them to a handler function."""
        if not self.connected:
            self.logger.error("Not connected to server")
            return
        
        self.logger.info("Listening for price updates...")
        
        try:
            while True:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=30)
                data = json.loads(response)
                
                # Update our price cache
                await self.process_price_update(data)
                
                # Call the handler function if provided
                if handler_function and callable(handler_function):
                    await handler_function(data)
        
        except asyncio.TimeoutError:
            self.logger.warning("No message received in 30 seconds, sending ping")
            await self.websocket.ping()
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.error("Connection closed unexpectedly")
            self.connected = False
        
        except Exception as e:
            self.logger.error(f"Error in update listener: {e}")


# Example usage in main
async def main():
    broker = Broker()
    
    try:
        # Connect to server
        symbols = await broker.connect()
        
        if not symbols:
            print("No symbols available")
            return
        
        # Subscribe to a symbol
        symbol = "AAPL"
        if symbol in symbols:
            await broker.subscribe(symbol)
        
        # Place a buy order
        order = broker.place_order(symbol, "BUY", 10, 175.0)
        
        if order:
            # Execute the order
            broker.execute_order(order.order_id)
            
            # Get the account value
            account_value = broker.get_account_value()
            print(f"Account value: ${account_value['total_value']:.2f}")
            
            # Print a summary
            broker.print_account_summary()
            
            # Reset the account
            broker.reset_account()
        
        # Listen for updates
        await broker.listen_for_updates()
    
    finally:
        # Disconnect from server
        await broker.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBroker stopped by user")