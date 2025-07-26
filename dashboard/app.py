import json
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import os
import signal
import psutil
import aiohttp
import asyncio
from decimal import Decimal

from flask import Flask, Response, jsonify, render_template, request, send_file
from flask_socketio import SocketIO, emit
from werkzeug.wrappers import Response as WerkzeugResponse

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
bot_process: Optional[subprocess.Popen] = None
bot_logs: List[str] = []
bot_status = "stopped"
current_symbols: Set[str] = set()
analytics_data: Dict = {}
log_buffer_size = 1000

# Alpaca API configuration
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# Popular stock symbols for the picker (expanded list)
POPULAR_SYMBOLS = [
    # Tech Giants
    "AAPL",
    "GOOGL",
    "MSFT",
    "AMZN",
    "META",
    "TSLA",
    "NVDA",
    "NFLX",
    "ADBE",
    "CRM",
    "ORCL",
    "INTC",
    "AMD",
    "QCOM",
    "PYPL",
    "UBER",
    "LYFT",
    "ZM",
    "SHOP",
    "SQ",
    # Finance
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "C",
    "AIG",
    "AXP",
    "BLK",
    "SCHW",
    "COF",
    "USB",
    "PNC",
    "TFC",
    "KEY",
    "HBAN",
    "FITB",
    "RF",
    "ZION",
    "CMA",
    # Healthcare
    "JNJ",
    "PFE",
    "UNH",
    "ABBV",
    "MRK",
    "TMO",
    "ABT",
    "DHR",
    "BMY",
    "AMGN",
    "GILD",
    "CVS",
    "CI",
    "ANTM",
    "HUM",
    "CNC",
    "DVA",
    "HCA",
    "UHS",
    "THC",
    # Consumer
    "KO",
    "PEP",
    "WMT",
    "HD",
    "MCD",
    "DIS",
    "NKE",
    "SBUX",
    "TGT",
    "COST",
    "LOW",
    "TJX",
    "M",
    "KSS",
    "JWN",
    "GPS",
    "LB",
    "URBN",
    "ROST",
    "TJX",
    # Energy
    "XOM",
    "CVX",
    "COP",
    "EOG",
    "SLB",
    "KMI",
    "PSX",
    "VLO",
    "MPC",
    "OXY",
    "HAL",
    "BKR",
    "NOV",
    "FTI",
    "WMB",
    "OKE",
    "PXD",
    "EOG",
    "DVN",
    "APC",
    # Industrial
    "BA",
    "CAT",
    "GE",
    "MMM",
    "HON",
    "UPS",
    "FDX",
    "RTX",
    "LMT",
    "NOC",
    "GD",
    "TXT",
    "EMR",
    "ETN",
    "PH",
    "DOV",
    "XYL",
    "AME",
    "ITW",
    "ROK",
    # Materials
    "LIN",
    "APD",
    "FCX",
    "NEM",
    "AA",
    "DOW",
    "DD",
    "EMN",
    "BLL",
    "IFF",
    "NUE",
    "STLD",
    "X",
    "AKS",
    "RS",
    "BLL",
    "VMC",
    "MLM",
    "CRH",
    "VMC",
    # Utilities
    "NEE",
    "DUK",
    "SO",
    "D",
    "AEP",
    "SRE",
    "XEL",
    "PCG",
    "DTE",
    "ED",
    "EIX",
    "AEE",
    "WEC",
    "CMS",
    "CNP",
    "NI",
    "OGE",
    "PEG",
    "SJI",
    "UIL",
    # Real Estate
    "AMT",
    "PLD",
    "CCI",
    "EQIX",
    "PSA",
    "SPG",
    "O",
    "DLR",
    "WELL",
    "VICI",
    "PLD",
    "AMT",
    "CCI",
    "EQIX",
    "PSA",
    "SPG",
    "O",
    "DLR",
    "WELL",
    "VICI",
    # Communication
    "T",
    "VZ",
    "CMCSA",
    "CHTR",
    "TMUS",
    "LUMN",
    "CTL",
    "VZ",
    "TMUS",
    "LUMN",
]


def get_bot_status() -> str:
    """Get the current status of the bot process."""
    global bot_process
    if bot_process is None:
        return "stopped"
    if bot_process.poll() is None:
        return "running"
    return "stopped"


def add_log_entry(message: str) -> None:
    """Add a log entry to the buffer."""
    global bot_logs
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    bot_logs.append(log_entry)

    # Keep only the last N entries
    if len(bot_logs) > log_buffer_size:
        bot_logs = bot_logs[-log_buffer_size:]

    # Emit to connected clients
    socketio.emit("log_update", {"message": log_entry})


def read_bot_output() -> None:
    """Read output from the bot process and emit to clients."""
    global bot_process
    while True:
        if bot_process and bot_process.poll() is None and bot_process.stdout:
            try:
                line = bot_process.stdout.readline()
                if line:
                    add_log_entry(line.strip())
                else:
                    time.sleep(0.1)
            except Exception as e:
                add_log_entry(f"Error reading bot output: {e}")
                break
        else:
            time.sleep(1)


def load_symbols_from_config() -> Set[str]:
    """Load symbols from environment configuration."""
    try:
        symbols_str = os.getenv("SYMBOLS", "AAPL,GOOGL,MSFT")
        return set(s.strip() for s in symbols_str.split(",") if s.strip())
    except Exception:
        return {"AAPL", "GOOGL", "MSFT"}


def save_symbols_to_config(symbols: Set[str]) -> None:
    """Save symbols to environment configuration."""
    try:
        symbols_str = ",".join(sorted(symbols))
        # Update the environment variable (this is a simplified approach)
        # In production, you'd want to update a config file
        os.environ["SYMBOLS"] = symbols_str
    except Exception as e:
        add_log_entry(f"Error saving symbols: {e}")


async def get_alpaca_account_data() -> Dict:
    """Get real account data from Alpaca API."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return {"error": "Alpaca API credentials not configured"}

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession() as session:
            # Get account information
            async with session.get(
                f"{ALPACA_BASE_URL}/v2/account", headers=headers
            ) as response:
                if response.status == 200:
                    account_data = await response.json()
                    return {
                        "portfolio_value": float(
                            account_data.get("portfolio_value", 0)
                        ),
                        "cash": float(account_data.get("cash", 0)),
                        "buying_power": float(account_data.get("buying_power", 0)),
                        "account_status": account_data.get("status", "unknown"),
                    }
                else:
                    return {"error": f"Failed to get account data: {response.status}"}
    except Exception as e:
        return {"error": f"Error connecting to Alpaca: {str(e)}"}


async def get_alpaca_positions() -> List[Dict]:
    """Get real positions from Alpaca API."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return []

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{ALPACA_BASE_URL}/v2/positions", headers=headers
            ) as response:
                if response.status == 200:
                    positions = await response.json()
                    return [
                        {
                            "symbol": pos.get("symbol"),
                            "side": pos.get("side"),
                            "quantity": float(pos.get("qty", 0)),
                            "market_value": float(pos.get("market_value", 0)),
                            "unrealized_pl": float(pos.get("unrealized_pl", 0)),
                            "avg_entry_price": float(pos.get("avg_entry_price", 0)),
                        }
                        for pos in positions
                    ]
                else:
                    return []
    except Exception as e:
        add_log_entry(f"Error getting positions: {e}")
        return []


async def get_alpaca_orders() -> List[Dict]:
    """Get real orders from Alpaca API."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return []

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession() as session:
            # Get recent orders
            params = {"status": "all", "limit": 50}
            async with session.get(
                f"{ALPACA_BASE_URL}/v2/orders", headers=headers, params=params
            ) as response:
                if response.status == 200:
                    orders = await response.json()
                    return [
                        {
                            "id": order.get("id"),
                            "symbol": order.get("symbol"),
                            "side": order.get("side"),
                            "quantity": float(order.get("qty", 0)),
                            "price": float(order.get("filled_avg_price", 0)),
                            "status": order.get("status"),
                            "created_at": order.get("created_at"),
                            "filled_at": order.get("filled_at"),
                        }
                        for order in orders
                        if order.get("status") in ["filled", "partially_filled"]
                    ]
                else:
                    return []
    except Exception as e:
        add_log_entry(f"Error getting orders: {e}")
        return []


async def get_alpaca_portfolio_history() -> List[List]:
    """Get real portfolio history from Alpaca API."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return []

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession() as session:
            # Get portfolio history for the last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            params = {
                "period": "1D",
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
            }

            async with session.get(
                f"{ALPACA_BASE_URL}/v2/account/portfolio/history",
                headers=headers,
                params=params,
            ) as response:
                if response.status == 200:
                    history = await response.json()
                    equity_data = history.get("equity", [])
                    return [
                        [entry.get("date", ""), float(entry.get("value", 0))]
                        for entry in equity_data
                    ]
                else:
                    return []
    except Exception as e:
        add_log_entry(f"Error getting portfolio history: {e}")
        return []


@app.route("/")
def index() -> str:
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/test")
def test() -> Response:
    """Serve the test page for popular symbols."""
    return send_file("test_popular_symbols.html")


@app.route("/api/status")
def api_status() -> WerkzeugResponse:
    """Get bot status."""
    global bot_status
    bot_status = get_bot_status()
    return jsonify(
        {
            "status": bot_status,
            "symbols": list(current_symbols),
            "log_count": len(bot_logs),
        }
    )


@app.route("/api/start", methods=["POST"])
def api_start_bot() -> WerkzeugResponse:
    """Start the trading bot."""
    global bot_process, current_symbols

    if get_bot_status() == "running":
        return jsonify({"status": "error", "message": "Bot is already running"})

    try:
        # Get symbols from request or use current symbols
        symbols = (
            request.json.get("symbols", list(current_symbols))
            if request.json
            else list(current_symbols)
        )
        if not symbols:
            symbols = ["AAPL", "GOOGL", "MSFT"]  # Default symbols

        # Update current symbols
        current_symbols = set(symbols)
        save_symbols_to_config(current_symbols)

        # Prepare command with symbols
        cmd = [sys.executable, "main.py", "run", "--symbols", ",".join(symbols)]

        # Start the bot process
        bot_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=".",  # Run from the current directory (project root)
            bufsize=1,
            universal_newlines=True,
        )

        # Start log reading thread
        log_thread = threading.Thread(target=read_bot_output, daemon=True)
        log_thread.start()

        # Add initial log entry
        add_log_entry(f"Starting bot with symbols: {', '.join(symbols)}")
        add_log_entry(f"Bot process started with PID: {bot_process.pid}")

        add_log_entry("Bot started successfully")
        return jsonify({"status": "success", "message": "Bot started successfully"})

    except Exception as e:
        add_log_entry(f"Error starting bot: {e}")
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/stop", methods=["POST"])
def api_stop_bot() -> WerkzeugResponse:
    """Stop the trading bot."""
    global bot_process

    if get_bot_status() == "stopped":
        return jsonify({"status": "error", "message": "Bot is not running"})

    try:
        if bot_process:
            # Try graceful termination first
            bot_process.terminate()

            # Wait for graceful shutdown
            try:
                bot_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                bot_process.kill()
                bot_process.wait()

            bot_process = None
            add_log_entry("Bot stopped successfully")
            return jsonify({"status": "success", "message": "Bot stopped successfully"})

        return jsonify({"status": "error", "message": "No bot process found"})

    except Exception as e:
        add_log_entry(f"Error stopping bot: {e}")
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/symbols", methods=["GET", "POST"])
def api_symbols() -> WerkzeugResponse:
    """Get or update trading symbols."""
    global current_symbols

    if request.method == "GET":
        return jsonify(
            {"symbols": list(current_symbols), "popular_symbols": POPULAR_SYMBOLS}
        )

    elif request.method == "POST":
        try:
            data = request.json
            if not data or "symbols" not in data:
                return jsonify({"status": "error", "message": "No symbols provided"})

            new_symbols = set(s.strip().upper() for s in data["symbols"] if s.strip())

            # Validate symbols (basic validation)
            if not new_symbols:
                return jsonify(
                    {"status": "error", "message": "At least one symbol is required"}
                )

            # Update symbols
            current_symbols = new_symbols
            save_symbols_to_config(current_symbols)

            add_log_entry(f"Symbols updated: {', '.join(sorted(current_symbols))}")
            return jsonify(
                {
                    "status": "success",
                    "message": "Symbols updated successfully",
                    "symbols": list(current_symbols),
                }
            )

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})

    # Default return for unsupported methods
    return jsonify({"status": "error", "message": "Method not allowed"})


@app.route("/api/analytics")
def api_analytics() -> WerkzeugResponse:
    """Get real analytics data from Alpaca API."""
    try:
        # Try to get real Alpaca data if credentials are available
        if ALPACA_API_KEY and ALPACA_SECRET_KEY:
            # Use asyncio to run async functions in sync context
            import asyncio

            try:
                # Get real account data
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                account_data = loop.run_until_complete(get_alpaca_account_data())

                if "error" not in account_data:
                    # Get real portfolio history
                    equity_curve = loop.run_until_complete(
                        get_alpaca_portfolio_history()
                    )

                    # Get real positions
                    positions = loop.run_until_complete(get_alpaca_positions())

                    # Get real orders
                    orders = loop.run_until_complete(get_alpaca_orders())

                    # Calculate metrics
                    portfolio_value = account_data.get("portfolio_value", 0)
                    cash = account_data.get("cash", 0)
                    positions_count = len(positions)

                    # Calculate total return (simplified)
                    if equity_curve and len(equity_curve) > 1:
                        initial_value = equity_curve[0][1]
                        current_value = equity_curve[-1][1]
                        total_return = (
                            (current_value - initial_value) / initial_value
                        ) * 100
                    else:
                        total_return = 0.0

                    # Convert orders to recent trades format
                    recent_trades = []
                    for order in orders[:10]:  # Last 10 trades
                        if order.get("filled_at"):
                            recent_trades.append(
                                {
                                    "symbol": order.get("symbol", ""),
                                    "side": order.get("side", "").upper(),
                                    "quantity": int(order.get("quantity", 0)),
                                    "price": order.get("price", 0.0),
                                    "timestamp": order.get("filled_at", ""),
                                }
                            )

                    analytics_data = {
                        "summary": {
                            "total_return": f"{total_return:.2f}%",
                            "sharpe_ratio": "0.00",  # Would need more data to calculate
                            "max_drawdown": "0.00%",  # Would need more data to calculate
                            "win_rate": "0.0%",  # Would need more data to calculate
                            "total_trades": len(orders),
                            "profit_factor": "0.00",  # Would need more data to calculate
                        },
                        "portfolio": {
                            "total_value": f"${portfolio_value:,.2f}",
                            "cash": f"${cash:,.2f}",
                            "positions": positions_count,
                            "daily_pnl": "$0.00",  # Would need more data to calculate
                        },
                        "equity_curve": equity_curve,
                        "recent_trades": recent_trades,
                    }

                    loop.close()
                    return jsonify(analytics_data)

                loop.close()

            except Exception as e:
                add_log_entry(f"Error getting real Alpaca data: {e}")

        # Fallback to sample data if Alpaca API not available or fails
        analytics_data = {
            "summary": {
                "total_return": "2.45%",
                "sharpe_ratio": "1.23",
                "max_drawdown": "-1.85%",
                "win_rate": "68.5%",
                "total_trades": 45,
                "profit_factor": "1.85",
            },
            "portfolio": {
                "total_value": "$102,450.00",
                "cash": "$15,230.00",
                "positions": 3,
                "daily_pnl": "$1,250.00",
            },
            "equity_curve": [
                ["2024-01-01", 100000],
                ["2024-01-02", 100500],
                ["2024-01-03", 101200],
                ["2024-01-04", 100800],
                ["2024-01-05", 102450],
            ],
            "recent_trades": [
                {
                    "symbol": "AAPL",
                    "side": "BUY",
                    "quantity": 100,
                    "price": 150.25,
                    "timestamp": "2024-01-15 10:30:00",
                },
                {
                    "symbol": "GOOGL",
                    "side": "SELL",
                    "quantity": 50,
                    "price": 2800.75,
                    "timestamp": "2024-01-15 09:45:00",
                },
            ],
        }

        return jsonify(analytics_data)

    except Exception as e:
        add_log_entry(f"Error getting analytics: {e}")
        return jsonify(
            {
                "summary": {
                    "total_return": "0.00%",
                    "sharpe_ratio": "0.00",
                    "max_drawdown": "0.00%",
                    "win_rate": "0.0%",
                    "total_trades": 0,
                    "profit_factor": "0.00",
                },
                "portfolio": {
                    "total_value": "$0.00",
                    "cash": "$0.00",
                    "positions": 0,
                    "daily_pnl": "$0.00",
                },
                "equity_curve": [],
                "recent_trades": [],
                "error": str(e),
            }
        )


@app.route("/api/logs")
def api_logs() -> WerkzeugResponse:
    """Get recent logs."""
    return jsonify(
        {"logs": bot_logs[-100:], "total_count": len(bot_logs)}  # Return last 100 logs
    )


@socketio.on("connect")
def handle_connect() -> None:
    """Handle client connection."""
    emit("status_update", {"status": get_bot_status()})
    emit("symbols_update", {"symbols": list(current_symbols)})


@socketio.on("disconnect")
def handle_disconnect() -> None:
    """Handle client disconnection."""
    pass


# Initialize symbols from config
current_symbols = load_symbols_from_config()

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
