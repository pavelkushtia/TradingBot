import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import psutil
from flask import Flask, Response, jsonify, render_template, request
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

# Popular stock symbols for the picker
POPULAR_SYMBOLS = [
    # Tech
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
    # Communication
    "T",
    "VZ",
    "CMCSA",
    "CHTR",
    "TMUS",
    "LUMN",
    "CTL",
    "VZ",
    "T",
    "TMUS",
]


def get_bot_status() -> str:
    """Get the current status of the bot process."""
    global bot_process
    if bot_process is None:
        return "stopped"
    if bot_process.poll() is None:
        return "running"
    return "stopped"


def add_log_entry(message: str):
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


def read_bot_output():
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


def save_symbols_to_config(symbols: Set[str]):
    """Save symbols to environment configuration."""
    try:
        symbols_str = ",".join(sorted(symbols))
        # Update the environment variable (this is a simplified approach)
        # In production, you'd want to update a config file
        os.environ["SYMBOLS"] = symbols_str
    except Exception as e:
        add_log_entry(f"Error saving symbols: {e}")


@app.route("/")
def index() -> str:
    """Main dashboard page."""
    return render_template("index.html")


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
            cwd="..",  # Run from the project root directory
            bufsize=1,
            universal_newlines=True,
        )

        # Start log reading thread
        log_thread = threading.Thread(target=read_bot_output, daemon=True)
        log_thread.start()

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
    """Get analytics data."""
    # Generate sample analytics data
    # In a real implementation, this would come from your trading bot
    import random

    # Generate sample equity curve
    base_value = 100000
    equity_curve = []
    current_value = base_value

    for i in range(30):
        date = (datetime.now() - timedelta(days=29 - i)).strftime("%Y-%m-%d")
        # Add some realistic volatility
        change = random.uniform(-0.02, 0.03) * current_value
        current_value += change
        equity_curve.append([date, round(current_value, 2)])

    # Calculate metrics
    total_return = ((current_value - base_value) / base_value) * 100
    sharpe_ratio = random.uniform(0.8, 2.5)
    max_drawdown = random.uniform(-0.05, -0.02) * 100

    analytics_data = {
        "summary": {
            "total_return": f"{total_return:.2f}%",
            "sharpe_ratio": f"{sharpe_ratio:.2f}",
            "max_drawdown": f"{max_drawdown:.2f}%",
            "win_rate": f"{random.uniform(45, 75):.1f}%",
            "total_trades": random.randint(50, 200),
            "profit_factor": f"{random.uniform(1.1, 2.5):.2f}",
        },
        "portfolio": {
            "total_value": f"${current_value:,.2f}",
            "cash": f"${random.uniform(10000, 50000):,.2f}",
            "positions": random.randint(3, 8),
            "daily_pnl": f"${random.uniform(-5000, 10000):,.2f}",
        },
        "equity_curve": equity_curve,
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


@app.route("/api/logs")
def api_logs() -> WerkzeugResponse:
    """Get recent logs."""
    return jsonify(
        {"logs": bot_logs[-100:], "total_count": len(bot_logs)}  # Return last 100 logs
    )


@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    emit("status_update", {"status": get_bot_status()})
    emit("symbols_update", {"symbols": list(current_symbols)})


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    pass


# Initialize symbols from config
current_symbols = load_symbols_from_config()

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
