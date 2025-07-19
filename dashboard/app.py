import subprocess
import sys
from typing import Optional

from flask import Flask, Response, jsonify, render_template
from werkzeug.wrappers import Response as WerkzeugResponse

app = Flask(__name__)

bot_process: Optional[subprocess.Popen] = None


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start_bot() -> WerkzeugResponse:
    global bot_process
    if bot_process is None or bot_process.poll() is not None:
        try:
            # Execute the main bot script
            bot_process = subprocess.Popen(
                [sys.executable, "-m", "trading_bot.main"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd="..",  # Run from the project root directory
            )
            return jsonify({"status": "success", "message": "Bot started"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    return jsonify({"status": "error", "message": "Bot is already running"})


@app.route("/stop", methods=["POST"])
def stop_bot() -> WerkzeugResponse:
    global bot_process
    if bot_process and bot_process.poll() is None:
        try:
            bot_process.terminate()
            bot_process.wait(timeout=5)
            bot_process = None
            return jsonify({"status": "success", "message": "Bot stopped"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    return jsonify({"status": "error", "message": "Bot is not running"})


@app.route("/status")
def status() -> WerkzeugResponse:
    global bot_process
    if bot_process and bot_process.poll() is None:
        return jsonify({"status": "running"})
    return jsonify({"status": "stopped"})


def stream_logs():
    """Stream the bot's logs."""
    if bot_process and bot_process.stdout:
        # Non-blocking read from stdout
        for line in iter(bot_process.stdout.readline, ""):
            yield line
    else:
        yield ""


@app.route("/logs")
def logs() -> WerkzeugResponse:
    """Stream the bot's logs."""
    return Response(stream_logs(), mimetype="text/plain")


@app.route("/analytics")
def analytics():
    """Serve the analytics dashboard."""
    return render_template("analytics.html")


@app.route("/api/analytics")
def api_analytics():
    """Provide analytics data from a backtest."""
    # This is a placeholder for running a backtest and generating analytics
    # In a real application, you would run a backtest and return the results
    return jsonify(
        {
            "summary": {
                "total_return": "10.5%",
                "sharpe_ratio": "1.5",
                "max_drawdown": "-5.2%",
            },
            "portfolio": {"total_trades": 123},
            "equity_curve": [
                ["2023-01-01", 100000],
                ["2023-01-02", 101000],
                ["2023-01-03", 100500],
                ["2023-01-04", 102000],
                ["2023-01-05", 103000],
            ],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
