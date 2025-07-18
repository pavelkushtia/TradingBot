import subprocess
import sys
from typing import Optional

from flask import Flask, Response, jsonify, render_template, request
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


@app.route("/logs")
def logs() -> WerkzeugResponse:
    """Stream bot logs."""

    def generate():
        if bot_process and bot_process.stdout:
            # Non-blocking read from stdout
            for line in iter(bot_process.stdout.readline, ""):
                yield line
        else:
            yield ""

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
