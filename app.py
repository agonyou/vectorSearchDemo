from flask import Flask, render_template, request, jsonify
import subprocess
import shlex
from pathlib import Path

app = Flask(__name__)
BASE_DIR = Path(__file__).parent

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run_cmd():
    cmd = request.json.get("cmd", "")

    if not cmd:
        return jsonify(stdout="", stderr="No command provided")

    try:
        args = shlex.split(cmd)

        result = subprocess.run(
            args,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )

        return jsonify(
            stdout=result.stdout,
            stderr=result.stderr
        )

    except Exception as e:
        return jsonify(stdout="", stderr=str(e))

if __name__ == "__main__":
    app.run(debug=True)
