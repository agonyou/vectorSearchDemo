from flask import Flask, request, jsonify
import subprocess
import shlex
import sys
from pathlib import Path

app = Flask(__name__)
BASE_DIR = Path(__file__).parent

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Command Runner</title>
<style>
body {
    font-family: system-ui, sans-serif;
    background: #f6f7f9;
    padding: 20px;
}

.container {
    max-width: 900px;
    margin: auto;
}

h1 {
    margin-bottom: 10px;
}

input[type=text] {
    width: 100%;
    padding: 10px;
    font-family: monospace;
    font-size: 14px;
    margin-bottom: 10px;
}

textarea {
    width: 100%;
    height: 300px;
    padding: 10px;
    font-family: monospace;
    font-size: 13px;
    background: #111;
    color: #0f0;
    resize: vertical;
}

button {
    padding: 8px 12px;
    margin-right: 6px;
    margin-bottom: 10px;
    cursor: pointer;
}

.presets button {
    background: #e4e6eb;
    border: 1px solid #ccc;
}

.run {
    background: #2563eb;
    color: white;
    border: none;
}
</style>
</head>
<body>
<div class="container">
    <h1>Command Runner</h1>

    <div class="presets">
        <button onclick="setCmd('python load.py --foo --bar --baz')">Load</button>
        <button onclick="setCmd('python rag.py')">RAG</button>
        <button onclick="setCmd('python test.py')">Test</button>
    </div>

    <input id="cmd" type="text" placeholder="Enter command..." />

    <button class="run" onclick="run()">Run</button>

    <textarea id="output" readonly></textarea>
</div>

<script>
function setCmd(value) {
    document.getElementById('cmd').value = value;
}

async function run() {
    const cmd = document.getElementById('cmd').value;
    const output = document.getElementById('output');

    output.value = 'Running...\\n';

    const res = await fetch('/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cmd})
    });

    const data = await res.json();

    output.value =
        '$ ' + cmd + '\\n\\n' +
        data.stdout +
        (data.stderr ? '\\n--- stderr ---\\n' + data.stderr : '');
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return HTML

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
    app.run(host="0.0.0.0", port=5000, debug=True)
