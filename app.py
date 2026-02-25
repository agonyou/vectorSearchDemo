import sys
import platform
import subprocess
import shlex
from pathlib import Path
from flask import Flask, render_template, request, Response

app = Flask(__name__)
BASE_DIR = Path(__file__).parent
IS_WINDOWS = platform.system() == "Windows"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run_cmd():
    cmd = request.json.get("cmd", "")
    if not cmd:
        return Response("No command provided\n", mimetype="text/plain")

    def generate():
        process = None
        try:
            cmd_args = shlex.split(cmd)

            # 🔑 CRITICAL FIX:
            # If user typed "python ...", force the venv interpreter
            if cmd_args and cmd_args[0].lower() == "python":
                cmd_args[0] = sys.executable

            # Linux-only: force line buffering
            if not IS_WINDOWS:
                cmd_args = ["stdbuf", "-oL"] + cmd_args

            yield f"$ {cmd}\n\n"

            process = subprocess.Popen(
                cmd_args,
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in process.stdout:
                yield line

            process.wait()
            yield "\n=== finished ===\n"

        except GeneratorExit:
            # client disconnected
            if process:
                process.kill()
            raise
        except Exception as e:
            yield f"\nERROR: {e}\n"

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=8080)