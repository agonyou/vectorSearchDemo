from flask import Flask, render_template, request, Response
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
        return Response("No command provided\n", mimetype="text/plain")

    def generate():
        try:
            args = ["stdbuf", "-oL"] + shlex.split(cmd)

            process = subprocess.Popen(
                args,
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            yield f"$ {cmd}\n\n"

            for line in process.stdout:
                yield line

            process.wait()
            yield "\n=== finished ===\n"

        except GeneratorExit:
            # client disconnected
            process.kill()
            raise
        except Exception as e:
            yield f"\nERROR: {e}\n"

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=8080)
