function setCmd(value) {
    document.getElementById('cmd').value = value;
}

async function run() {
    const cmd = document.getElementById('cmd').value;
    const output = document.getElementById('output');

    output.value = 'Running...\n';

    const res = await fetch('/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ cmd })
    });

    const data = await res.json();

    output.value =
        '$ ' + cmd + '\n\n' +
        data.stdout +
        (data.stderr ? '\n--- stderr ---\n' + data.stderr : '');
}
