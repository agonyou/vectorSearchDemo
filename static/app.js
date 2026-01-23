function setLoadCmd(value) {
    document.getElementById('loadCmd').value = value;
}

function setRagCmd(value) {
    document.getElementById('ragCmd').value = value;
}

async function runLoad() {
    return runCommand('loadCmd', 'outputLoad');
}

async function runRag() {
    return runCommand('ragCmd', 'outputQuery');
}

async function runCommand(cmdInputId, outputId) {
    const cmd = document.getElementById(cmdInputId).value;
    const output = document.getElementById(outputId);

    output.value = '';
    output.scrollTop = 0;

    const res = await fetch('/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cmd })
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        output.value += chunk;
        output.scrollTop = output.scrollHeight;
    }
}

function setRagCmdComposite() {
    const sender = document.getElementById('sender').value.trim();
    const receiver = document.getElementById('receiver').value.trim();
    const prompt = document.getElementById('prompt').value.trim();

    const bucket = 'vectorSearchDemo';
    const scope = '_default';
    const collection = 'emails';

    let cmd = `python ragComposite.py ` +
              `--bucket ${bucket} ` +
              `--scope ${scope} ` +
              `--collection ${collection}`;

    if (sender) {
        cmd += ` --sender "${sender}"`;
    }

    if (receiver) {
        cmd += ` --receiver "${receiver}"`;
    }

    if (prompt) {
        cmd += ` --prompt "${prompt}"`;
    }

    setRagCmd(cmd);
}
