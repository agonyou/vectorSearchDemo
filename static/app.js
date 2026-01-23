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
