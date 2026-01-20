function setCmd(value) {
    document.getElementById('cmd').value = value;
}

async function run() {
    const cmd = document.getElementById('cmd').value;
    const output = document.getElementById('output');

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
