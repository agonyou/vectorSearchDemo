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

function setRagCmdHyperscale() {
    const prompt = document.getElementById('hyperPrompt').value.trim();

    const bucket = 'vectorSearchDemo';
    const scope = '_default';
    const collection = 'movies';

    let cmd = `python ragHyperscale.py ` +
              `--bucket ${bucket} ` +
              `--scope ${scope} ` +
              `--collection ${collection}`;

    cmd += ` --prompt "${prompt}"`;

    setRagCmd(cmd);
}

// Hybrid load: provider picks the collection so the vectors match the collection's index dimension.
// Only the yelp dataset has both a 384 (yelp) and a 1536 (yelp_openai) collection in this cluster.
function setLoadCmdHybrid() {
    const provider = document.getElementById('hybridLoadProvider').value;
    const collection = provider === 'openai' ? 'yelp_openai' : 'yelp';

    const cmd = `python load.py ` +
                `--data data/yelp_academic_dataset_business.json ` +
                `--text-fields categories ` +
                `--bucket vectorSearchDemo --scope _default --collection ${collection} ` +
                `--copy-fields latitude longitude name ` +
                `--limit 5 --id-field business_id ` +
                `--embedding-provider ${provider}`;

    setLoadCmd(cmd);
}

function setRagCmdHybrid() {
    const prompt = document.getElementById('hybridPrompt').value.trim();

    const bucket = 'vectorSearchDemo';
    const scope = '_default';

    // Provider selector swaps collection + FTS index together so the 384/1536 dimensions agree.
    const provider = document.getElementById('hybridProvider').value;
    const collection = provider === 'openai' ? 'yelp_openai' : 'yelp';
    const indexName = provider === 'openai' ? 'ix-yelp-openai-vector' : 'ix-yelp-business-vector';

    const latitude = document.getElementById('hybridLatitude').value.trim();
    const longitude = document.getElementById('hybridLongitude').value.trim();
    const radius = document.getElementById('hybridRadius').value.trim();

    let cmd = `python ragHybrid.py ` +
              `--bucket ${bucket} ` +
              `--scope ${scope} ` +
              `--collection ${collection} ` +
              `--index-name ${indexName} ` +
              `--embedding-provider ${provider}`;

    if (prompt) {
        cmd += ` --prompt "${prompt}"`;
    }

    if (latitude && longitude && radius) {
        cmd += ` --latitude ${latitude} --longitude ${longitude} --radius ${radius}`;
    }

    setRagCmd(cmd);
}