function setLoadCmd(value) {
    document.getElementById('loadCmd').value = value;
}

function setRagCmd(value) {
    document.getElementById('ragCmd').value = value;
}

async function runLoad() {
    return runCommand('loadCmd', 'outputLoad', 'preflightLoad');
}

async function runRag() {
    return runCommand('ragCmd', 'outputQuery', 'preflightRag');
}

// Watch the streamed output for the scripts' preflight markers (#2) and reflect them
// on a status badge so the pass/fail is visible without scrolling the output.
function updatePreflightBadge(badge, text) {
    if (!badge || badge.dataset.settled === '1') return;
    if (text.includes('Preflight OK')) {
        badge.textContent = '✓ Preflight passed';
        badge.className = 'preflight-badge pass';
        badge.dataset.settled = '1';
    } else if (text.includes('Embedding dimension mismatch')) {
        badge.textContent = '✗ Dimension mismatch';
        badge.className = 'preflight-badge fail';
        badge.dataset.settled = '1';
    } else if (text.includes('Preflight skipped')) {
        badge.textContent = 'Preflight skipped';
        badge.className = 'preflight-badge skip';
        badge.dataset.settled = '1';
    }
}

async function runCommand(cmdInputId, outputId, badgeId) {
    const cmd = document.getElementById(cmdInputId).value;
    const output = document.getElementById(outputId);
    const badge = badgeId ? document.getElementById(badgeId) : null;

    output.value = '';
    output.scrollTop = 0;
    if (badge) {
        badge.textContent = '';
        badge.className = 'preflight-badge';
        badge.dataset.settled = '0';
    }

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
        updatePreflightBadge(badge, output.value);
    }
}

// ---------------------------------------------------------------------------
// Global embedding controls (provider + model) — drive every load/RAG command.
// ---------------------------------------------------------------------------

function embeddingProvider() {
    const el = document.getElementById('globalProvider');
    return el ? el.value : 'local';
}

function embeddingModel() {
    const el = document.getElementById('globalModel');
    return el ? el.value.trim() : '';
}

// Map a base (local) collection name to the collection for the selected provider.
// OpenAI (1536-dim) data lives in a parallel "<base>_openai" collection.
function collectionFor(base) {
    return embeddingProvider() === 'openai' ? `${base}_openai` : base;
}

// Append --embedding-provider and (when set) --embedding-model to a command.
function withEmbeddingFlags(cmd) {
    cmd += ` --embedding-provider ${embeddingProvider()}`;
    const model = embeddingModel();
    if (model) {
        cmd += ` --embedding-model "${model}"`;
    }
    return cmd;
}

// Update the model placeholder to the provider's default when the provider changes.
function onProviderChange() {
    const el = document.getElementById('globalModel');
    if (el) {
        el.placeholder = embeddingProvider() === 'openai'
            ? 'text-embedding-3-small (1536-dim)'
            : 'all-MiniLM-L6-v2 (384-dim)';
    }
}

// ---------------------------------------------------------------------------
// Load builders
// ---------------------------------------------------------------------------

function setLoadCmdComposite() {
    const collection = collectionFor('emails');
    const cmd = `python load.py ` +
                `--data data/dataset.csv --text-fields subject message_body ` +
                `--bucket vectorSearchDemo --scope _default --collection ${collection} ` +
                `--copy-fields subject sender receiver message_body ` +
                `--limit 5 --id-field sender timestamp`;
    setLoadCmd(withEmbeddingFlags(cmd));
}

function setLoadCmdHyperscale() {
    const collection = collectionFor('movies');
    const cmd = `python load.py ` +
                `--data data/wiki_movie_plots_deduped.csv --text-fields Plot ` +
                `--bucket vectorSearchDemo --scope _default --collection ${collection} ` +
                `--copy-fields Title "Release Year" Director ` +
                `--limit 5 --id-field Title "Release Year"`;
    setLoadCmd(withEmbeddingFlags(cmd));
}

function setLoadCmdHybrid() {
    const collection = collectionFor('yelp');
    const cmd = `python load.py ` +
                `--data data/yelp_academic_dataset_business.json --text-fields categories ` +
                `--bucket vectorSearchDemo --scope _default --collection ${collection} ` +
                `--copy-fields latitude longitude name ` +
                `--limit 5 --id-field business_id`;
    setLoadCmd(withEmbeddingFlags(cmd));
}

// ---------------------------------------------------------------------------
// RAG builders
// ---------------------------------------------------------------------------

function setRagCmdComposite() {
    const sender = document.getElementById('sender').value.trim();
    const receiver = document.getElementById('receiver').value.trim();
    const prompt = document.getElementById('prompt').value.trim();

    const collection = collectionFor('emails');

    let cmd = `python ragComposite.py ` +
              `--bucket vectorSearchDemo ` +
              `--scope _default ` +
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

    setRagCmd(withEmbeddingFlags(cmd));
}

function setRagCmdHyperscale() {
    const prompt = document.getElementById('hyperPrompt').value.trim();

    const collection = collectionFor('movies');

    let cmd = `python ragHyperscale.py ` +
              `--bucket vectorSearchDemo ` +
              `--scope _default ` +
              `--collection ${collection}`;

    if (prompt) {
        cmd += ` --prompt "${prompt}"`;
    }

    setRagCmd(withEmbeddingFlags(cmd));
}

function setRagCmdHybrid() {
    const prompt = document.getElementById('hybridPrompt').value.trim();

    const collection = collectionFor('yelp');
    // The FTS index is dimension-specific, so it swaps with the provider too.
    const indexName = embeddingProvider() === 'openai' ? 'ix-yelp-openai-vector' : 'ix-yelp-business-vector';

    const latitude = document.getElementById('hybridLatitude').value.trim();
    const longitude = document.getElementById('hybridLongitude').value.trim();
    const radius = document.getElementById('hybridRadius').value.trim();

    let cmd = `python ragHybrid.py ` +
              `--bucket vectorSearchDemo ` +
              `--scope _default ` +
              `--collection ${collection} ` +
              `--index-name ${indexName}`;

    if (prompt) {
        cmd += ` --prompt "${prompt}"`;
    }

    if (latitude && longitude && radius) {
        cmd += ` --latitude ${latitude} --longitude ${longitude} --radius ${radius}`;
    }

    setRagCmd(withEmbeddingFlags(cmd));
}
