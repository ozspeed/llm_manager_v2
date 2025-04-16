// [V2-7.1.1] Model Management UI JS (AJAX CRUD)
document.addEventListener('DOMContentLoaded', function() {
    fetchModels();

    // Upload form handler
    document.getElementById('upload-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        const form = e.target;
        const formData = new FormData(form);
        setFeedback('upload-feedback', 'Uploading...', false);
        try {
            const resp = await fetch('/api/models', {
                method: 'POST',
                body: formData
            });
            const data = await resp.json();
            if (resp.ok) {
                setFeedback('upload-feedback', 'Model uploaded!', false);
                fetchModels();
                form.reset();
            } else {
                setFeedback('upload-feedback', data.error || 'Upload failed', true);
            }
        } catch (err) {
            setFeedback('upload-feedback', err.toString(), true);
        }
    });
});

function fetchModels() {
    fetch('/api/models')
        .then(r => r.json())
        .then(models => renderModels(models))
        .catch(e => setFeedback('models-feedback', e.toString(), true));
}

function renderModels(models) {
    const list = document.getElementById('models-list');
    list.innerHTML = '';
    if (!models || models.length === 0) {
        list.innerHTML = `
            <div class="card" style="text-align:center; padding:2.5rem 1.5rem;">
                <div style="font-size:2.2rem; color:var(--primary);">📦</div>
                <div style="margin-top:1rem; font-size:1.15rem; color:var(--text);">No models in your library yet.</div>
                <div style="margin-top:0.7rem; color:#888;">Use the form above to add your first model.</div>
            </div>
        `;
        return;
    }
    for (const m of models) {
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
            <div style="display:flex; align-items:center; gap:1rem;">
                <div style="font-size:2.1rem;">🟣</div>
                <div>
                    <div style="font-size:1.13rem; font-weight:600; color:var(--primary-dark);">${escape(m.name)}</div>
                    <div style="font-size:0.98rem; color:#888;">${escape(m.framework)}${m.size_mb ? ' • ' + m.size_mb.toFixed(2) + ' MB' : ''}</div>
                </div>
                <div style="margin-left:auto; font-size:0.98rem; color:#888;">${m.last_used ? 'Last used: ' + m.last_used.split('T')[0] : ''}</div>
            </div>
            <div style="margin-top:0.7rem;">
                <span style="color:#555; font-size:0.98rem;">Publisher:</span> ${escape(m.publisher || '-')}
            </div>
            <div class="actions">
                <button onclick="showDetails(${m.id})" aria-label="Show details for ${escape(m.name)}">Details</button>
                <button onclick="deleteModel(${m.id})" class="danger" aria-label="Delete ${escape(m.name)}">Delete</button>
            </div>
        `;
        list.appendChild(card);
    }
}

function showDetails(id) {
    fetch(`/api/models/${id}`)
        .then(r => r.json())
        .then(model => {
            const content = document.getElementById('model-details-content');
            content.innerHTML = `<pre>${escape(JSON.stringify(model, null, 2))}</pre>`;
            showModal('model-details-modal');
        })
        .catch(e => alert('Failed to fetch details: ' + e));
}

function deleteModel(id) {
    if (!confirm('Delete this model?')) return;
    fetch(`/api/models/${id}`, { method: 'DELETE' })
        .then(r => r.json())
        .then(data => {
            setFeedback('models-feedback', data.message || 'Deleted', false);
            fetchModels();
        })
        .catch(e => setFeedback('models-feedback', e.toString(), true));
}

function setFeedback(id, msg, isError) {
    const el = document.getElementById(id);
    el.textContent = msg;
    el.style.color = isError ? 'red' : 'green';
}

function showModal(id) {
    const modal = document.getElementById(id);
    modal.style.display = 'block';
    modal.querySelector('.close').onclick = () => modal.style.display = 'none';
    window.onclick = function(event) {
        if (event.target === modal) modal.style.display = 'none';
    };
}

function escape(str) {
    return String(str).replace(/[&<>"]/g, function(c) {
        return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];
    });
}
