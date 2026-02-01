document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const srcText = document.getElementById('src-text');
    const tgtText = document.getElementById('tgt-text');
    const srcLangSelect = document.getElementById('src-lang-select');
    const tgtLangSelect = document.getElementById('tgt-lang-select');
    const charCount = document.getElementById('char-count');
    const timingInfo = document.getElementById('timing-info');
    const loader = document.getElementById('translation-loader');
    const detectedBadge = document.getElementById('detected-badge');
    const navLinks = document.querySelectorAll('.nav-links a');
    const views = document.querySelectorAll('.view');
    const healthIndicator = document.getElementById('health-indicator');
    const modelsList = document.getElementById('models-list');
    const copyBtn = document.getElementById('copy-btn');
    const themeToggle = document.getElementById('theme-toggle');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const sidebar = document.querySelector('.sidebar');

    let debounceTimer;
    let languages = {};
    let languageNames = {};
    let activeController = null;

    let settings = {
        beam_size: 2,
        patience: 1,
        length_penalty: 1.0,
        coverage_penalty: 0.0,
        repetition_penalty: 1.0
    };

    // 0. Theme Logic
    function initTheme() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        if (savedTheme === 'light') {
            document.body.classList.add('light-mode');
            updateThemeUI(true);
        }
    }

    function updateThemeUI(isLight) {
        const text = themeToggle.querySelector('.mode-text');
        text.textContent = isLight ? 'Light Mode' : 'Dark Mode';
    }

    themeToggle.addEventListener('click', () => {
        const isLight = document.body.classList.toggle('light-mode');
        localStorage.setItem('theme', isLight ? 'light' : 'dark');
        updateThemeUI(isLight);
    });

    // 0.1 Sidebar Logic
    function initSidebar() {
        const isCollapsed = localStorage.getItem('sidebar-collapsed') === 'true';
        if (isCollapsed) sidebar.classList.add('collapsed');
    }

    sidebarToggle.addEventListener('click', () => {
        const isCollapsed = sidebar.classList.toggle('collapsed');
        localStorage.setItem('sidebar-collapsed', isCollapsed);
    });

    // 0.2 Inference Settings Logic
    function initSettings() {
        const saved = localStorage.getItem('inference-settings');
        if (saved) {
            try {
                const parsed = JSON.parse(saved);
                settings = { ...settings, ...parsed };
            } catch (e) { console.error("Failed to parse settings", e); }
        }
        updateSettingsUI();
    }

    function updateSettingsUI() {
        // Sync values to inputs
        Object.keys(settings).forEach(key => {
            const input = document.getElementById(`setting-${key.replace('_', '-')}`);
            if (input) {
                input.value = settings[key];
                const valDisplay = input.nextElementSibling;
                if (valDisplay && valDisplay.classList.contains('setting-val')) {
                    valDisplay.textContent = settings[key];
                }
            }
        });
    }

    function saveSettings() {
        localStorage.setItem('inference-settings', JSON.stringify(settings));
    }

    // Add listeners to all settings inputs
    const settingsInputs = [
        'setting-beam-size', 'setting-patience', 'setting-length-penalty',
        'setting-coverage-penalty', 'setting-repetition-penalty'
    ];

    settingsInputs.forEach(id => {
        const input = document.getElementById(id);
        const key = id.replace('setting-', '').replace(/-/g, '_');

        input.addEventListener('input', () => {
            let val = parseFloat(input.value);
            if (id === 'setting-beam-size' || id === 'setting-patience') val = parseInt(input.value);

            settings[key] = val;

            // Enforcement: patience <= beam_size
            if (id === 'setting-beam-size') {
                if (settings.patience > settings.beam_size) {
                    settings.patience = settings.beam_size;
                    const patienceInput = document.getElementById('setting-patience');
                    patienceInput.value = settings.patience;
                    patienceInput.nextElementSibling.textContent = settings.patience;
                }
                // Update patience max slider to match beam_size for better UX? 
                // User said "maximum 10", so let's stick to that but cap the value.
            } else if (id === 'setting-patience') {
                if (val > settings.beam_size) {
                    val = settings.beam_size;
                    input.value = val;
                    settings.patience = val;
                }
            }

            const valDisplay = input.nextElementSibling;
            if (valDisplay && valDisplay.classList.contains('setting-val')) {
                valDisplay.textContent = val;
            }
            saveSettings();
        });
    });

    document.getElementById('reset-settings').addEventListener('click', () => {
        settings = {
            beam_size: 2,
            patience: 1,
            length_penalty: 1.0,
            coverage_penalty: 0.0,
            repetition_penalty: 1.0
        };
        updateSettingsUI();
        saveSettings();
    });

    // 1. Fetch available languages and populate selects
    async function initLanguages() {
        try {
            const res = await fetch('/api/languages');
            if (res.ok) {
                const data = await res.json();
                languages = data.pairs;
                languageNames = data.names;
                populateSelects();
                updateHealth(true);
            }
        } catch (e) {
            console.error("Failed to load languages", e);
            updateHealth(false);
        }
    }

    function populateSelects() {
        const currentSrc = srcLangSelect.value;
        // Keep only the first "Auto-detect" option
        srcLangSelect.innerHTML = '<option value="">Auto-detect</option>';

        const sources = Object.keys(languages);

        // Populate Source Languages
        sources.forEach(lang => {
            const opt = document.createElement('option');
            opt.value = lang;
            opt.textContent = languageNames[lang] || lang.toUpperCase();
            srcLangSelect.appendChild(opt);
        });

        // Restore selection if it still exists
        if (currentSrc && languages[currentSrc]) {
            srcLangSelect.value = currentSrc;
        }

        // Trigger target population for default selection
        updateTargetOptions();
    }

    function updateTargetOptions() {
        const src = srcLangSelect.value;
        const currentTgt = tgtLangSelect.value;

        // Clear targets
        tgtLangSelect.innerHTML = '';

        let availableTgts = [];
        if (src) {
            availableTgts = languages[src] || [];
        } else {
            // If auto-detect, union of all targets
            const allTgts = new Set();
            Object.values(languages).forEach(list => list.forEach(l => allTgts.add(l)));
            availableTgts = Array.from(allTgts).sort();
        }

        availableTgts.forEach(lang => {
            const opt = document.createElement('option');
            opt.value = lang;
            opt.textContent = languageNames[lang] || lang.toUpperCase();
            if (lang === currentTgt || (availableTgts.length === 1)) opt.selected = true;
            tgtLangSelect.appendChild(opt);
        });
    }

    // 2. Translation Logic
    async function performTranslation() {
        const fullText = srcText.value;
        if (!fullText.trim()) {
            tgtText.value = '';
            timingInfo.textContent = 'Ready';
            detectedBadge.classList.remove('visible');
            return;
        }

        // Abort previous requests
        if (activeController) activeController.abort();
        activeController = new AbortController();
        const { signal } = activeController;

        const lines = fullText.split('\n');
        const translatedLines = new Array(lines.length).fill('');
        let srcLang = srcLangSelect.value || null;
        const tgtLang = tgtLangSelect.value;

        loader.classList.remove('hidden');
        let completedLines = 0;
        let totalToTranslate = lines.filter(l => l.trim()).length;

        try {
            // Step 1: If auto-detect mode, detect language for entire input first
            if (!srcLang && fullText.trim()) {
                const detectResponse = await fetch('/api/identify-language', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        src: fullText,
                        k: 1,
                        threshold: 0.0
                    }),
                    signal
                });

                if (detectResponse.ok) {
                    const detectData = await detectResponse.json();
                    // Get the detected language from the response
                    if (detectData.results && detectData.results.length > 0) {
                        srcLang = detectData.results[0].lang;
                        const srcName = languageNames[srcLang] || srcLang.toUpperCase();
                        detectedBadge.textContent = `Detected: ${srcName}`;
                        detectedBadge.classList.add('visible');
                    }
                }
            }

            // Step 2: Translate all lines with known source language
            const updateTgtUI = () => {
                tgtText.value = translatedLines.join('\n');
            };

            const translateParagraph = async (line, index) => {
                if (!line.trim()) {
                    translatedLines[index] = line;
                    updateTgtUI();
                    return;
                }

                try {
                    const response = await fetch('/api/translate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            src: line,
                            src_lang: srcLang,  // Now we always have a source language
                            tgt_lang: tgtLang,
                            ...settings
                        }),
                        signal
                    });

                    if (response.ok) {
                        const data = await response.json();
                        translatedLines[index] = data.translation;

                        completedLines++;
                        updateTgtUI();
                        timingInfo.textContent = `Translating: ${Math.round((completedLines / totalToTranslate) * 100)}%`;
                    }
                } catch (e) {
                    if (e.name !== 'AbortError') {
                        console.error("Line translation error", e);
                        translatedLines[index] = `[[Error: ${line}]]`;
                    }
                } finally {
                    if (completedLines === totalToTranslate) {
                        loader.classList.add('hidden');
                        timingInfo.textContent = 'Done';
                    }
                }
            };

            // Fire all translation requests in parallel
            lines.forEach((line, i) => translateParagraph(line, i));

        } catch (e) {
            if (e.name !== 'AbortError') {
                console.error("Translation error", e);
                loader.classList.add('hidden');
                timingInfo.textContent = 'Error';
            }
        }
    }

    // 3. Models View
    async function fetchModels() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();

            modelsList.innerHTML = '';

            // Use DocumentFragment for better performance
            const fragment = document.createDocumentFragment();

            data.models.forEach(m => {
                const card = document.createElement('div');
                card.className = 'model-card';
                card.innerHTML = `
                    <div class="model-lang-pair">
                        <span>${m.src_name || m.src_lang.toUpperCase()}</span>
                        <span>→</span>
                        <span>${m.tgt_name || m.tgt_lang.toUpperCase()}</span>
                    </div>
                    <div class="model-id">${m.model_id}</div>
                    ${m.loaded ? '<span class="loaded-badge">Currently Loaded</span>' : ''}
                `;
                fragment.appendChild(card);
            });

            // Single DOM update instead of multiple
            modelsList.appendChild(fragment);
        } catch (e) {
            modelsList.innerHTML = '<p>Error loading models</p>';
        }
    }

    // 4. UI Helpers
    function updateHealth(isOnline) {
        if (isOnline) {
            healthIndicator.className = 'status-pill status-online';
            healthIndicator.querySelector('.status-text').textContent = 'Online';
        } else {
            healthIndicator.className = 'status-pill status-loading';
            healthIndicator.querySelector('.status-text').textContent = 'Offline';
        }
    }

    // Event Listeners
    srcText.addEventListener('input', () => {
        charCount.textContent = `${srcText.value.length} characters`;
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(performTranslation, 250);
    });

    srcLangSelect.addEventListener('change', () => {
        updateTargetOptions();
        performTranslation();
    });

    tgtLangSelect.addEventListener('change', performTranslation);

    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(tgtText.value);
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        setTimeout(() => copyBtn.textContent = originalText, 2000);
    });

    // Navigation
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);

            navLinks.forEach(l => l.parentElement.classList.remove('active'));
            link.parentElement.classList.add('active');

            views.forEach(v => {
                v.classList.remove('active');
                if (v.id === `${targetId}-view`) v.classList.add('active');
            });

            if (targetId === 'models') fetchModels();
        });
    });

    // Start
    initTheme();
    initSidebar();
    initSettings();
    initLanguages();
    setInterval(initLanguages, 10000); // Pulse health check
});
