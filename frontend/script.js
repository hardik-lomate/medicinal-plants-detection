/**
 * PlantScope — Botanical Computer Vision & Species Identification
 * Client Application Logic (v4.0)
 * 
 * Features:
 * - Professional SVG icon system (zero emojis)
 * - Accessible toast notification dispatch (no browser alert popups)
 * - Camera viewfinder controls with stream cleanup
 * - Multi-stage analysis progression and visual scan hairline
 * - Monograph formatting for identified, uncertain, and botanical-only states
 * - Interactive botanical species directory with search & category filters
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- SVG Icon Library (Clean line icons) ---
    const ICONS = {
        check: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`,
        info: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`,
        alert: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
        shield: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>`,
        leaf: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>`,
        plant: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>`,
        external: `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>`,
        chevronDown: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>`,
        chevronUp: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>`,
        close: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`
    };

    // --- DOM Elements: Inputs & Controls ---
    const uploadInput = document.getElementById('upload-input');
    const takePhotoBtn = document.getElementById('take-photo-btn');
    const dropZone = document.getElementById('drop-zone');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const scanOverlay = document.getElementById('scan-overlay');
    const previewFocusBadge = document.getElementById('preview-focus-badge');
    const previewFilename = document.getElementById('preview-filename');
    const identifyBtn = document.getElementById('identify-btn');
    const resetBtn = document.getElementById('reset-btn');
    const focusSegmentedControl = document.getElementById('focus-segmented-control');

    // --- DOM Elements: States ---
    const idleState = document.getElementById('idle-state');
    const loadingState = document.getElementById('loading-state');
    const resultSection = document.getElementById('result-section');
    const resultContent = document.getElementById('result-content');
    const toastContainer = document.getElementById('toast-container');

    // --- DOM Elements: Camera Modal ---
    const cameraModal = document.getElementById('camera-modal');
    const cameraFeed = document.getElementById('camera-feed');
    const captureBtn = document.getElementById('capture-btn');
    const closeCameraBtn = document.getElementById('close-camera-btn');
    const closeCameraX = document.getElementById('close-camera-x');

    // --- DOM Elements: Species Directory Modal ---
    const browseSpeciesBtn = document.getElementById('browse-species-btn');
    const speciesModal = document.getElementById('species-modal');
    const closeSpeciesX = document.getElementById('close-species-x');
    const speciesSearchInput = document.getElementById('species-search-input');
    const speciesGrid = document.getElementById('species-grid');
    const totalSpeciesCount = document.getElementById('total-species-count');
    const filterChips = document.querySelectorAll('.filter-chip');

    // --- Application State ---
    let currentStream = null;
    let selectedFile = null;
    let currentFocusMode = 'auto'; // 'auto', 'leaf', 'whole_plant'
    let allSpeciesList = [];
    let activeFilter = 'all';

    // ==========================================
    // 1. Toast Notification System
    // ==========================================
    function showToast(message, type = 'info') {
        if (!toastContainer) return;
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        let iconHtml = ICONS.info;
        if (type === 'error' || type === 'warning') iconHtml = ICONS.alert;
        if (type === 'success') iconHtml = ICONS.check;

        toast.innerHTML = `
            <div style="display: flex; align-items: flex-start; gap: 10px;">
                <span style="flex-shrink: 0; margin-top: 1px;">${iconHtml}</span>
                <span style="flex: 1; line-height: 1.45;">${escapeHtml(message)}</span>
            </div>
        `;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(8px)';
            toast.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
            setTimeout(() => {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            }, 250);
        }, 4200);
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    // ==========================================
    // 2. Focus Mode Selection Bar
    // ==========================================
    if (focusSegmentedControl) {
        focusSegmentedControl.addEventListener('click', (e) => {
            const btn = e.target.closest('.segment-btn');
            if (!btn) return;

            focusSegmentedControl.querySelectorAll('.segment-btn').forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-checked', 'false');
            });
            btn.classList.add('active');
            btn.setAttribute('aria-checked', 'true');
            currentFocusMode = btn.dataset.type;

            const modeLabels = {
                'auto': 'Mode: Auto-detect (Stage 1 Router)',
                'leaf': 'Mode: Leaf close-up classifier',
                'whole_plant': 'Mode: Whole-plant classifier'
            };
            if (previewFocusBadge) {
                previewFocusBadge.textContent = modeLabels[currentFocusMode] || 'Mode: Auto-detect';
            }
        });
    }

    // ==========================================
    // 3. File Selection & Drag-and-Drop
    // ==========================================
    function handleFileSelection(file) {
        if (!file || !file.type.startsWith('image/')) {
            showToast('Please select a valid image file (JPG, PNG, or WEBP).', 'warning');
            return;
        }

        if (file.size > 20 * 1024 * 1024) {
            showToast('File size exceeds 20MB limit. Please upload a smaller photograph.', 'warning');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            if (previewFilename) {
                previewFilename.textContent = file.name || 'Photograph loaded';
            }
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            
            // Reset right panel to idle state until user triggers analysis
            if (resultSection) resultSection.classList.add('hidden');
            if (idleState) idleState.classList.remove('hidden');
            if (loadingState) loadingState.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    if (uploadInput) {
        uploadInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFileSelection(e.target.files[0]);
            }
        });
    }

    if (dropZone) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('dragover');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length > 0) {
                handleFileSelection(dt.files[0]);
            }
        });
    }

    function resetWorkflow() {
        selectedFile = null;
        if (uploadInput) uploadInput.value = '';
        if (imagePreview) imagePreview.src = '';
        if (scanOverlay) scanOverlay.classList.remove('active');
        if (previewContainer) previewContainer.classList.add('hidden');
        if (dropZone) dropZone.classList.remove('hidden');
        if (resultSection) resultSection.classList.add('hidden');
        if (loadingState) loadingState.classList.add('hidden');
        if (idleState) idleState.classList.remove('hidden');
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', resetWorkflow);
    }

    // ==========================================
    // 4. Device Camera Capture
    // ==========================================
    if (takePhotoBtn) {
        takePhotoBtn.addEventListener('click', async () => {
            try {
                currentStream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
                    audio: false
                });
                cameraFeed.srcObject = currentStream;
                cameraModal.classList.remove('hidden');
            } catch (err) {
                showToast('Camera access unavailable. Please grant browser camera permissions or upload a photograph file.', 'error');
                console.error('Camera error:', err);
            }
        });
    }

    function stopCamera() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }
        if (cameraModal) cameraModal.classList.add('hidden');
    }

    if (closeCameraBtn) closeCameraBtn.addEventListener('click', stopCamera);
    if (closeCameraX) closeCameraX.addEventListener('click', stopCamera);

    if (captureBtn) {
        captureBtn.addEventListener('click', () => {
            if (!cameraFeed) return;
            const canvas = document.createElement('canvas');
            canvas.width = cameraFeed.videoWidth || 640;
            canvas.height = cameraFeed.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);

            canvas.toBlob((blob) => {
                if (blob) {
                    const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
                    handleFileSelection(file);
                    stopCamera();
                    showToast('Photograph captured from camera viewfinder.', 'success');
                }
            }, 'image/jpeg', 0.95);
        });
    }

    // ==========================================
    // 5. Species Directory Modal
    // ==========================================
    if (browseSpeciesBtn && speciesModal) {
        browseSpeciesBtn.addEventListener('click', () => {
            speciesModal.classList.remove('hidden');
            if (allSpeciesList.length === 0) {
                loadSpeciesDirectory();
            }
        });

        if (closeSpeciesX) {
            closeSpeciesX.addEventListener('click', () => {
                speciesModal.classList.add('hidden');
            });
        }

        speciesModal.addEventListener('click', (e) => {
            if (e.target === speciesModal) {
                speciesModal.classList.add('hidden');
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (!cameraModal.classList.contains('hidden')) stopCamera();
                if (!speciesModal.classList.contains('hidden')) speciesModal.classList.add('hidden');
            }
        });
    }

    async function loadSpeciesDirectory() {
        if (!speciesGrid) return;
        speciesGrid.innerHTML = '<div style="padding: 24px; color: var(--text-muted); font-size: 0.875rem;">Loading catalog entries...</div>';

        try {
            const res = await fetch('/plants');
            const data = await res.json();
            if (data.success && data.plants) {
                allSpeciesList = data.plants;
                if (totalSpeciesCount) totalSpeciesCount.textContent = allSpeciesList.length;
                renderSpeciesGrid(allSpeciesList);
            } else {
                speciesGrid.innerHTML = '<p class="notice-card notice-warning">Unable to load species directory from server.</p>';
            }
        } catch (err) {
            console.error('Error loading species:', err);
            speciesGrid.innerHTML = '<p class="notice-card notice-warning">Could not establish connection to the species catalog service.</p>';
        }
    }

    function renderSpeciesGrid(plants) {
        if (!speciesGrid) return;
        if (!plants || plants.length === 0) {
            speciesGrid.innerHTML = '<div style="padding: 32px; text-align: center; color: var(--text-muted); font-size: 0.875rem;">No cataloged species match your query.</div>';
            return;
        }

        let html = '';
        plants.forEach(plant => {
            const isMed = plant.is_medicinal;
            const hasDb = plant.has_medicinal_info;

            html += `
                <div class="catalog-card">
                    <div>
                        <div class="catalog-card-header">
                            <h4>${escapeHtml(plant.common_name)}</h4>
                        </div>
                        <p class="catalog-sci-name">${escapeHtml(plant.scientific_name || 'Binomial not cataloged')}</p>
                    </div>
                    <div class="catalog-tags">
                        ${isMed ? '<span class="cat-tag med">Medicinal</span>' : '<span class="cat-tag">Botanical</span>'}
                        ${hasDb ? '<span class="cat-tag">Monograph Corroborated</span>' : '<span class="cat-tag">Taxonomy Only</span>'}
                        ${plant.leaf_available ? '<span class="cat-tag">Leaf</span>' : ''}
                        ${plant.whole_plant_available ? '<span class="cat-tag">Plant</span>' : ''}
                    </div>
                </div>
            `;
        });
        speciesGrid.innerHTML = html;
    }

    if (speciesSearchInput) {
        speciesSearchInput.addEventListener('input', filterSpecies);
    }

    filterChips.forEach(chip => {
        chip.addEventListener('click', () => {
            filterChips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            activeFilter = chip.dataset.filter;
            filterSpecies();
        });
    });

    function filterSpecies() {
        const query = (speciesSearchInput ? speciesSearchInput.value : '').toLowerCase().trim();
        let filtered = allSpeciesList.filter(p => {
            const matchesQuery = p.common_name.toLowerCase().includes(query) ||
                                 (p.scientific_name && p.scientific_name.toLowerCase().includes(query));
            if (!matchesQuery) return false;

            if (activeFilter === 'medicinal') return p.is_medicinal;
            if (activeFilter === 'db_ready') return p.has_medicinal_info;
            return true;
        });
        renderSpeciesGrid(filtered);
    }

    // ==========================================
    // 6. Neural Identification Pipeline
    // ==========================================
    if (identifyBtn) {
        identifyBtn.addEventListener('click', async () => {
            if (!selectedFile) {
                showToast('Please select or capture a photograph first.', 'warning');
                return;
            }

            // UI State transition: Activate loading indicator and scanning beam
            if (idleState) idleState.classList.add('hidden');
            if (resultSection) resultSection.classList.add('hidden');
            if (loadingState) loadingState.classList.remove('hidden');
            if (scanOverlay) scanOverlay.classList.add('active');

            // Animate progress step cues
            const stepRows = [
                document.getElementById('step-row-1'),
                document.getElementById('step-row-2'),
                document.getElementById('step-row-3')
            ];
            stepRows.forEach(r => r && r.classList.remove('active'));
            if (stepRows[0]) stepRows[0].classList.add('active');

            const t1 = setTimeout(() => {
                if (stepRows[1]) stepRows[1].classList.add('active');
            }, 550);
            const t2 = setTimeout(() => {
                if (stepRows[2]) stepRows[2].classList.add('active');
            }, 1100);

            const formData = new FormData();
            formData.append('image', selectedFile);
            if (currentFocusMode && currentFocusMode !== 'auto') {
                formData.append('input_type', currentFocusMode);
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errData = await response.json().catch(() => ({}));
                    throw new Error(errData.error || `Server responded with HTTP ${response.status}`);
                }

                const data = await response.json();
                displayResults(data);
            } catch (error) {
                showToast(`Classification error: ${error.message}`, 'error');
                console.error('Inference error:', error);
                if (idleState) idleState.classList.remove('hidden');
            } finally {
                clearTimeout(t1);
                clearTimeout(t2);
                if (loadingState) loadingState.classList.add('hidden');
                if (scanOverlay) scanOverlay.classList.remove('active');
            }
        });
    }

    // ==========================================
    // 7. Results & Monograph Rendering
    // ==========================================
    function displayResults(data) {
        if (!resultContent || !resultSection) return;
        resultContent.innerHTML = '';

        const status = data.status || 'identified';
        const alternatives = data.alternatives || [];
        const pred = data.prediction || {};
        const confPercent = Math.round((data.confidence || pred.confidence || 0) * 100);
        const marginPct = data.margin !== undefined ? Math.round(data.margin * 100) : 0;
        const inputType = data.input_type || pred.input_type || 'leaf';
        const inputTypeLabel = inputType === 'whole_plant' ? 'Whole plant' : 'Leaf close-up';
        const inputTypeIcon = inputType === 'whole_plant' ? ICONS.plant : ICONS.leaf;

        // --- State A: Bad Image Quality ---
        if (status === 'bad_image') {
            resultContent.innerHTML = `
                <div class="notice-card notice-warning">
                    <h3>Photograph Unsuitable for Botanical Identification</h3>
                    <p>${escapeHtml(data.message || 'The uploaded photograph does not exhibit distinguishable botanical features.')}</p>
                    <div style="margin-top: 16px; font-size: 0.8125rem; line-height: 1.6; color: #7B5113;">
                        <strong>Guidelines for reliable computer-vision results:</strong>
                        <ul style="padding-left: 20px; margin-top: 6px;">
                            <li>Position the primary leaf flat and centered in the frame.</li>
                            <li>Ensure diffuse, even lighting without direct flash glare or heavy shadows.</li>
                            <li>Keep the camera steady to prevent motion blur.</li>
                        </ul>
                    </div>
                </div>
                <div class="result-actions-bottom">
                    <button type="button" class="btn btn-secondary btn-block" onclick="document.getElementById('reset-btn').click()">
                        Try another photograph
                    </button>
                </div>
            `;
            resultSection.classList.remove('hidden');
            return;
        }

        // --- State B: Unknown Non-Plant / OOD ---
        if (status === 'unknown') {
            resultContent.innerHTML = `
                <div class="notice-card notice-warning">
                    <h3>Species Not Confidently Recognized</h3>
                    <p>${escapeHtml(data.message || 'The photograph does not strongly match any cataloged botanical species in the model database.')}</p>
                    <p style="font-size: 0.8125rem; color: #7B5113; margin-top: 8px;">
                        The model detected low morphological similarity with our 90 Indian medicinal and botanical reference classes.
                    </p>
                </div>

                ${renderAlternativesAccordion(alternatives, 'Nearest candidate classifications')}

                ${data.disclaimer ? `<p class="footer-disclaimer" style="margin-top: 20px;">${escapeHtml(data.disclaimer)}</p>` : ''}

                <div class="result-actions-bottom">
                    <button type="button" class="btn btn-secondary btn-block" onclick="document.getElementById('reset-btn').click()">
                        Upload a different specimen
                    </button>
                </div>
            `;
            resultSection.classList.remove('hidden');
            return;
        }

        // --- State C: Uncertain Identification ---
        if (status === 'uncertain') {
            resultContent.innerHTML = `
                <div class="notice-card notice-warning">
                    <h3>Plant identification is uncertain.</h3>
                    <p style="font-size: 0.9375rem; font-weight: 500; margin-bottom: 6px; color: #5C3D0E;">
                        Try a clearer photo with the plant or leaf centered in the image.
                    </p>
                    <p style="font-size: 0.8125rem; margin-bottom: 12px; color: #7B5113;">
                        ${escapeHtml(data.message || 'The photograph does not provide sufficient detail to distinguish between candidate species with high confidence.')}
                    </p>
                    <div style="font-size: 0.8125rem; color: #7B5113; line-height: 1.5;">
                        <strong>Recommendations:</strong>
                        <ul style="padding-left: 20px; margin-top: 4px;">
                            <li>If taking a photo of a single leaf, ensure margin serration and primary veins are crisp.</li>
                            <li>If capturing a shrub or foliage canopy, toggle the <strong>Whole plant</strong> focus mode.</li>
                        </ul>
                    </div>
                </div>

                ${renderAlternativesAccordion(alternatives, 'Candidate probability breakdown')}

                ${data.disclaimer ? `<p class="footer-disclaimer" style="margin-top: 20px;">${escapeHtml(data.disclaimer)}</p>` : ''}

                <div class="result-actions-bottom">
                    <button type="button" class="btn btn-secondary btn-block" onclick="document.getElementById('reset-btn').click()">
                        Try a clearer photo
                    </button>
                </div>
            `;
            resultSection.classList.remove('hidden');
            return;
        }

        // --- State D: Confident Identification (with DB Monograph OR Botanical-Only) ---
        const hasVerifiedMonograph = Boolean(
            pred.medicinal_information_available && 
            pred.medicinal_uses && 
            pred.medicinal_uses.length > 0
        );

        // Determine Confidence Bar Styling
        let confFillClass = 'conf-fill-high';
        if (confPercent < 65) confFillClass = 'conf-fill-low';
        else if (confPercent < 80) confFillClass = 'conf-fill-medium';

        let html = `
            <!-- Primary Result Header -->
            <div class="result-primary-header">
                <div class="result-header-meta">
                    <span class="result-overline">Stage 2 Identification</span>
                    <span class="res-badge ${hasVerifiedMonograph ? 'badge-med-ok' : 'badge-med-none'}">
                        ${hasVerifiedMonograph ? 'Verified monograph' : 'Botanical record'}
                    </span>
                </div>
                <h2 class="result-plant-name">${escapeHtml(pred.plant_name || 'Identified Plant')}</h2>
                <div class="result-scientific-name">${escapeHtml(pred.scientific_name || 'Binomial not cataloged')}</div>
                <div class="result-badges-row">
                    <span class="res-badge" style="display: inline-flex; align-items: center; gap: 5px;">
                        ${inputTypeIcon} <span>${inputTypeLabel}</span>
                    </span>
                    ${data.confidence_tier ? `<span class="res-badge">${data.confidence_tier} tier</span>` : ''}
                    ${data.prediction_stability !== undefined ? `<span class="res-badge">${Math.round(data.prediction_stability * 100)}% stability</span>` : ''}
                    ${marginPct > 0 ? `<span class="res-badge">+${marginPct}% margin</span>` : ''}
                </div>
            </div>

            <!-- Confidence Metric Strip -->
            <div class="confidence-strip">
                <div class="conf-meta-row">
                    <span class="conf-title">Calibrated Confidence</span>
                    <span class="conf-percentage">${confPercent}%</span>
                </div>
                <div class="conf-bar-track">
                    <div class="conf-bar-fill ${confFillClass}" style="width: ${confPercent}%"></div>
                </div>
                ${data.confidence_interpretation ? `
                    <p style="font-size: 0.78125rem; color: var(--text-muted); margin-top: 6px; line-height: 1.45;">
                        ${escapeHtml(data.confidence_interpretation)}
                    </p>
                ` : ''}
            </div>

            <!-- Two-Column Botanical Specs Grid -->
            <div class="botanical-specs-grid">
                <div class="spec-cell">
                    <span class="spec-cell-label">Common Name</span>
                    <span class="spec-cell-value">${escapeHtml(pred.plant_name || 'N/A')}</span>
                </div>
                <div class="spec-cell">
                    <span class="spec-cell-label">Scientific Binomial</span>
                    <span class="spec-cell-value italic">${escapeHtml(pred.scientific_name || 'N/A')}</span>
                </div>
                <div class="spec-cell">
                    <span class="spec-cell-label">Morphological Input</span>
                    <span class="spec-cell-value">${inputTypeLabel}</span>
                </div>
                <div class="spec-cell">
                    <span class="spec-cell-label">Database Monograph</span>
                    <span class="spec-cell-value">${hasVerifiedMonograph ? 'Available & verified' : 'Taxonomic record only'}</span>
                </div>
                ${data.image_quality !== undefined ? `
                    <div class="spec-cell">
                        <span class="spec-cell-label">Photo Quality Score</span>
                        <span class="spec-cell-value">${Math.round(data.image_quality * 100)}%</span>
                    </div>
                ` : ''}
                ${data.embedding_similarity !== undefined ? `
                    <div class="spec-cell">
                        <span class="spec-cell-label">Feature Alignment</span>
                        <span class="spec-cell-value">${Math.round(data.embedding_similarity * 100)}% prototype similarity</span>
                    </div>
                ` : ''}
            </div>
        `;

        if (hasVerifiedMonograph) {
            // Full Monograph Presentation
            html += `
                <!-- Overview / Description -->
                <p class="monograph-overview">
                    ${escapeHtml(pred.description || 'Verified botanical record for medicinal flora.')}
                </p>

                <!-- Evidence-Supported Uses (Tags) -->
                ${pred.medicinal_uses && pred.medicinal_uses.length > 0 ? `
                    <div class="monograph-section">
                        <h4 class="monograph-section-heading">Evidence-supported medicinal uses</h4>
                        <div class="use-tags-list">
                            ${pred.medicinal_uses.map(u => `<span class="use-tag-item">${escapeHtml(u)}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}

                <!-- Traditional Applications (Bullets) -->
                ${pred.traditional_uses && pred.traditional_uses.length > 0 ? `
                    <div class="monograph-section">
                        <h4 class="monograph-section-heading">Traditional & Ayurvedic applications</h4>
                        <ul class="monograph-bullets">
                            ${pred.traditional_uses.map(t => `<li>${escapeHtml(t)}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}

                <!-- Parts Utilized (Tags) -->
                ${pred.parts_used && pred.parts_used.length > 0 ? `
                    <div class="monograph-section">
                        <h4 class="monograph-section-heading">Parts utilized</h4>
                        <div class="use-tags-list">
                            ${pred.parts_used.map(p => `<span class="use-tag-item">${escapeHtml(p)}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}

                <!-- Precautions & Contraindications -->
                ${pred.precautions && pred.precautions.length > 0 ? `
                    <div class="precautions-box">
                        <h5 class="precautions-heading">Important precautions & contraindications</h5>
                        <ul class="precautions-list">
                            ${pred.precautions.map(pr => `<li>${escapeHtml(pr)}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}

                <!-- Authoritative Citations -->
                ${pred.sources && pred.sources.length > 0 ? `
                    <div class="monograph-section">
                        <h4 class="monograph-section-heading">Corroborating references</h4>
                        <ul class="sources-list">
                            ${pred.sources.map(s => {
                                const isUrl = s.startsWith('http://') || s.startsWith('https://');
                                return isUrl ? 
                                    `<li class="source-item"><a href="${escapeHtml(s)}" target="_blank" rel="noopener noreferrer"><span>${escapeHtml(s)}</span> ${ICONS.external}</a></li>` :
                                    `<li class="source-item" style="font-size: 0.8125rem; color: var(--text-muted);">${escapeHtml(s)}</li>`;
                            }).join('')}
                        </ul>
                    </div>
                ` : ''}
            `;
        } else {
            // Botanical Only State (No database medicinal data)
            html += `
                <div class="notice-card notice-info">
                    <h4>Plant identified</h4>
                    <p>No verified medicinal information is available for this plant in our database.</p>
                </div>
                <p class="monograph-overview">
                    ${escapeHtml(pred.description || 'This species is recognized by the botanical computer-vision model, but clinical medicinal monographs have not yet been corroborated for this entry.')}
                </p>
            `;
        }

        // Grad-CAM Visual Attention Heatmap (Explainability)
        if (data.explanation_available && data.explanation_image) {
            html += `
                <div class="gradcam-card">
                    <div class="gradcam-header">
                        <span class="gradcam-title">Neural Attention Map (Grad-CAM)</span>
                        <span class="gradcam-hint">EfficientNet-B0 activations</span>
                    </div>
                    <div class="gradcam-image-frame">
                        <img src="${data.explanation_image}" alt="Grad-CAM Visual Attention Heatmap" loading="lazy">
                    </div>
                    <p class="gradcam-caption">
                        Heatmap highlights high-gradient anatomical regions (e.g. leaf margins, venation, apex) driving the model's identification.
                    </p>
                </div>
            `;
        }

        // Secondary Candidate Matches Accordion
        html += renderAlternativesAccordion(alternatives, 'Other candidate matches');

        // Advisory Note
        if (data.disclaimer) {
            html += `
                <p class="footer-disclaimer" style="margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--border-subtle);">
                    ${escapeHtml(data.disclaimer)}
                </p>
            `;
        }

        // Bottom Action Row
        html += `
            <div class="result-actions-bottom">
                <button type="button" class="btn btn-secondary btn-block" onclick="document.getElementById('reset-btn').click()">
                    Analyze another photograph
                </button>
            </div>
        `;

        resultContent.innerHTML = html;
        resultSection.classList.remove('hidden');

        // Attach Accordion Toggle Handlers
        attachAccordionListeners();
    }

    // Helper: Secondary Alternatives Accordion
    function renderAlternativesAccordion(alternatives, title) {
        if (!alternatives || alternatives.length <= 1) return '';

        const secondaryList = alternatives.slice(1);
        if (secondaryList.length === 0) return '';

        return `
            <div class="alternatives-container">
                <button type="button" class="alt-toggle-btn" aria-expanded="false">
                    <span>${title} (${secondaryList.length})</span>
                    <span class="alt-chevron">${ICONS.chevronDown}</span>
                </button>
                <div class="alt-collapse-wrap hidden">
                    ${secondaryList.map((alt, idx) => {
                        const pct = Math.round(alt.confidence * 100);
                        const cleanName = alt.name.replace(/_/g, ' ');
                        return `
                            <div class="alt-row-item">
                                <span class="alt-rank">#${idx + 2}</span>
                                <span class="alt-label">${escapeHtml(cleanName)}</span>
                                <div class="alt-track">
                                    <div class="alt-fill" style="width: ${pct}%"></div>
                                </div>
                                <span class="alt-pct">${pct}%</span>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
    }

    function attachAccordionListeners() {
        const toggleBtns = resultContent.querySelectorAll('.alt-toggle-btn');
        toggleBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const wrap = btn.nextElementSibling;
                if (!wrap) return;
                const isExpanded = btn.getAttribute('aria-expanded') === 'true';
                btn.setAttribute('aria-expanded', !isExpanded);
                if (isExpanded) {
                    wrap.classList.add('hidden');
                    btn.querySelector('.alt-chevron').innerHTML = ICONS.chevronDown;
                } else {
                    wrap.classList.remove('hidden');
                    btn.querySelector('.alt-chevron').innerHTML = ICONS.chevronUp;
                }
            });
        });
    }
});
