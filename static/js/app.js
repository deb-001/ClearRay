/**
 * ESRGAN XAAHA — Frontend Application Logic
 * Handles: mode tabs, model selection, upload, drag-and-drop, API calls,
 * comparison slider, animated metrics, and multi-model comparison.
 */

// ─── DOM Elements ────────────────────────────────────────────────────────────
const heroSection = document.getElementById('heroSection');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const compareSection = document.getElementById('compareSection');
const fileInput = document.getElementById('fileInput');

// Mode tabs & panels
const tabSingle = document.getElementById('tabSingle');
const tabCompare = document.getElementById('tabCompare');
const panelSingle = document.getElementById('panelSingle');
const panelCompare = document.getElementById('panelCompare');

// Single mode elements
const uploadZone = document.getElementById('uploadZone');
const btnUploadSingle = document.getElementById('btnUploadSingle');
const btnSampleSingle = document.getElementById('btnSampleSingle');
const modelSelect = document.getElementById('modelSelect');

// Compare mode elements
const uploadZoneCompare = document.getElementById('uploadZoneCompare');
const btnUploadCompare = document.getElementById('btnUploadCompare');
const btnSampleCompare = document.getElementById('btnSampleCompare');

// New image / navigation
const btnNewImage = document.getElementById('btnNewImage');
const btnNewImageCompare = document.getElementById('btnNewImageCompare');

// Loading
const loadingTitle = document.getElementById('loadingTitle');
const loadingSteps = document.getElementById('loadingSteps');
const progressFill = document.getElementById('progressFill');

// Results elements
const psnrValue = document.getElementById('psnrValue');
const ssimValue = document.getElementById('ssimValue');
const dimValue = document.getElementById('dimValue');
const modelUsedName = document.getElementById('modelUsedName');
const srBadgeLabel = document.getElementById('srBadgeLabel');

// Slider
const comparisonContainer = document.getElementById('comparisonContainer');
const compSR = document.getElementById('compSR');
const sliderHandle = document.getElementById('sliderHandle');
const sliderLR = document.getElementById('sliderLR');
const sliderSR = document.getElementById('sliderSR');

// Triple images
const tripleGT = document.getElementById('tripleGT');
const tripleLR = document.getElementById('tripleLR');
const tripleSR = document.getElementById('tripleSR');
const gtDim = document.getElementById('gtDim');
const lrDim = document.getElementById('lrDim');
const srDim = document.getElementById('srDim');

// Compare elements
const compareTableBody = document.getElementById('compareTableBody');
const compareGrid = document.getElementById('compareGrid');

// ─── State ───────────────────────────────────────────────────────────────────
let currentMode = 'single'; // 'single' | 'compare'
let pendingAction = null;   // tracks what the file input should do on change


// ─── Mode Tab Switching ──────────────────────────────────────────────────────
function switchMode(mode) {
    currentMode = mode;

    tabSingle.classList.toggle('active', mode === 'single');
    tabCompare.classList.toggle('active', mode === 'compare');
    panelSingle.classList.toggle('active', mode === 'single');
    panelCompare.classList.toggle('active', mode === 'compare');
}

tabSingle.addEventListener('click', () => switchMode('single'));
tabCompare.addEventListener('click', () => switchMode('compare'));


// ─── View Management ─────────────────────────────────────────────────────────
function showView(view) {
    heroSection.style.display = view === 'hero' ? '' : 'none';
    loadingSection.style.display = view === 'loading' ? '' : 'none';
    resultsSection.style.display = view === 'results' ? '' : 'none';
    compareSection.style.display = view === 'compare' ? '' : 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}


// ─── Loading Animation ──────────────────────────────────────────────────────
let loadingInterval = null;

function startLoading(title, messages) {
    showView('loading');
    loadingTitle.textContent = title || 'Processing Image';
    let step = 0;
    progressFill.style.width = '0%';
    loadingSteps.textContent = messages[0];

    loadingInterval = setInterval(() => {
        step++;
        if (step < messages.length) {
            loadingSteps.textContent = messages[step];
            progressFill.style.width = `${(step / messages.length) * 90}%`;
        }
    }, 1500);
}

function stopLoading() {
    clearInterval(loadingInterval);
    progressFill.style.width = '100%';
}

const singleLoadingMsgs = [
    'Preparing image for processing...',
    'Downscaling image by 4×...',
    'Running super-resolution model...',
    'Applying attention mechanisms...',
    'Reconstructing high-resolution image...',
    'Computing PSNR & SSIM metrics...',
    'Finalizing results...',
];

const compareLoadingMsgs = [
    'Preparing image for comparison...',
    'Downscaling image by 4×...',
    'Running ESRGAN + XAAHA...',
    'Running SRCNN...',
    'Running SRGAN...',
    'Running SRNet...',
    'Running SRDiff...',
    'Running multi-SR...',
    'Computing metrics for all models...',
    'Ranking models by performance...',
    'Finalizing comparison...',
];


// ─── Animated Counter ────────────────────────────────────────────────────────
function animateValue(element, target, decimals = 2, duration = 1200) {
    const start = 0;
    const startTime = performance.now();
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (target - start) * eased;
        element.textContent = current.toFixed(decimals);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}


// ─── Display Single Model Results ────────────────────────────────────────────
function displayResults(data) {
    stopLoading();

    // Model badge
    modelUsedName.textContent = data.model_name || 'ESRGAN + XAAHA';
    srBadgeLabel.textContent = data.model_name || 'ESRGAN + XAAHA';

    // Set slider images
    sliderLR.src = `data:image/png;base64,${data.low_res}`;
    sliderSR.src = `data:image/png;base64,${data.super_res}`;

    // Set triple images
    tripleGT.src = `data:image/png;base64,${data.ground_truth}`;
    tripleLR.src = `data:image/png;base64,${data.low_res}`;
    tripleSR.src = `data:image/png;base64,${data.super_res}`;

    if (data.dimensions) {
        gtDim.textContent = data.dimensions.original;
        lrDim.textContent = data.dimensions.low_res;
        srDim.textContent = data.dimensions.super_res;
        dimValue.textContent = data.dimensions.original;
    }

    setSliderPosition(50);

    setTimeout(() => {
        showView('results');
        setTimeout(() => {
            animateValue(psnrValue, data.psnr, 3, 1500);
            animateValue(ssimValue, data.ssim, 4, 1500);
        }, 200);
    }, 400);
}


// ─── Display Comparison Results ──────────────────────────────────────────────
function displayComparison(data) {
    stopLoading();

    // Build metrics table
    compareTableBody.innerHTML = '';
    data.models.forEach((m, idx) => {
        const rank = idx + 1;
        const rankClass = rank <= 3 ? `rank-${rank}` : '';
        const statusHtml = m.trained
            ? '<span class="status-trained">✓ Trained</span>'
            : '<span class="status-init">Initialized</span>';

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="rank-cell ${rankClass}">#${rank}</td>
            <td class="model-cell">${m.model_name}<small>${m.paper}</small></td>
            <td class="metric-cell">${m.psnr.toFixed(3)}</td>
            <td class="metric-cell">${m.ssim.toFixed(4)}</td>
            <td>${statusHtml}</td>
        `;
        compareTableBody.appendChild(tr);
    });

    // Build image grid
    compareGrid.innerHTML = '';

    // Add Ground Truth card
    const gtCard = document.createElement('div');
    gtCard.className = 'compare-card';
    gtCard.innerHTML = `
        <div class="compare-card-img">
            <img src="data:image/png;base64,${data.ground_truth}" alt="Original (Ground Truth)">
            <span class="compare-card-badge badge-rank">Original</span>
        </div>
        <div class="compare-card-info">
            <div class="compare-card-name">Original (Ground Truth)</div>
            <div class="compare-card-metrics">
                <span>Resolution: ${data.dimensions.original}</span>
            </div>
        </div>
    `;
    compareGrid.appendChild(gtCard);

    // Add Low Res card
    const lrCard = document.createElement('div');
    lrCard.className = 'compare-card';
    lrCard.innerHTML = `
        <div class="compare-card-img">
            <img src="data:image/png;base64,${data.low_res}" alt="Low Resolution">
            <span class="compare-card-badge badge-rank">Input</span>
        </div>
        <div class="compare-card-info">
            <div class="compare-card-name">Low Resolution (4× Downscaled)</div>
            <div class="compare-card-metrics">
                <span>Resolution: ${data.dimensions.low_res}</span>
            </div>
        </div>
    `;
    compareGrid.appendChild(lrCard);

    data.models.forEach((m, idx) => {
        const isBest = idx === 0;
        const rank = idx + 1;

        const card = document.createElement('div');
        card.className = `compare-card${isBest ? ' best-card' : ''}`;
        card.innerHTML = `
            <div class="compare-card-img">
                <img src="data:image/png;base64,${m.super_res}" alt="${m.model_name}">
                <span class="compare-card-badge ${isBest ? 'badge-best' : 'badge-rank'}">
                    ${isBest ? '★ BEST' : `#${rank}`}
                </span>
            </div>
            <div class="compare-card-info">
                <div class="compare-card-name">${m.model_name}</div>
                <div class="compare-card-metrics">
                    <span class="${isBest ? 'metric-highlight' : ''}">PSNR: ${m.psnr.toFixed(2)}</span>
                    <span class="${isBest ? 'metric-highlight' : ''}">SSIM: ${m.ssim.toFixed(4)}</span>
                </div>
            </div>
        `;
        compareGrid.appendChild(card);
    });

    setTimeout(() => showView('compare'), 400);
}


// ─── API Calls ───────────────────────────────────────────────────────────────

async function processImage(file, isSample = false) {
    const formData = new FormData();
    if (isSample) {
        formData.append('use_sample', 'true');
    } else if (file) {
        formData.append('image', file);
    } else {
        alert('Please select an image to upload.');
        return;
    }

    const modelKey = modelSelect.value;
    formData.append('model', modelKey);
    startLoading('Processing Image', singleLoadingMsgs);

    try {
        const response = await fetch('/process', { method: 'POST', body: formData });
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        displayResults(data);
    } catch (err) {
        stopLoading();
        showView('hero');
        alert(`Error processing image: ${err.message}`);
    }
}

async function compareAllModels(file, isSample = false) {
    const formData = new FormData();
    if (isSample) {
        formData.append('use_sample', 'true');
    } else if (file) {
        formData.append('image', file);
    } else {
        alert('Please select an image to upload.');
        return;
    }

    startLoading('Comparing All Models', compareLoadingMsgs);

    try {
        const response = await fetch('/compare', { method: 'POST', body: formData });
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        displayComparison(data);
    } catch (err) {
        stopLoading();
        showView('hero');
        alert(`Error comparing models: ${err.message}`);
    }
}


// ─── File Input Handler ─────────────────────────────────────────────────────
// Single shared file input — `pendingAction` determines what happens on change

function openFilePicker(action) {
    pendingAction = action; // 'single' or 'compare'
    fileInput.value = '';   // reset so same-file re-upload fires change event
    fileInput.click();
}

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        if (pendingAction === 'compare') {
            compareAllModels(file, false);
        } else {
            processImage(file, false);
        }
    }
    pendingAction = null;
});


// ─── Button Event Listeners ─────────────────────────────────────────────────

// Single model buttons
btnUploadSingle.addEventListener('click', () => openFilePicker('single'));
btnSampleSingle.addEventListener('click', () => processImage(null, true));

// Compare buttons
btnUploadCompare.addEventListener('click', () => openFilePicker('compare'));
btnSampleCompare.addEventListener('click', () => compareAllModels(null, true));

// New image buttons
btnNewImage.addEventListener('click', () => showView('hero'));
btnNewImageCompare.addEventListener('click', () => showView('hero'));


// ─── Drag and Drop (Single Model Upload Zone) ───────────────────────────────
function setupDragDrop(zone, handler) {
    zone.addEventListener('click', () => {
        if (handler === 'single') openFilePicker('single');
        else openFilePicker('compare');
    });

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (handler === 'compare') {
                compareAllModels(file, false);
            } else {
                processImage(file, false);
            }
        }
    });
}

setupDragDrop(uploadZone, 'single');
setupDragDrop(uploadZoneCompare, 'compare');


// ─── Comparison Slider ───────────────────────────────────────────────────────
let isDragging = false;

function setSliderPosition(percent) {
    percent = Math.max(0, Math.min(100, percent));
    compSR.style.clipPath = `inset(0 0 0 ${percent}%)`;
    sliderHandle.style.left = `${percent}%`;
}

function getSliderPercent(e) {
    const rect = comparisonContainer.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    return ((clientX - rect.left) / rect.width) * 100;
}

comparisonContainer.addEventListener('mousedown', (e) => {
    isDragging = true;
    setSliderPosition(getSliderPercent(e));
});

comparisonContainer.addEventListener('touchstart', (e) => {
    isDragging = true;
    setSliderPosition(getSliderPercent(e));
}, { passive: true });

document.addEventListener('mousemove', (e) => {
    if (isDragging) setSliderPosition(getSliderPercent(e));
});

document.addEventListener('touchmove', (e) => {
    if (isDragging) setSliderPosition(getSliderPercent(e));
}, { passive: true });

document.addEventListener('mouseup', () => { isDragging = false; });
document.addEventListener('touchend', () => { isDragging = false; });


// ─── Keyboard Accessibility ──────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    if (resultsSection.style.display !== 'none') {
        const currentPercent = parseFloat(sliderHandle.style.left) || 50;
        if (e.key === 'ArrowLeft') setSliderPosition(currentPercent - 2);
        else if (e.key === 'ArrowRight') setSliderPosition(currentPercent + 2);
    }
});
