import { Imagebox3 } from "https://episphere.github.io/imagebox3/imagebox3.mjs"
import { UMAP } from "https://esm.sh/umap-js"
import { createTileSource as createOSDTileSource } from "https://prafulb.github.io/WSITileSource/wsiTileSource.js"

const patchEmbed = {}

let viewer = null;
let worker = null;
let db = null;
let currentEmbeddings = [];
let currentClusters = [];
let selectedRegion = null;
let isSelecting = false;
let selectionOverlay = null;
let hnswIndex = null;
let heatmapOverlays = [];
let heatmapVisible = true;
let availableModels = [];
let selectedModel = null;
let clusterLabels = {}; // Stores mapping: { clusterIndex: "Label Text" }

document.addEventListener('DOMContentLoaded', async () => {
    await initIndexedDB()
    await loadModels();
    initViewer()
    initWorker()
    bindEvents()
    bindNewEvents()
})

function bindNewEvents() {
    document.getElementById('downloadEmbeddings').addEventListener('click', () => {
        downloadEmbeddings()
    })
}


async function initIndexedDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('WSIEmbeddings', 1)

        request.onerror = () => reject(request.error)
        request.onsuccess = () => {
            db = request.result
            resolve()
        }

        request.onupgradeneeded = (event) => {
            const db = event.target.result
            if (!db.objectStoreNames.contains('embeddings')) {
                const store = db.createObjectStore('embeddings', { keyPath: 'patchNum', autoIncrement: true })
                store.createIndex('imageId', 'imageId', { unique: false })
                store.createIndex('imageId_x_y', ['imageId', 'topLeftX', 'topLeftY'], { unique: false })
            }
            if (!db.objectStoreNames.contains('indices')) {
                db.createObjectStore('indices', { keyPath: 'imageId' })
            }
        }
    })
}

function initViewer() {
    viewer = OpenSeadragon({
        id: 'viewer',
        prefixUrl: 'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/',
        showNavigationControl: true,
        showZoomControl: true,
        showHomeControl: true,
        showFullPageControl: false,
        immediateRender: true,
        gestureSettingsMouse: {
            clickToZoom: false,
            dblClickToZoom: true
        }
    })

    viewer.addHandler('canvas-click', handleViewerClick)
    viewer.addHandler('canvas-drag', handleViewerDrag)
    viewer.addHandler('canvas-drag-end', handleViewerDragEnd)
}

let workerPool = [];
let availableWorkers = []; // Stores worker IDs

function initWorker(numWorkers = 4) {
    const baseURL = import.meta.url.split("/").slice(0, -1).join("/")

    // Clear existing pool if any
    workerPool.forEach(w => w.terminate());
    workerPool = [];
    availableWorkers = [];

    for (let i = 0; i < numWorkers; i++) {
        const worker = new Worker(URL.createObjectURL(new Blob([`importScripts("${baseURL}/worker.js")`])))
        const workerId = i;

        worker.onmessage = (e) => handleWorkerMessage(e, workerId);
        worker.onerror = (error) => {
            console.error(`Worker ${workerId} error:`, error)
        }

        workerPool.push(worker);
        availableWorkers.push(workerId);
    }
}

async function handleWorkerMessage(e, workerId) {
    const { success, data, error, final, patchParams } = e.data

    removeViewerOverlay(workerId);

    // Return worker to pool before processing results
    availableWorkers.push(workerId);
    workerQueue.process(); // Signal queue to check for pending tasks

    // Always resolve the queue task if we have params
    if (patchParams) {
        workerQueue.resolveTask(patchParams);
    }

    const precision = document.getElementById('precisionSelect').value;
    const storageType = document.getElementById('storageSelect').value;

    if (success) {
        updateEmbeddingCount()

        // Build HNSW index (placeholder for now)
        // await buildHNSWIndex(data)

        updateStatus('Moving to next patch...')
    } else {
        console.error(`Worker ${workerId} error:`, error)
        updateStatus('Error generating embeddings')
    }
}

async function setupImageBox3Instance(input) {
    if (!patchEmbed.imagebox3Instance) {
        // const numWorkers = Math.floor(navigator.hardwareConcurrency / 2)
        const numWorkers = 1
        patchEmbed.imagebox3Instance = new Imagebox3(input, numWorkers)
        await patchEmbed.imagebox3Instance.init()
    }
    else if (patchEmbed.imagebox3Instance.getImageSource()) {
        await patchEmbed.imagebox3Instance.changeImageSource(input)
    }
}

async function createTileSource(input) {
    if (!patchEmbed.imagebox3Instance || patchEmbed.imagebox3Instance.getImageSource() !== input) {
        await setupImageBox3Instance(input)
    }

    const numWorkers = 4

    let tileSources = {}
    try {
        tileSources = await createOSDTileSource(input, numWorkers)
        // tileSources = await OpenSeadragon.GeoTIFFTileSource.getAllTileSources(input, { logLatency: false, cache: true, slideOnly: true, pool: viewer.world?.getItemAt(0)?.source._pool });
        await new Promise(res => setTimeout(res, 1000)) // Wait for decoders to finish setting up
    }
    catch (e) {
        console.error(e)
        alert("An error occurred while loading the image. Please check the web browser's Console for more information.")
        return undefined
    }
    return tileSources
}

async function buildHNSWIndex(embeddings) {
    const indexData = {
        imageId: 'current',
        embeddings: embeddings.map(e => e.embedding),
        metadata: embeddings.map(e => ({
            x: e.patchTopLeftX,
            y: e.patchTopLeftY,
            w: e.width,
            h: e.height
        }))
    }

    const transaction = db.transaction(['indices'], 'readwrite')
    const store = transaction.objectStore('indices')
    await new Promise((resolve, reject) => {
        const request = store.put(indexData)
        request.onsuccess = () => resolve()
        request.onerror = () => reject(request.error)
    })
}


async function runUMAP(vectors) {
    if (!patchEmbed.umapInstance) {
        patchEmbed.umapInstance = new UMAP({
            nComponents: 3,
            nNeighbors: 15,
            minDist: 0.1,
            seed: 42
        })
        patchEmbed.umapInstance.fit(vectors)
    }
    return patchEmbed.umapInstance.transform(vectors)
}

export async function retrieveEmbeddings(image = patchEmbed.imagebox3Instance?.getImageSource(), lowerBound = [0, 0], upperBound = [Infinity, Infinity]) {
    let imageSource = image
    if (imageSource instanceof File) {
        imageSource = imageSource.name
    }
    const objectStore = db.transaction("embeddings", "readonly").objectStore("embeddings").index("imageId_x_y")

    return new Promise((resolve, reject) => {

        if (!imageSource) {
            objectStore.getAll().onsuccess = (e) => {
                resolve({ result: e.target.result })
            }
        } else {
            if (!lowerBound || !Array.isArray(lowerBound) || !upperBound || !Array.isArray(upperBound)) {
                reject("Malformed query")
            }
            let queryResult = []

            const cursorSource = objectStore
            const range = IDBKeyRange.bound([imageSource, ...lowerBound], [imageSource, ...upperBound], true, true)

            const cursorRequest = cursorSource.openCursor(range)
            cursorRequest.onsuccess = (e) => {
                const cursor = e.target.result
                if (cursor) {
                    // console.log(`No cursor, found ${queryResult.length} items for query`, queryOpts)
                    queryResult.push(cursor.value)
                    cursor.continue()
                } else {
                    resolve({ result: queryResult })
                }
            }
            cursorRequest.onerror = (e) => {
                console.log(e)
                reject(e)
            }
        }
    })
}

// Load available models from config
async function loadModels() {
    try {
        // For demo purposes, using hardcoded models
        // In production, fetch from config.json
        availableModels = SUPPORTED_MODELS

        // Uncomment this line to load from config.json in production:
        // const response = await fetch('config.json');
        // availableModels = await response.json();

        populateModelDropdown();
    } catch (error) {
        console.error('Failed to load models:', error);
        updateStatus('Failed to load model configurations');
    }
}

// Populate model dropdown
function populateModelDropdown() {
    const select = document.getElementById('modelSelect');
    select.innerHTML = '<option value="">Select a model...</option>';

    availableModels.filter(model => model.enabled).forEach(model => {
        const option = document.createElement('option');
        option.value = model.modelName;
        option.textContent = `${model.modelName} (${model.embeddingDimension}D)`;
        select.appendChild(option);
    });

    // Auto-select first model
    if (availableModels.length > 0) {
        select.value = availableModels[0].modelName;
        selectedModel = availableModels[0];
    }
}

// K-means clustering implementation
// function kMeansCluster(embeddings, k) {
//     const points = embeddings.result.map(e => e.embedding);
//     const n = points.length;
//     const dim = points[0].length;

//     // Initialize centroids randomly
//     let centroids = [];
//     for (let i = 0; i < k; i++) {
//         centroids.push(points[Math.floor(Math.random() * n)].slice());
//     }

//     let assignments = new Array(n);
//     let changed = true;
//     let iterations = 0;
//     const maxIterations = 100;

//     while (changed && iterations < maxIterations) {
//         changed = false;
//         iterations++;

//         // Assign points to nearest centroid
//         for (let i = 0; i < n; i++) {
//             let bestCluster = 0;
//             let bestDistance = Infinity;

//             for (let j = 0; j < k; j++) {
//                 const distance = euclideanDistance(points[i], centroids[j]);
//                 if (distance < bestDistance) {
//                     bestDistance = distance;
//                     bestCluster = j;
//                 }
//             }

//             if (assignments[i] !== bestCluster) {
//                 assignments[i] = bestCluster;
//                 changed = true;
//             }
//         }

//         // Update centroids
//         for (let j = 0; j < k; j++) {
//             const clusterPoints = [];
//             for (let i = 0; i < n; i++) {
//                 if (assignments[i] === j) {
//                     clusterPoints.push(points[i]);
//                 }
//             }

//             if (clusterPoints.length > 0) {
//                 for (let d = 0; d < dim; d++) {
//                     centroids[j][d] = clusterPoints.reduce((sum, p) => sum + p[d], 0) / clusterPoints.length;
//                 }
//             }
//         }
//     }

//     return assignments;
// }

function performKMeans(embeddingsFlat, numEmbeddings, dim, k) {
    // embeddingsFlat is a single Float32Array containing all vectors concatenated

    // Initialize Centroids (Randomly pick k embeddings)
    const centroids = new Float32Array(k * dim);
    for (let i = 0; i < k; i++) {
        const idx = Math.floor(Math.random() * numEmbeddings);
        const start = idx * dim;
        centroids.set(embeddingsFlat.subarray(start, start + dim), i * dim);
    }

    const assignments = new Int32Array(numEmbeddings);
    const iterations = 10; // Lower iterations for speed, usually converges fast enough
    for (let iter = 0; iter < iterations; iter++) {
        let changed = false;
        const clusterCounts = new Float32Array(k).fill(0);
        const newCentroids = new Float32Array(k * dim).fill(0);

        // Assignment Step
        for (let i = 0; i < numEmbeddings; i++) {
            let minDist = Infinity;
            let bestCluster = -1;
            const vecStart = i * dim;

            for (let c = 0; c < k; c++) {
                let dist = 0;
                const centerStart = c * dim;
                for (let d = 0; d < dim; d++) {
                    const val = embeddingsFlat[vecStart + d] - centroids[centerStart + d];
                    dist += val * val;
                }
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = c;
                }
            }
            if (assignments[i] !== bestCluster) changed = true;
            assignments[i] = bestCluster;

            // Accumulate for update step
            clusterCounts[bestCluster]++;
            const bestCenterStart = bestCluster * dim;
            for (let d = 0; d < dim; d++) {
                newCentroids[bestCenterStart + d] += embeddingsFlat[vecStart + d];
            }
        }

        // Update Step
        for (let c = 0; c < k; c++) {
            if (clusterCounts[c] > 0) {
                const centerStart = c * dim;
                for (let d = 0; d < dim; d++) {
                    centroids[centerStart + d] = newCentroids[centerStart + d] / clusterCounts[c];
                }
            }
        }

        if (!changed) break;
    }
    return assignments;
}

function euclideanDistance(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += (a[i] - b[i]) ** 2;
    }
    return Math.sqrt(sum);
}

// Run clustering and create heatmap
async function runClusteringAndHeatmap(embeddings) {
    updateStatus('Running clustering...');

    const method = document.getElementById('clusteringMethod').value;
    // Note: I changed the HTML ID to numClustersInput to avoid conflict
    const numClustersInput = document.getElementById('numClustersInput') || document.getElementById('numClusters');
    const numClusters = parseInt(numClustersInput.value);

    // Reset labels if cluster count changed (optional logic)
    if (Object.keys(clusterLabels).length !== numClusters) {
        clusterLabels = {};
    }

    let clusterAssignments;
    const rawData = embeddings.result;
    if (rawData.length === 0) return;

    const firstEmb = rawData[0].embedding;
    const isBinary = firstEmb instanceof BigUint64Array;
    const dim = isBinary ? firstEmb.length * 64 : firstEmb.length;
    const numEmbeddings = rawData.length;
    const flatEmbeddings = new Float32Array(numEmbeddings * dim);

    rawData.forEach((item, i) => {
        const emb = item.embedding;
        if (emb instanceof BigUint64Array) {
            // Unpack BigInt bits to floats (0.0 or 1.0) for K-means
            for (let j = 0; j < emb.length; j++) {
                const bigVal = emb[j];
                for (let bit = 0; bit < 64; bit++) {
                    flatEmbeddings[i * dim + j * 64 + bit] = (bigVal & (1n << BigInt(bit))) ? 1.0 : 0.0;
                }
            }
        } else {
            flatEmbeddings.set(emb, i * dim);
        }
    });

    // ... (Existing switch/case logic for clustering remains exactly the same) ...
    switch (method) {
        case 'kmeans':
            clusterAssignments = performKMeans(flatEmbeddings, numEmbeddings, dim, numClusters);
            break;
        // ... other cases ...
        default:
            clusterAssignments = performKMeans(flatEmbeddings, numEmbeddings, dim, numClusters);
    }

    // Create cluster data
    currentClusters = embeddings.result.map((embedding, i) => ({
        ...embedding,
        cluster: clusterAssignments[i]
    }));

    // Create heatmap overlay
    createHeatmapOverlay(currentClusters, numClusters);

    // *** NEW: Render the annotation UI ***
    renderAnnotationUI(numClusters);

    // updateClusterCount(numClusters);
    updateStatus('Clustering complete');
}

// Create heatmap overlay on the viewer
let heatmapCanvas = null;

function drawHeatmap(clusters, highlightedClusterIndex = null, selectedPatches = null) {
    if (!heatmapCanvas) return;
    const ctx = heatmapCanvas.getContext('2d');
    const world = viewer.world.getItemAt(0);
    if (!world) return;
    const imageSize = world.getContentSize();
    const scale = heatmapCanvas.width / imageSize.x;
    ctx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);

    const numClusters = Math.max(...clusters.map(c => c.cluster)) + 1;
    const colors = generateClusterColors(numClusters);

    clusters.forEach(patch => {
        const color = colors[patch.cluster];
        let opacity = 0.5;

        if (highlightedClusterIndex !== null) {
            opacity = (patch.cluster === highlightedClusterIndex) ? 0.9 : 0.1;
        } else if (selectedPatches && selectedPatches.length > 0) {
            const isSelected = selectedPatches.some(p => p.topLeftX === patch.topLeftX && p.topLeftY === patch.topLeftY);
            opacity = isSelected ? 1.0 : 0.3;
        }

        ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${opacity})`;
        ctx.fillRect(
            patch.topLeftX * scale,
            patch.topLeftY * scale,
            patch.width * scale,
            patch.height * scale
        );

        if (selectedPatches && selectedPatches.some(p => p.topLeftX === patch.topLeftX && p.topLeftY === patch.topLeftY)) {
            ctx.strokeStyle = '#FBBF24';
            ctx.lineWidth = 2;
            ctx.strokeRect(
                patch.topLeftX * scale,
                patch.topLeftY * scale,
                patch.width * scale,
                patch.height * scale
            );
        }
    });
}

async function createHeatmapOverlay(clusters, numClusters) {
    clearHeatmapOverlays();

    const world = viewer.world.getItemAt(0);
    if (!world) return;
    const imageSize = world.getContentSize();

    heatmapCanvas = document.createElement('canvas');
    heatmapCanvas.id = 'heatmap-canvas';
    const scale = Math.min(1.0, 4096 / Math.max(imageSize.x, imageSize.y));
    heatmapCanvas.width = imageSize.x * scale;
    heatmapCanvas.height = imageSize.y * scale;
    heatmapCanvas.style.pointerEvents = 'none';

    drawHeatmap(clusters);

    const fullImageRect = viewer.viewport.imageToViewportRectangle(0, 0, imageSize.x, imageSize.y);
    viewer.addOverlay(heatmapCanvas, fullImageRect);
}

function clearHeatmapOverlays() {
    if (heatmapCanvas) {
        viewer.removeOverlay(heatmapCanvas);
        heatmapCanvas = null;
    }
    // Backward compatibility for any remaining heatmapOverlays logic
    if (typeof heatmapOverlays !== 'undefined' && Array.isArray(heatmapOverlays)) {
        heatmapOverlays.forEach(o => viewer.removeOverlay(o));
        heatmapOverlays = [];
    }
}

// Generate distinct colors for clusters
function generateClusterColors(numClusters) {
    const colors = [];
    for (let i = 0; i < numClusters; i++) {
        const hue = (i * 360) / numClusters;
        const rgb = hslToRgb(hue / 360, 0.7, 0.5);
        colors.push({ r: rgb[0], g: rgb[1], b: rgb[2], a: 0.5 });
    }
    return colors;
}

function hslToRgb(h, s, l) {
    let r, g, b;
    if (s === 0) {
        r = g = b = l;
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
        };
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// Clear heatmap overlays
// function clearHeatmapOverlays() {
//     heatmapOverlays.forEach(overlay => {
//         viewer.removeOverlay(overlay);
//     });
//     heatmapOverlays = [];
// }

// Toggle heatmap visibility
function toggleHeatmap() {
    heatmapVisible = !heatmapVisible;
    const button = document.getElementById('toggleHeatmap');

    heatmapCanvas.style.display = heatmapVisible ? 'block' : 'none';

    button.textContent = heatmapVisible ? 'Hide Heatmap' : 'Show Heatmap';
    button.className = heatmapVisible ?
        'bg-indigo-600 hover:bg-indigo-700 px-3 py-1 rounded text-sm' :
        'bg-gray-600 hover:bg-gray-700 px-3 py-1 rounded text-sm';
}

async function runUMAPAndPlot() {
    updateStatus('Running UMAP...')

    const umapResults = await runUMAP(embeddings, 3)

    const x = umapResults.map(p => p[0])
    const y = umapResults.map(p => p[1])
    const z = umapResults.map(p => p[2])

    const text = embeddings.map((e, i) =>
        `Patch ${i}<br>X: ${e.patchTopLeftX}<br>Y: ${e.patchTopLeftY}<br>Size: ${e.width}x${e.height}`
    )

    const plotData = [{
        x: x,
        y: y,
        z: z,
        mode: 'markers',
        type: 'scatter3d',
        text: text,
        hovertemplate: '%{text}<extra></extra>',
        marker: {
            size: 5,
            color: z,
            colorscale: 'Viridis',
            showscale: true
        }
    }]

    const layout = {
        title: '3D UMAP Visualization',
        scene: {
            xaxis: { title: 'UMAP 1' },
            yaxis: { title: 'UMAP 2' },
            zaxis: { title: 'UMAP 3' }
        },
        paper_bgcolor: 'rgba(55, 65, 81, 1)',
        plot_bgcolor: 'rgba(55, 65, 81, 1)',
        font: { color: 'white' }
    }

    const config = {
        displayModeBar: true,
        responsive: true
    }

    Plotly.newPlot('plot', plotData, layout, config)

    document.getElementById('plot').on('plotly_click', (data) => {
        const pointIndex = data.points[0].pointIndex
        const embedding = embeddings[pointIndex]
        showTile(embedding.patchTopLeftX, embedding.patchTopLeftY,
            embedding.width, embedding.height)
    })

    currentPlotData = { data: plotData, embeddings: embeddings }
}

function showTile(topLeftX, topLeftY, width, height) {
    if (!viewer) return

    const rect = new OpenSeadragon.Rect(topLeftX, topLeftY, width, height)
    viewer.viewport.fitBounds(rect, true)

    highlightRegion(topLeftX, topLeftY, width, height)
}

function highlightRegion(x, y, width, height) {
    viewer.clearOverlays()

    const element = document.createElement('div')
    element.style.border = '3px solid #10B981'
    element.style.backgroundColor = 'rgba(16, 185, 129, 0.2)'
    element.style.pointerEvents = 'none'

    viewer.addOverlay(element, new OpenSeadragon.Rect(x, y, width, height))

    setTimeout(() => {
        viewer.removeOverlay(element)
    }, 3000)
}

function handleViewerClick(event) {
    if (!isSelecting) return

    const webPoint = event.position
    const viewportPoint = viewer.viewport.pointFromPixel(webPoint)

    if (!selectedRegion) {
        selectedRegion = {
            startX: viewportPoint.x,
            startY: viewportPoint.y
        }
    }
}

function handleViewerDrag(event) {
    if (!isSelecting || !selectedRegion) return

    const webPoint = event.position
    const viewportPoint = viewer.viewport.pointFromPixel(webPoint)

    updateSelectionOverlay(selectedRegion.startX, selectedRegion.startY,
        viewportPoint.x, viewportPoint.y)
}

function handleViewerDragEnd(event) {
    if (!isSelecting || !selectedRegion) return

    const webPoint = event.position
    const viewportPoint = viewer.viewport.pointFromPixel(webPoint)

    selectedRegion.endX = viewportPoint.x
    selectedRegion.endY = viewportPoint.y

    processSelectedRegion()

    isSelecting = false
}

function updateSelectionOverlay(startX, startY, endX, endY) {
    if (selectionOverlay) {
        viewer.removeOverlay(selectionOverlay)
    }

    const element = document.createElement('div')
    element.style.border = '2px dashed #3B82F6'
    element.style.backgroundColor = 'rgba(59, 130, 246, 0.1)'
    element.style.pointerEvents = 'none'

    const rect = new OpenSeadragon.Rect(
        Math.min(startX, endX),
        Math.min(startY, endY),
        Math.abs(endX - startX),
        Math.abs(endY - startY)
    )

    viewer.addOverlay(element, rect)
    selectionOverlay = element
}

// Process selected region
async function processSelectedRegion() {
    if (!selectedRegion || !currentClusters.length) return;

    const minX = Math.min(selectedRegion.startX, selectedRegion.endX);
    const maxX = Math.max(selectedRegion.startX, selectedRegion.endX);
    const minY = Math.min(selectedRegion.startY, selectedRegion.endY);
    const maxY = Math.max(selectedRegion.startY, selectedRegion.endY);

    // Find patches in selected region
    const selectedPatches = currentClusters.filter(patch => {
        const patchCenterX = patch.patchTopLeftX + patch.width / 2;
        const patchCenterY = patch.patchTopLeftY + patch.height / 2;
        return patchCenterX >= minX && patchCenterX <= maxX &&
            patchCenterY >= minY && patchCenterY <= maxY;
    });

    if (selectedPatches.length === 0) return;

    // Highlight selected patches in the heatmap
    highlightSelectedPatches(selectedPatches);

    // Find similar patches
    await findSimilarPatches(selectedPatches);
}

// async function processSelectedRegion() {
//     if (!selectedRegion || !currentEmbeddings.length) return

//     const minX = Math.min(selectedRegion.startX, selectedRegion.endX)
//     const maxX = Math.max(selectedRegion.startX, selectedRegion.endX)
//     const minY = Math.min(selectedRegion.startY, selectedRegion.endY)
//     const maxY = Math.max(selectedRegion.startY, selectedRegion.endY)

//     const selectedPatches = currentEmbeddings.filter(patch => {
//         const patchCenterX = patch.patchTopLeftX + patch.width / 2
//         const patchCenterY = patch.patchTopLeftY + patch.height / 2
//         return patchCenterX >= minX && patchCenterX <= maxX &&
//             patchCenterY >= minY && patchCenterY <= maxY
//     })

//     if (selectedPatches.length === 0) return

//     highlightPlotPoints(selectedPatches)

//     await findSimilarPatches(selectedPatches)
// }

// function highlightPlotPoints(patches) {
//     if (!currentPlotData) return

//     const highlightIndices = patches.map(patch =>
//         currentEmbeddings.findIndex(e => e === patch)
//     ).filter(index => index !== -1)

//     const colors = currentPlotData.data[0].marker.color.map((_, i) =>
//         highlightIndices.includes(i) ? 'red' : 'blue'
//     )

//     Plotly.restyle('plot', { 'marker.color': [colors] }, [0])
// }

// Hamming distance for binary vectors
function hammingDistance(vectorA, vectorB) {
    if (vectorA.length !== vectorB.length) {
        throw new Error("Vectors must be of the same length");
    }
    let distance = 0;
    for (let i = 0; i < vectorA.length; i++) {
        const xorResult = vectorA[i] ^ vectorB[i];
        // Popcount equivalent for BigInt
        distance += xorResult.toString(2).replace(/0/g, "").length;
    }
    return distance;
}

// Download embeddings as JSON
async function downloadEmbeddings() {
    const imageSource = patchEmbed.imagebox3Instance?.getImageSource();
    if (!imageSource) {
        alert("Please load an image first.");
        return;
    }
    let imageId = imageSource;
    if (imageId instanceof File) imageId = imageId.name;

    const storageType = document.getElementById('storageSelect').value;
    let data = [];

    // Helper to format a single embedding object
    const formatEmbedding = (patch, embeddingVector) => {
        const patchSize = 224; // Assuming 224 for now, ideally strictly from params
        const x = patch.topLeftX
        const y = patch.topLeftY
        const w = patch.width
        const h = patch.height
        const precision = patch.precision
        // Find cluster and annotation
        // We need to match this patch with the currentClusters to get the cluster ID
        // unique key: x-y
        const clusterMatch = currentClusters.find(c =>
            c.topLeftX === x && c.topLeftY === y
        );

        let annotation = "Unknown";
        if (clusterMatch && clusterMatch.cluster !== undefined) {
            annotation = clusterLabels[clusterMatch.cluster] || undefined
        }
        const usp = new URLSearchParams(document.location.search);
        const annotationKey = usp.get("gleason") ? "gleason_score" : "annotation";
        const id = `${imageId}_${x}_${y}_${w}_${h}_${patchSize}`;
        const returnObj = {
            "id": id,
            "wsiId": imageId,
            "wsiURL": imageId, // processing URL same as ID for now if file
            "tileParams": {
                "tileX": x,
                "tileY": y,
                "tileWidth": w,
                "tileHeight": h,
                "tileSize": w
            },
            "model": selectedModel ? selectedModel.modelName : "Unknown",
            "embedding": Array.from(embeddingVector), // Ensure plain array
            "properties": {}
        };
        if (annotation) {
            returnObj.properties[annotationKey] = annotation;
        }
        return returnObj;
    };

    if (storageType === 'indexeddb') {
        const result = await retrieveEmbeddings(imageId);
        // result.result is an array of objects from IDB
        // Each object has: imageId, patchTopLeftX, patchTopLeftY, width, height, embedding (Float32Array), etc.
        data = result.result.map(item => formatEmbedding(item, item.embedding));

    } else {
        // Retrieve from OPFS
        try {
            // NOTE: OPFS retrieval logic currently retrieves specific binary file.
            // Converting that back to structured JSON with coordinates requires reading metadata
            // which we might not have stored fully in OPFS in previous steps (only binary data).
            // However, the previous code just exported a metadata object about the binary file.
            // If we want FULL JSON export from OPFS, we need to know coordinate order which implies 
            // we should probably trust IndexedDB for the JSON export or we need a metadata file in OPFS.
            // Given the complexity, I will pull from RetrieveEmbeddings logic which reads from IDB for metadata usually?
            // Wait, previous code for OPFS read `embeddings_${precision}.bin`. 
            // If the user wants granular JSON export, they likely need the metadata which is in IDB.
            // Let's assume for JSON export we ALWAYS use data from memory/IDB if available, 
            // or we warn if we only have raw binary. 
            // For now, I will fallback to the same logic as IDB if possible, looking at `retrieveEmbeddings`.
            // `retrieveEmbeddings` pulls from IDB. 
            // If storage was OPFS, do we have metadata in IDB? 
            // worker.js: storeInOPFS stores binary. storeEmbeddings (IDB) stores metadata + embedding.
            // If storageType is OPFS, worker ONLY calls storeInOPFS? 
            // Checking worker.js... 
            // `if (storageType === 'indexeddb') { ... } else if (storageType === 'opfs') { storeInOPFS ... }`
            // So if OPFS is used, IDB is EMPTY. 
            // This means we CANNOT easily generate the requested JSON (with coordinates) from OPFS 
            // without a separate metadata file. 
            // For now, I will keep the OPFS export as "Binary Blob Info" OR alert user not supported for JSON export yet.
            // BUT, the user request implies they want this structure. 

            // Re-reading worker.js: storeInOPFS only writes binary buffer. No coordinates.
            // This means we CANNOT satisfy the user request for OPFS storage type currently.
            // I will add a check/alert for now, or fallback to simple export if strictly necessary.
            // Given the instruction "in downloadEmbeddings...", I will focus on the structure.
            // If data is available in memory (currentClusters), we could use that?
            // `currentEmbeddings` or `currentClusters` might hold the data if recently generated.
            // `currentClusters` is populated in `runClusteringAndHeatmap` which takes `embeddings`.
            // If we have `currentClusters` in memory, we can use it!

            if (currentClusters && currentClusters.length > 0) {
                data = currentClusters.map(item => formatEmbedding(item, item.embedding));
            } else {
                alert("For OPFS storage, please generate embeddings/clusters first to have them in memory for JSON export, as metadata is not persisted in OPFS yet.");
                return;
            }

        } catch (e) {
            console.error(e);
            alert("Error preparing export from OPFS/Memory.");
            return;
        }
    }

    const blob = new Blob([JSON.stringify(data, (key, value) =>
        typeof value === 'bigint' ? value.toString() : value, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${imageId}_embeddings.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// Highlight selected patches in the heatmap
function highlightSelectedPatches(patches) {
    drawHeatmap(currentClusters, null, patches);
}

// Find similar patches using simple similarity
async function findSimilarPatches(selectedPatches) {
    if (selectedPatches.length === 0) return;

    const precision = document.getElementById('precisionSelect').value;

    let similarities;

    if (precision === 'binary') {
        // Use Hamming distance for binary
        const queryVector = selectedPatches[0].embedding; // Using the first selected patch as query
        similarities = currentClusters.map(patch => {
            const distance = hammingDistance(queryVector, patch.embedding);
            const totalBits = queryVector.length * 64;
            return 1 - (distance / totalBits); // Normalize to 0..1 similarity
        });
    } else {
        // Calculate average embedding of selected patches (only for non-binary)
        const avgEmbedding = new Array(selectedPatches[0].embedding.length).fill(0);
        selectedPatches.forEach(patch => {
            patch.embedding.forEach((val, i) => {
                avgEmbedding[i] += val / selectedPatches.length;
            });
        });

        // Find similar patches using cosine similarity
        similarities = currentClusters.map(patch => {
            let dotProduct = 0, normA = 0, normB = 0;
            patch.embedding.forEach((val, i) => {
                dotProduct += val * avgEmbedding[i];
                normA += val * val;
                normB += avgEmbedding[i] * avgEmbedding[i];
            });
            return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
        });
    }

    // Get top similar patches
    const threshold = 0.7;
    const similarPatches = currentClusters.filter((_, i) => similarities[i] > threshold);

    // Highlight similar regions on viewer
    similarPatches.forEach(patch => {
        const patchIndex = currentClusters.findIndex(c => c === patch);
        if (patchIndex >= 0 && heatmapOverlays[patchIndex]) {
            heatmapOverlays[patchIndex].style.border = '3px solid #10B981';
            heatmapOverlays[patchIndex].style.opacity = '1.0';

            setTimeout(() => {
                const colors = generateClusterColors(Math.max(...currentClusters.map(c => c.cluster)) + 1);
                const color = colors[patch.cluster];
                heatmapOverlays[patchIndex].style.border = `1px solid rgba(${color.r}, ${color.g}, ${color.b}, 0.8)`;
                heatmapOverlays[patchIndex].style.opacity = '0.6';
            }, 3000);
        }
    });
}

// async function findSimilarPatches(selectedPatches) {
//     if (selectedPatches.length === 0) return

//     const avgEmbedding = new Array(selectedPatches[0].embedding.length).fill(0)
//     selectedPatches.forEach(patch => {
//         patch.embedding.forEach((val, i) => {
//             avgEmbedding[i] += val / selectedPatches.length
//         })
//     })

//     const similarities = currentEmbeddings.map(patch => {
//         let dotProduct = 0, normA = 0, normB = 0
//         patch.embedding.forEach((val, i) => {
//             dotProduct += val * avgEmbedding[i]
//             normA += val * val
//             normB += avgEmbedding[i] * avgEmbedding[i]
//         })
//         return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
//     })

//     const threshold = 0.7
//     const similarPatches = currentEmbeddings.filter((_, i) => similarities[i] > threshold)

//     similarPatches.forEach(patch => {
//         const element = document.createElement('div')
//         element.style.border = '2px solid #F59E0B'
//         element.style.backgroundColor = 'rgba(245, 158, 11, 0.3)'
//         element.style.pointerEvents = 'none'

//         const rect = new OpenSeadragon.Rect(
//             patch.patchTopLeftX, patch.patchTopLeftY,
//             patch.width, patch.height
//         )

//         viewer.addOverlay(element, rect)

//         setTimeout(() => {
//             viewer.removeOverlay(element)
//         }, 5000)
//     })
// }

function bindEvents() {
    document.getElementById('loadImage').addEventListener('click', loadImage)
    document.getElementById('generateEmbeddings').addEventListener('click', generateEmbeddingsHandler)
    document.getElementById('selectRegion').addEventListener('click', toggleSelection)
    document.getElementById('loadImage').addEventListener('click', loadImage);
    document.getElementById('clearSelection').addEventListener('click', clearSelection);
    document.getElementById('toggleHeatmap').addEventListener('click', toggleHeatmap);
    document.getElementById('browseFile').addEventListener('click', () => {
        document.getElementById('localFile').click();
    });
    document.getElementById('localFile').addEventListener('change', handleFileSelect);
    document.getElementById('modelSelect').addEventListener('change', handleModelChange);
    document.getElementById('clusteringMethod').addEventListener('change', updateClustering);
    // document.getElementById('numClusters').addEventListener('input', updateClustering);
    // In bindEvents()
    const numClustersEl = document.getElementById('numClustersInput');
    if (numClustersEl) {
        numClustersEl.addEventListener('change', updateClustering);
        numClustersEl.addEventListener('input', updateClustering); // Optional: if you want real-time update while typing
    }
    // Event listener for exportAnnotations removed

    // Add drag and drop functionality
    const urlInput = document.getElementById('imageUrl');
    urlInput.addEventListener('dragover', (e) => {
        e.preventDefault();
        urlInput.style.borderColor = '#3B82F6';
    });
    urlInput.addEventListener('dragleave', (e) => {
        urlInput.style.borderColor = '#6B7280';
    });
    urlInput.addEventListener('drop', (e) => {
        e.preventDefault();
        urlInput.style.borderColor = '#6B7280';
        if (e.dataTransfer.files.length > 0) {
            document.getElementById('localFile').files = e.dataTransfer.files;
            handleFileSelect({ target: { files: e.dataTransfer.files } });
        }
    });
}

async function loadImage() {
    const url = document.getElementById('imageUrl').value
    const file = document.getElementById('localFile').files[0]

    let input = null
    if (url) {
        input = url
    } else if (file) {
        input = file
    } else {
        alert('Please provide a URL or select a file')
        return
    }

    await setupImageBox3Instance(input)
    const tileSource = await createTileSource(input)
    if (tileSource) {
        viewer.open(tileSource)
        updateStatus('Image loaded')
        document.getElementById("generateEmbeddings").removeAttribute('disabled')
        await new Promise(res => setTimeout(res, 5000))
        const allEmbeddings = await retrieveEmbeddings()
        if (allEmbeddings.result.length > 0) {
            await runClusteringAndHeatmap(allEmbeddings);
        }
    } else {
        updateStatus('Failed to load image')
        document.getElementById("generateEmbeddings").setAttribute('disabled', true)
    }
}

async function getTissueRegions(cellWidth = 224 * 8, cellHeight = 224 * 8) {
    if (!patchEmbed.imagebox3Instance) return
    console.time("thumbnail")
    const imageInfo = await patchEmbed.imagebox3Instance.getInfo()
    const { width: imageWidth, height: imageHeight } = imageInfo
    const thumbnailBlob = await patchEmbed.imagebox3Instance.getThumbnail(512, 366)
    const thumbnailURL = URL.createObjectURL(thumbnailBlob)
    console.timeEnd("thumbnail")
    const thumbnailImg = new Image()

    return new Promise((resolve) => {
        thumbnailImg.onload = () => {
            const thumbnailWidth = thumbnailImg.naturalWidth
            const thumbnailHeight = thumbnailImg.naturalHeight

            const tissueRegions = [];
            const offscreenCanvas = new OffscreenCanvas(32, 32);
            const offscreenCtx = offscreenCanvas.getContext("2d");

            for (let y = 0; y < imageHeight; y += cellHeight) {
                for (let x = 0; x < imageWidth; x += cellWidth) {
                    const w = Math.min(cellWidth, imageWidth - x);
                    const h = Math.min(cellHeight, imageHeight - y);

                    // Project to thumbnail
                    const thumbX = (x / imageWidth) * thumbnailWidth;
                    const thumbY = (y / imageHeight) * thumbnailHeight;
                    const thumbW = (w / imageWidth) * thumbnailWidth;
                    const thumbH = (h / imageHeight) * thumbnailHeight;

                    offscreenCtx.clearRect(0, 0, offscreenCanvas.width, offscreenCanvas.height);
                    offscreenCtx.drawImage(
                        thumbnailImg,
                        thumbX, thumbY, thumbW, thumbH,
                        0, 0, offscreenCanvas.width, offscreenCanvas.height
                    );

                    const tileContent = isTileEmpty(offscreenCanvas, offscreenCtx, 0.95, true);

                    if (!tileContent.isEmpty) {
                        tissueRegions.push({
                            topLeftX: x,
                            topLeftY: y,
                            width: w,
                            height: h,
                            ...tileContent
                        });
                    }
                }
            }

            resolve(tissueRegions)
            URL.revokeObjectURL(thumbnailURL)
        }
        thumbnailImg.src = thumbnailURL
    })
}

const isTileEmpty = (canvas, ctx, threshold = 0.95, returnEmptyProportion = false) => {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    const pixels = imageData.data
    const numPixels = pixels.length / 4

    let whitePixelCount = 0

    for (let i = 0; i < pixels.length; i += 4) {
        const r = pixels[i]
        const g = pixels[i + 1]
        const b = pixels[i + 2]

        if (r > 200 && g > 200 && b > 200) {
            whitePixelCount++
        }
    }

    const whiteProportion = whitePixelCount / numPixels
    let isEmpty = false
    if (whiteProportion >= threshold) {
        isEmpty = true
    }
    const returnObj = { isEmpty }
    if (returnEmptyProportion) {
        returnObj["emptyProportion"] = whiteProportion
    }
    return returnObj
}

const getImageTile = async (tileParams) => {
    const tileURL = URL.createObjectURL(
        await patchEmbed.imagebox3Instance.getTile(...Object.values(tileParams))
    )
    return tileURL
}

const addViewerOverlay = (tileParams, workerId) => {
    if (viewer) {
        const overlayId = `worker-${workerId}-overlay`;
        const existingOverlay = document.getElementById(overlayId);
        if (existingOverlay) viewer.removeOverlay(existingOverlay);

        const elt = document.createElement("div");
        elt.id = overlayId;
        const colors = ['border-blue-400', 'border-purple-400', 'border-cyan-400', 'border-pink-400', 'border-yellow-400'];
        const borderColor = colors[workerId % colors.length];

        elt.className = `highlight border-4 border-dashed ${borderColor} transition ease-linear duration-300 bg-white/10 pointer-events-none`;

        viewer.addOverlay({
            element: elt,
            location: viewer.viewport.imageToViewportRectangle(tileParams.topLeftX, tileParams.topLeftY, tileParams.width, tileParams.height)
        });
    }
}

const removeViewerOverlay = (workerId) => {
    if (viewer) {
        const overlayId = `worker-${workerId}-overlay`;
        const existingOverlay = document.getElementById(overlayId);
        if (existingOverlay) viewer.removeOverlay(existingOverlay);
    }
}

const workerQueue = {
    pending: [],
    taskResolvers: new Map(), // (x-y) -> resolve function

    async add(task) {
        return new Promise((resolve, reject) => {
            this.pending.push({ task, resolve, reject });
            this.process();
        });
    },

    async process() {
        if (availableWorkers.length === 0 || this.pending.length === 0) return;

        const { task, resolve, reject } = this.pending.shift();
        const workerId = availableWorkers.shift();
        const worker = workerPool[workerId];

        try {
            const { imageSource, modelIdentifier, region, tissueRegionIndex, totalRegions } = task;
            const { emptyProportion, isEmpty, ...patchParams } = region;

            console.log(`[Queue] Fetching & Processing region ${tissueRegionIndex + 1}/${totalRegions} (Worker: ${workerId}, Pending: ${this.pending.length})`);

            const patchURL = await getImageTile([
                ...Object.values(patchParams),
                Math.max(patchParams.width, patchParams.height)
            ]);

            addViewerOverlay(patchParams, workerId);

            const tempImg = new Image();
            tempImg.src = patchURL;
            await new Promise((res, rej) => {
                tempImg.onload = res;
                tempImg.onerror = rej;
            });

            const oc = new OffscreenCanvas(tempImg.naturalWidth, tempImg.naturalHeight);
            const ctx = oc.getContext('2d');
            ctx.drawImage(tempImg, 0, 0);
            const bitmap = await createImageBitmap(oc);

            // Create a promise that waits for this specific patch's completion
            const completionPromise = new Promise((res) => {
                const id = `${patchParams.topLeftX}-${patchParams.topLeftY}`;
                this.taskResolvers.set(id, res);
            });

            const precision = document.getElementById('precisionSelect').value;
            const storageType = document.getElementById('storageSelect').value;

            worker.postMessage({
                imageSource,
                modelIdentifier,
                patchParams,
                bitmap,
                final: tissueRegionIndex === totalRegions - 1,
                precision,
                storageType
            }, [bitmap]);

            console.log(`[Queue] Region ${tissueRegionIndex + 1} sent to Worker ${workerId}.`);
            URL.revokeObjectURL(patchURL);

            // Wait for worker to finish this specific patch
            await completionPromise;

            console.log(`[Queue] Region ${tissueRegionIndex + 1} finalized (Worker ${workerId} free).`);
            resolve();
        } catch (err) {
            console.error(`[Queue] Error in region ${task.tissueRegionIndex + 1} (Worker ${workerId}):`, err);
            availableWorkers.push(workerId); // Return worker to pool on error
            reject(err);
        } finally {
            this.process();
        }
    }
    ,

    resolveTask(patchParams) {
        const id = `${patchParams.topLeftX}-${patchParams.topLeftY}`;
        const resolver = this.taskResolvers.get(id);
        if (resolver) {
            resolver();
            this.taskResolvers.delete(id);
        }
    }
};

const generatePatchEmbeddings = async (imageSource, modelIdentifier, tissueRegions) => {
    updateStatus('Generating embeddings...');
    const promises = tissueRegions.map((region, index) => {
        return workerQueue.add({
            imageSource,
            modelIdentifier,
            region,
            tissueRegionIndex: index,
            totalRegions: tissueRegions.length
        });
    });

    await Promise.all(promises);
};

function generateEmbeddingsHandler(e) {
    if (!patchEmbed.imagebox3Instance) {
        alert("Please load the image first!")
        return
    }
    e.target.setAttribute('disabled', "true")
    return generateEmbeddings(patchEmbed.imagebox3Instance.getImageSource(), document.getElementById('modelSelect').value)
}

export async function generateEmbeddings(imageSource, modelIdentifier = "CTransPath") {
    if (!patchEmbed.imagebox3Instance && !imageSource) {
        alert("Please load the image first!")
        return
    } else if (patchEmbed.imagebox3Instance?.getImageSource() !== imageSource) {
        await setupImageBox3Instance(imageSource)
    }

    if (!db) {
        await initIndexedDB()
    }

    if (!workerPool || workerPool.length === 0) {
        await initWorker()
    }

    console.time("allEmbeddings")
    const tissueRegions = await getTissueRegions()

    await generatePatchEmbeddings(patchEmbed.imagebox3Instance.getImageSource(), modelIdentifier, tissueRegions)

    console.timeEnd("allEmbeddings")
    updateStatus('Embeddings generated for all patches!')

    // Dispatch event for any other listeners
    const allEmbeddingsGeneratedEvent = new CustomEvent('allEmbeddingsGenerated')
    document.dispatchEvent(allEmbeddingsGeneratedEvent)

    // Finalization logic
    viewer.currentOverlays.forEach(overlay => viewer.removeOverlay(overlay.element))
    if (document?.getElementById("generateEmbeddings")) {
        document?.getElementById("generateEmbeddings").removeAttribute('disabled')
    }
    const allEmbeddings = await retrieveEmbeddings()
    await runClusteringAndHeatmap(allEmbeddings);
}

function toggleSelection() {
    isSelecting = !isSelecting
    const button = document.getElementById('selectRegion')

    if (isSelecting) {
        button.textContent = 'Stop Selecting'
        button.className = 'bg-yellow-600 hover:bg-yellow-700 px-3 py-1 rounded text-sm'
        updateStatus('Selection mode active - click and drag to select region')
    } else {
        button.textContent = 'Select Region'
        button.className = 'bg-green-600 hover:bg-green-700 px-3 py-1 rounded text-sm'
        updateStatus('Selection mode deactivated')
    }
}

// function clearSelection() {
//     selectedRegion = null
//     isSelecting = false

//     if (selectionOverlay) {
//         viewer.removeOverlay(selectionOverlay)
//         selectionOverlay = null
//     }

//     viewer.clearOverlays()

//     if (currentPlotData) {
//         const colors = currentPlotData.data[0].z
//         Plotly.restyle('plot', { 'marker.color': [colors] }, [0])
//     }

//     const button = document.getElementById('selectRegion')
//     button.textContent = 'Select Region'
//     button.className = 'bg-green-600 hover:bg-green-700 px-3 py-1 rounded text-sm'

//     updateStatus('Selection cleared')
// }

function updateStatus(message) {
    if (document?.getElementById('status')?.textContent)
        document.getElementById('status').textContent = message
}

function countIDBRecords() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('WSIEmbeddings')

        request.onerror = () => {
            reject(`Failed to open database: ${request.error}`)
        }

        request.onsuccess = () => {
            const db = request.result
            const transaction = db.transaction('embeddings', 'readonly')
            const store = transaction.objectStore('embeddings')

            const countRequest = store.count()

            countRequest.onsuccess = () => {
                resolve(countRequest.result)
                db.close()
            }

            countRequest.onerror = () => {
                reject(`Count operation failed: ${countRequest.error}`)
                db.close()
            }
        }
    })
}

async function updateEmbeddingCount() {
    const numEmbeddings = await countIDBRecords()
    if (document?.getElementById('embeddingCount')?.textContent)
        document.getElementById('embeddingCount').textContent = numEmbeddings
}

// Update clustering when parameters change
async function updateClustering() {
    const currentEmbeddings = await retrieveEmbeddings()
    if (currentEmbeddings.result.length > 0) {
        runClusteringAndHeatmap(currentEmbeddings);
    }
}

function handleModelChange(event) {
    const modelId = parseInt(event.target.value);
    selectedModel = availableModels.find(m => m.modelId === modelId);
    if (selectedModel) {
        updateStatus(`Selected model: ${selectedModel.modelName}`);
    }
}

// Clear selection
function clearSelection() {
    selectedRegion = null;
    isSelecting = false;

    if (selectionOverlay) {
        viewer.removeOverlay(selectionOverlay);
        selectionOverlay = null;
    }

    // Reset heatmap patch appearances
    heatmapOverlays.forEach((overlay, i) => {
        if (currentClusters[i]) {
            const colors = generateClusterColors(Math.max(...currentClusters.map(c => c.cluster)) + 1);
            const color = colors[currentClusters[i].cluster];
            // overlay.style.opacity = '0.2';
            // overlay.style.border = `1px solid rgba(${color.r}, ${color.g}, ${color.b}, 0.2)`;
        }
    });

    const button = document.getElementById('selectRegion');
    button.textContent = 'Select Region';
    button.className = 'bg-green-600 hover:bg-green-700 px-3 py-1 rounded text-sm';

    updateStatus('Selection cleared');
}

// function updateClusterCount(count) {
//     document.getElementById('clusterCount').textContent = count;
// }
function renderAnnotationUI(numClusters) {
    const container = document.getElementById('annotationList');
    container.innerHTML = '';

    // Get the same colors used in the viewer
    const colors = generateClusterColors(numClusters);

    colors.forEach((color, index) => {
        const rgbString = `rgb(${color.r}, ${color.g}, ${color.b})`;
        const existingLabel = clusterLabels[index] || `Cluster ${index + 1}`;

        const row = document.createElement('div');
        row.className = 'flex items-center space-x-3 bg-gray-700/50 p-2 rounded border border-gray-600';

        row.innerHTML = `
            <div class="w-6 h-6 rounded flex-shrink-0 border border-white/20" style="background-color: ${rgbString};"></div>
            <div class="flex-1">
                <input type="text" 
                    data-cluster-index="${index}"
                    value="${existingLabel}"
                    class="cluster-label-input w-full bg-transparent border-none focus:ring-0 text-sm text-white placeholder-gray-500"
                    placeholder="Enter label for Cluster ${index + 1}..."
                >
            </div>
            <div class="text-xs text-gray-400 px-2">${Math.round((currentClusters.filter(c => c.cluster === index).length / currentClusters.length) * 100)}%</div>
        `;

        // Interactive Highlighting
        const input = row.querySelector('input');

        // Update state on change
        input.addEventListener('input', (e) => {
            clusterLabels[index] = e.target.value;
        });

        // Highlight on focus/hover
        row.addEventListener('mouseenter', () => highlightSpecificCluster(index));
        row.addEventListener('mouseleave', () => resetClusterHighlight());

        container.appendChild(row);
    });
}

function highlightSpecificCluster(clusterIndex) {
    drawHeatmap(currentClusters, clusterIndex);
}

function resetClusterHighlight() {
    drawHeatmap(currentClusters);
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        document.getElementById('imageUrl').placeholder = `File selected: ${file.name}`;
        document.getElementById('imageUrl').value = '';
    }
}