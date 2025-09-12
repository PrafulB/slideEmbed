import { Imagebox3 } from "https://episphere.github.io/imagebox3/imagebox3.mjs"
import { UMAP } from "https://esm.sh/umap-js"

const patchEmbed = {}

let viewer = null
let worker = null
let db = null
let currentEmbeddings = []
let currentPlotData = null
let selectedRegion = null
let isSelecting = false
let selectionOverlay = null
let hnswIndex = null


document.addEventListener('DOMContentLoaded', async () => {
    await initIndexedDB()
    initViewer()
    initWorker()
    bindEvents()
})


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
        gestureSettingsMouse: {
            clickToZoom: false,
            dblClickToZoom: true
        }
    })

    viewer.addHandler('canvas-click', handleViewerClick)
    viewer.addHandler('canvas-drag', handleViewerDrag)
    viewer.addHandler('canvas-drag-end', handleViewerDragEnd)
}

function initWorker() {
    const baseURL = import.meta.url.split("/").slice(0, -1).join("/")
    worker = new Worker(URL.createObjectURL(new Blob([`importScripts("${baseURL}/worker.js")`])))

    worker.onmessage = handleWorkerMessage
    worker.onerror = (error) => {
        console.error('Worker error:', error)
        updateStatus('Worker error occurred')
    }
}

async function handleWorkerMessage(e) {
    const { success, data, error, final } = e.data
    if (success) {
        self.workerIsEmbedding = false
        updateEmbeddingCount()

        // Build HNSW index (placeholder for now)
        await buildHNSWIndex(data)

        // await runUMAPAndPlot()
        if (final) {
            updateStatus('Embeddings generated for all patches!')
            const allEmbeddingsGeneratedEvent = new CustomEvent('allEmbeddingsGenerated')
            document.dispatchEvent(allEmbeddingsGeneratedEvent)
        } else {
            updateStatus('Embeddings generated for current patch, moving to next...')
        }
    } else {
        console.error('Worker error:', error)
        updateStatus('Error generating embeddings')
    }
}

async function setupImageBox3Instance(input) {
    if (!patchEmbed.imagebox3Instance) {
        const numWorkers = Math.floor(navigator.hardwareConcurrency / 2)
        patchEmbed.imagebox3Instance = new Imagebox3(input, numWorkers)
        await patchEmbed.imagebox3Instance.init()
    }
    else {
        await patchEmbed.imagebox3Instance.changeImageSource(input)
    }
}

async function createTileSource(input) {
    await setupImageBox3Instance(input)

    let tileSources = {}
    try {
        tileSources = await OpenSeadragon.GeoTIFFTileSource.getAllTileSources(input, { logLatency: false, cache: true, slideOnly: true, pool: patchEmbed.imagebox3Instance.workerPool })
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
            w: e.patchWidth,
            h: e.patchHeight
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

export async function retrieveEmbeddings(imageSource=patchEmbed.imagebox3Instance?.getImageSource(), lowerBound=[0,0], upperBound=[Infinity,Infinity]) {
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

async function runUMAPAndPlot() {
    updateStatus('Running UMAP...')

    const umapResults = await runUMAP(embeddings, 3)

    const x = umapResults.map(p => p[0])
    const y = umapResults.map(p => p[1])
    const z = umapResults.map(p => p[2])

    const text = embeddings.map((e, i) =>
        `Patch ${i}<br>X: ${e.patchTopLeftX}<br>Y: ${e.patchTopLeftY}<br>Size: ${e.patchWidth}x${e.patchHeight}`
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
            embedding.patchWidth, embedding.patchHeight)
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

async function processSelectedRegion() {
    if (!selectedRegion || !currentEmbeddings.length) return

    const minX = Math.min(selectedRegion.startX, selectedRegion.endX)
    const maxX = Math.max(selectedRegion.startX, selectedRegion.endX)
    const minY = Math.min(selectedRegion.startY, selectedRegion.endY)
    const maxY = Math.max(selectedRegion.startY, selectedRegion.endY)

    const selectedPatches = currentEmbeddings.filter(patch => {
        const patchCenterX = patch.patchTopLeftX + patch.patchWidth / 2
        const patchCenterY = patch.patchTopLeftY + patch.patchHeight / 2
        return patchCenterX >= minX && patchCenterX <= maxX &&
            patchCenterY >= minY && patchCenterY <= maxY
    })

    if (selectedPatches.length === 0) return

    highlightPlotPoints(selectedPatches)

    await findSimilarPatches(selectedPatches)
}

function highlightPlotPoints(patches) {
    if (!currentPlotData) return

    const highlightIndices = patches.map(patch =>
        currentEmbeddings.findIndex(e => e === patch)
    ).filter(index => index !== -1)

    const colors = currentPlotData.data[0].marker.color.map((_, i) =>
        highlightIndices.includes(i) ? 'red' : 'blue'
    )

    Plotly.restyle('plot', { 'marker.color': [colors] }, [0])
}

async function findSimilarPatches(selectedPatches) {
    if (selectedPatches.length === 0) return

    const avgEmbedding = new Array(selectedPatches[0].embedding.length).fill(0)
    selectedPatches.forEach(patch => {
        patch.embedding.forEach((val, i) => {
            avgEmbedding[i] += val / selectedPatches.length
        })
    })

    const similarities = currentEmbeddings.map(patch => {
        let dotProduct = 0, normA = 0, normB = 0
        patch.embedding.forEach((val, i) => {
            dotProduct += val * avgEmbedding[i]
            normA += val * val
            normB += avgEmbedding[i] * avgEmbedding[i]
        })
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
    })

    const threshold = 0.7
    const similarPatches = currentEmbeddings.filter((_, i) => similarities[i] > threshold)

    similarPatches.forEach(patch => {
        const element = document.createElement('div')
        element.style.border = '2px solid #F59E0B'
        element.style.backgroundColor = 'rgba(245, 158, 11, 0.3)'
        element.style.pointerEvents = 'none'

        const rect = new OpenSeadragon.Rect(
            patch.patchTopLeftX, patch.patchTopLeftY,
            patch.patchWidth, patch.patchHeight
        )

        viewer.addOverlay(element, rect)

        setTimeout(() => {
            viewer.removeOverlay(element)
        }, 5000)
    })
}

function bindEvents() {
    document.getElementById('loadImage').addEventListener('click', loadImage)
    document.getElementById('generateEmbeddings').addEventListener('click', generateEmbeddings)
    document.getElementById('selectRegion').addEventListener('click', toggleSelection)
    document.getElementById('clearSelection').addEventListener('click', clearSelection)
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
    } else {
        updateStatus('Failed to load image')
    }
}

async function getTissueRegions(gridDim = 16) {
    if (!patchEmbed.imagebox3Instance) return
    console.time("thumbnail")
    const imageInfo = await patchEmbed.imagebox3Instance.getInfo()
    const thumbnailBlob = await patchEmbed.imagebox3Instance.getThumbnail(512, 512)
    const thumbnailURL = URL.createObjectURL(thumbnailBlob)
    console.timeEnd("thumbnail")
    const thumbnailImg = new Image()

    return new Promise((resolve) => {
        thumbnailImg.onload = () => {
            const thumbnailWidth = thumbnailImg.naturalWidth
            const thumbnailHeight = thumbnailImg.naturalHeight
            const thumbnailRegions = Array(gridDim)
                .fill(undefined)
                .map((row, rowIdx) =>
                    Array(gridDim)
                        .fill(undefined)
                        .map((col, colIdx) => [
                            (thumbnailWidth * rowIdx) / gridDim,
                            (thumbnailHeight * colIdx) / gridDim
                        ])
                )
                .flat()
            const offscreenCanvas = new OffscreenCanvas(
                thumbnailWidth / gridDim,
                thumbnailHeight / gridDim
            )

            const tissueRegions = thumbnailRegions.map(([x, y]) => {
                const offscreenCtx = offscreenCanvas.getContext("2d")
                offscreenCtx.drawImage(
                    thumbnailImg,
                    x,
                    y,
                    offscreenCanvas.width,
                    offscreenCanvas.height,
                    0,
                    0,
                    offscreenCanvas.width,
                    offscreenCanvas.height
                )
                const tileContent = isTileEmpty(
                    offscreenCanvas,
                    offscreenCtx,
                    0.8,
                    true
                )
                const topLeftX = Math.floor((x * imageInfo.width) / thumbnailWidth)
                const topLeftY = Math.floor((y * imageInfo.height) / thumbnailHeight)
                return {
                    topLeftX,
                    topLeftY,
                    width: Math.floor(imageInfo.width / gridDim),
                    height: Math.floor(imageInfo.height / gridDim),
                    ...tileContent
                }
            })
                .filter((tile) => !tile.isEmpty)
                .sort((a, b) => a.emptyProportion - b.emptyProportion)
            resolve(tissueRegions)
        }
        thumbnailImg.src = thumbnailURL
    })

}

const isTileEmpty = (canvas, ctx, threshold = 0.9, returnEmptyProportion = false) => {
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

const addViewerOverlay = (tileParams) => {
    if (viewer) {
        const existingOverlay = document.getElementById("runtime-patch-overlay")
        if (existingOverlay) viewer.removeOverlay(existingOverlay)
    
        const elt = document.createElement("div")
        elt.id = "runtime-patch-overlay"
        elt.className = "highlight"
        viewer.addOverlay({
            element: elt,
            location: viewer.viewport.imageToViewportRectangle(tileParams.topLeftX, tileParams.topLeftY, tileParams.width, tileParams.height)
        })
    }
}

const generatePatchEmbeddings = async (imageSource, modelIdentifier, tissueRegions, tissueRegionIndex = 0, patchResolution = 256) => {
    console.log(`Embedded ${(await retrieveEmbeddings(imageSource)).result.length} patches for current image.`)
    console.log(`Starting tissue region ${tissueRegionIndex+1}/${tissueRegions.length}...`)
    try {
        const { emptyProportion, isEmpty, ...patchParams } = tissueRegions[tissueRegionIndex]
        const patchURL = await getImageTile([
            ...Object.values(patchParams),
            Math.max(patchParams.width, patchParams.height)
        ])
        addViewerOverlay(patchParams)
        const tempImg = new Image()
        tempImg.onload = async () => {
            const oc = new OffscreenCanvas(tempImg.naturalWidth, tempImg.naturalHeight)
            const ctx = oc.getContext('2d')
            ctx.drawImage(tempImg, 0, 0, oc.width, oc.height)
            const bitmap = await createImageBitmap(oc)
            let timeElapsed = 0
            while (self.workerIsEmbedding && timeElapsed < 100000) {
                await new Promise(res => setTimeout(res, 100))
                timeElapsed += 100
            }
            if (timeElapsed >= 100000) {
                console.error("Patch Embeddings generation took too long, exiting")
                return undefined
            }
            self.workerIsEmbedding = true
            worker.postMessage({ imageSource, modelIdentifier, patchParams, bitmap, final: tissueRegionIndex === tissueRegions.length - 1 }, [bitmap])
            URL.revokeObjectURL(patchURL)
            if (tissueRegionIndex < tissueRegions.length - 1) {
                generatePatchEmbeddings(imageSource, modelIdentifier, tissueRegions, tissueRegionIndex + 1)
            } else {
                updateStatus("Embeddings generated for all patches!")
            }
        }
        tempImg.src = patchURL
        tempImg.onerror = (e) => {
            console.log(`Error loading patch ${Object.values(tissueRegions[tissueRegionIndex])}`, e)
            if (tissueRegionIndex < tissueRegions.length - 1) {
                generatePatchEmbeddings(imageSource, modelIdentifier, tissueRegions, tissueRegionIndex + 1)
            } else {
                updateStatus("Embeddings generated for all patches!")
            }
        }
        updateStatus('Generating embeddings...')
    }
    catch (e) {
        console.log(
            `Error generating embeddings ${e}, ${tissueRegionIndex}`
        )
        updateStatus('Failed to generate some embeddings!')
        if (tissueRegionIndex < tissueRegions.length - 1) {
            generatePatchEmbeddings(imageSource, modelIdentifier, tissueRegions, tissueRegionIndex + 1)
        } else {
            updateStatus("Embeddings generated for all patches!")
        }
    }
}

export async function generateEmbeddings(imageSource, modelIdentifier="CTransPath") {
    if (!patchEmbed.imagebox3Instance && !imageSource) {
        alert("Please load the image first!")
        return
    } else if (patchEmbed.imagebox3Instance?.getImageSource() !== imageSource) {
        await setupImageBox3Instance(imageSource)
    }

    if (!db) {
        await initIndexedDB()
    }

    if (!worker) {
        await initWorker()
    }

    console.time("allEmbeddings")
    const tissueRegions = await getTissueRegions()
    console.log(tissueRegions)
    return new Promise(async (resolve) => {
        await generatePatchEmbeddings(patchEmbed.imagebox3Instance.getImageSource(), modelIdentifier, tissueRegions)
        document.addEventListener('allEmbeddingsGenerated', async () => {
            console.timeEnd("allEmbeddings")
            const allEmbeddings = await retrieveEmbeddings()
            resolve(allEmbeddings)
        })
    })
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

function clearSelection() {
    selectedRegion = null
    isSelecting = false

    if (selectionOverlay) {
        viewer.removeOverlay(selectionOverlay)
        selectionOverlay = null
    }

    viewer.clearOverlays()

    if (currentPlotData) {
        const colors = currentPlotData.data[0].z
        Plotly.restyle('plot', { 'marker.color': [colors] }, [0])
    }

    const button = document.getElementById('selectRegion')
    button.textContent = 'Select Region'
    button.className = 'bg-green-600 hover:bg-green-700 px-3 py-1 rounded text-sm'

    updateStatus('Selection cleared')
}

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
    if(document?.getElementById('embeddingCount')?.textContent)
        document.getElementById('embeddingCount').textContent = numEmbeddings
}


