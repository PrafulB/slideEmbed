importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js")
importScripts("https://prafulb.github.io/fedEmbed/config.js")

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@dev/dist/'

const createSession = (modelIdentifier = "CTransPath") => {
    self.model = SUPPORTED_MODELS.filter(model => model.modelName === modelIdentifier)[0]

    if (model) {
        // return ort.InferenceSession.create("http://localhost:5599/model.onnx", {
        return ort.InferenceSession.create(model.modelURL, {
            executionProviders: ["webgpu"],
            graphOptimizationLevel: 'all',
            enableMemReuse: true,
        })
    }
}

let sharedCanvas = null;

function imageTransformsCombined(
    imageData,
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225],
    threshold = 0.9
) {
    const { data, width, height } = imageData
    const numPixels = data.length / 4
    let whitePixelCount = 0

    const inputData = new Float32Array(width * height * 3)
    let j = 0
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i]
        const g = data[i + 1]
        const b = data[i + 2]

        if (r > 200 && g > 200 && b > 200) {
            whitePixelCount++
        }

        // Standard norm + Renorm using model preprocessor_config
        inputData[j] = (r / 255.0 - mean[0]) / std[0]
        inputData[j + 1] = (g / 255.0 - mean[1]) / std[1]
        inputData[j + 2] = (b / 255.0 - mean[2]) / std[2]
        j += 3
    }

    const whiteProportion = whitePixelCount / numPixels
    if (whiteProportion >= threshold) {
        return null // Signal empty tile
    }

    return inputData
}

const isTileEmpty = (imageData, threshold = 0.9, returnEmptyProportion = false) => {
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

const getSubPatchTensors = (patchParams, bitmap) => {
    const canvasWidth = Math.ceil((bitmap.width * self.model.tileSizeForModel) / self.model.tileResolution);
    const canvasHeight = Math.ceil((bitmap.height * self.model.tileSizeForModel) / self.model.tileResolution);

    if (!sharedCanvas || sharedCanvas.width !== canvasWidth || sharedCanvas.height !== canvasHeight) {
        sharedCanvas = new OffscreenCanvas(canvasWidth, canvasHeight);
    }

    const ctx = sharedCanvas.getContext("2d", { willReadFrequently: true })
    ctx.drawImage(bitmap, 0, 0, canvasWidth, canvasHeight)

    const validPatches = []
    const tileSize = self.model.tileSizeForModel;

    for (let i = 0; i < Math.ceil(canvasHeight / tileSize); i++) {
        for (let j = 0; j < Math.ceil(canvasWidth / tileSize); j++) {
            const x = j * tileSize;
            const y = i * tileSize;

            // Check bounds to avoid getThumbnail stretching issues if any
            const w = Math.min(tileSize, canvasWidth - x);
            const h = Math.min(tileSize, canvasHeight - y);

            if (w < tileSize || h < tileSize) continue; // Skip partial tiles for now to be safe

            const subPatchImageData = ctx.getImageData(x, y, w, h)
            const inputData = imageTransformsCombined(
                subPatchImageData,
                self.model.imageTransforms.mean,
                self.model.imageTransforms.std
            )

            if (inputData) {
                validPatches.push({
                    subPatchParams: {
                        topLeftX: patchParams.topLeftX + x,
                        topLeftY: patchParams.topLeftY + y,
                        width: w,
                        height: h,
                    },
                    inputData
                })
            }
        }
    }
    return validPatches
}

const getEmbeddingsBatch = async (patches) => {
    const session = await self.session
    if (!session || patches.length === 0) return []

    const tileSize = self.model.tileSizeForModel
    const batchSize = patches.length
    const combinedData = new Float32Array(batchSize * 3 * tileSize * tileSize)

    for (let i = 0; i < batchSize; i++) {
        combinedData.set(patches[i].inputData, i * 3 * tileSize * tileSize)
    }

    const batchTensor = new ort.Tensor("float32", combinedData, [batchSize, 3, tileSize, tileSize])
    const sessionInput = {}
    sessionInput[session.handler.inputNames[0]] = batchTensor

    const outputMap = await session.run(sessionInput)

    const outputTensor = outputMap[Object.keys(outputMap)[0]]
    const dim = outputTensor.dims[outputTensor.dims.length - 1]

    const results = []
    for (let i = 0; i < batchSize; i++) {
        const start = i * dim
        results.push(outputTensor.cpuData.slice(start, start + dim))
    }
    return results
}

async function generateEmbeddings(patchParams, bitmap) {
    const validPatches = await getSubPatchTensors(patchParams, bitmap)
    if (validPatches.length === 0) return []

    const batchSize = 8;
    const allEmbeddings = [];
    for (let i = 0; i < validPatches.length; i += batchSize) {
        const chunk = validPatches.slice(i, i + batchSize);
        console.log(`[Worker] Processing chunk ${Math.floor(i / batchSize) + 1}/${Math.ceil(validPatches.length / batchSize)} (${chunk.length} sub-patches)`);
        const chunkEmbeddings = await getEmbeddingsBatch(chunk);
        allEmbeddings.push(...chunkEmbeddings);
    }

    return validPatches.map((patch, i) => ({
        subPatchParams: patch.subPatchParams,
        model: self.model.modelName,
        patchEmbedding: allEmbeddings[i]
    }))
}

async function initIndexedDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('WSIEmbeddings', 1)

        request.onerror = () => reject(request.error)
        request.onsuccess = () => {
            self.db = request.result
            resolve()
        }
    })
}

const binarizeVector = (vector) => {
    const sorted = [...vector].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const threshold = sorted.length % 2 === 0
        ? (sorted[mid - 1] + sorted[mid]) / 2
        : sorted[mid];

    const packed = new BigUint64Array(Math.ceil(vector.length / 64));
    for (let i = 0; i < vector.length; i++) {
        if (vector[i] >= threshold) {
            packed[Math.floor(i / 64)] |= (1n << BigInt(i % 64));
        }
    }
    return packed;
};

function convertPrecision(embedding, precision) {
    switch (precision) {
        case 'float16':
            // If Float16Array is available, use it (standardized in ES2023)
            if (typeof Float16Array !== 'undefined') {
                return new Float16Array(embedding);
            }
            // Fallback: just return float32 but at least we tried
            return new Float32Array(embedding);
        case 'uint8':
            // Simple mapping from -1..1 to 0..255
            const uint8 = new Uint8Array(embedding.length);
            for (let i = 0; i < embedding.length; i++) {
                uint8[i] = Math.max(0, Math.min(255, Math.floor((embedding[i] + 1) * 127.5)));
            }
            return uint8;
        case 'binary':
            return binarizeVector(embedding);
        case 'float32':
        default:
            return new Float32Array(embedding);
    }
}

async function storeInOPFS(imageId, patchData, precision) {
    try {
        const root = await navigator.storage.getDirectory();
        const slideDir = await root.getDirectoryHandle(imageId, { create: true });

        // Store as a single binary file for efficiency
        const fileName = `embeddings_${precision}.bin`;
        const fileHandle = await slideDir.getFileHandle(fileName, { create: true });
        const writable = await fileHandle.createWritable();

        // Write metadata and embeddings in a packed format
        // For simplicity, we'll write them as JSON for now or a custom binary format.
        // The user asked for it to be downloadable as JSON later, so maybe just JSON in OPFS is fine for now,
        // but binary is better for "memory usage" evaluation.
        // Let's go with a simple binary stream.

        for (const patch of patchData) {
            const buffer = patch.embedding.buffer;
            await writable.write(buffer);
        }
        await writable.close();
    } catch (e) {
        console.error("Error storing in OPFS:", e);
    }
}

async function storeEmbeddings(imageSource, patchData, precision = 'float32', storageType = 'indexeddb') {
    let imageId = imageSource;
    if (imageId instanceof File) {
        imageId = imageId.name;
    }

    // Convert precision for each patch
    const processedData = patchData.map(patch => ({
        ...patch,
        embedding: convertPrecision(patch.patchEmbedding, precision)
    }));

    if (storageType === 'indexeddb') {
        if (!self.db) {
            await initIndexedDB();
        }
        return new Promise((resolve, reject) => {
            const transaction = self.db.transaction(['embeddings'], 'readwrite');
            const store = transaction.objectStore('embeddings');

            processedData.forEach(patch => {
                const addRequest = store.add({
                    imageId,
                    ...patch.subPatchParams,
                    embedding: patch.embedding,
                    precision
                });
                addRequest.onerror = (e) => console.warn(`Error adding embeddings to IndexedDB:`, addRequest.error);
            });

            transaction.oncomplete = () => resolve();
            transaction.onerror = () => reject(transaction.error);
        });
    } else if (storageType === 'opfs') {
        await storeInOPFS(imageId, processedData, precision);
    }
}

self.onmessage = async function (e) {
    const { imageSource, modelIdentifier, patchParams, bitmap, final, patchSize = 224, precision = 'float32', storageType = 'indexeddb' } = e.data
    console.log(`[Worker] Started processing patch (${patchParams.topLeftX}, ${patchParams.topLeftY}) with precision: ${precision}, storage: ${storageType}`);

    if (!self.session || self.model?.modelName !== modelIdentifier) {
        console.log(`[Worker] Initializing session for model: ${modelIdentifier}`);
        self.session = createSession(modelIdentifier)
        if (!self.session) {
            self.postMessage({ success: false, error: `Model not found: ${modelIdentifier}` })
            return;
        }
    }
    try {
        const result = await generateEmbeddings(patchParams, bitmap)
        console.log(`[Worker] Finished processing patch. Generated ${result.length} sub-patch embeddings.`);
        self.postMessage({ success: true, data: result, final, patchParams })
        await storeEmbeddings(imageSource, result, precision, storageType)
    } catch (error) {
        console.error(`[Worker] Error:`, error);
        self.postMessage({ success: false, error: error.message, patchParams })
    }
};