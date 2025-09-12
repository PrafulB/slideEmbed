importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js")
importScripts("https://prafulb.github.io/fedEmbed/config.js")

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@dev/dist/'

const createSession = (modelIdentifier = "CTransPath") => {
    self.model = SUPPORTED_MODELS.filter(model => model.modelName === modelIdentifier)[0]

    if (model) {
        return ort.InferenceSession.create(model.modelURL, {
            executionProviders: ["webgpu"]
        })
    }
}

function imageTransforms(
    imageData,
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225],
    canvasDim = 224
) {
    // Resize + Center cropping
    const canvas = new OffscreenCanvas(canvasDim, canvasDim)
    const ctx = canvas.getContext("2d")
    const shortestEdge = Math.min(imageData.width, imageData.height)
    const scale = canvasDim / shortestEdge
    const newWidth = imageData.width * scale
    const newHeight = imageData.height * scale
    const offsetX = (newWidth - canvasDim) / 2
    const offsetY = (newHeight - canvasDim) / 2
    ctx.putImageData(imageData, -offsetX, -offsetY)

    const transformedImageData = ctx.getImageData(0, 0, canvasDim, canvasDim)
    const { data, width, height } = transformedImageData

    const inputData = new Float32Array(width * height * 3)
    let j = 0
    for (let i = 0; i < data.length; i += 4) {
        // Standard norm
        let r = data[i] / 255.0
        let g = data[i + 1] / 255.0
        let b = data[i + 2] / 255.0

        // Renorm using model preprocessor_config
        inputData[j] = (r - mean[0]) / std[0]
        inputData[j + 1] = (g - mean[1]) / std[1]
        inputData[j + 2] = (b - mean[2]) / std[2]
        j += 3
    }

    const tensor = new ort.Tensor("float32", inputData, [
        1,
        3,
        canvasDim,
        canvasDim
    ])
    return tensor
}

const getSubPatchTensors = (patchParams, bitmap) => {
    //V.IMP TODO: Check that the border tiles (with width or height less than tileSizeForModel) are not being
    // stretched inordinately and creating bad embeddings.
    const oc = new OffscreenCanvas(
        (bitmap.width * self.model.tileSizeForModel) /
        self.model.tileResolution,
        (bitmap.height * self.model.tileSizeForModel) /
        self.model.tileResolution
    )
    const ctx = oc.getContext("2d", { willReadFrequently: true })
    ctx.drawImage(bitmap, 0, 0, oc.width, oc.height)
    const subPatchTensors = []
    for (let i = 0; i < Math.ceil(bitmap.height / self.model.tileResolution); i++) {
        for (let j = 0; j < Math.ceil(bitmap.width / self.model.tileResolution); j++) {
            const subPatchParamsOnCanvas = {
                topLeftX: j * self.model.tileSizeForModel,
                topLeftY: i * self.model.tileSizeForModel,
                width: self.model.tileSizeForModel,
                height: self.model.tileSizeForModel
            }
            const subPatchImageData = ctx.getImageData(...Object.values(subPatchParamsOnCanvas))
            const subPatchTensor = imageTransforms(subPatchImageData, self.model.imageTransforms.mean, self.model.imageTransforms.std, self.model.tileSizeForModel)
            subPatchTensors.push({
                subPatchParams: {
                    topLeftX: patchParams.topLeftX + subPatchParamsOnCanvas.topLeftX,
                    topLeftY: patchParams.topLeftY + subPatchParamsOnCanvas.topLeftY,
                    width: subPatchParamsOnCanvas.width,
                    height: subPatchParamsOnCanvas.height,
                },
                tensor: subPatchTensor
            })
        }
    }
    return subPatchTensors
}

const getEmbeddingFromTensor = async (imageTensor) => {
    const session = await self.session
    if (!session) {
        return undefined
    }
    const sessionInput = {}
    sessionInput[session.handler.inputNames[0]] = imageTensor

    const x = await session.run(sessionInput)
    const clsToken = x[Object.keys(x)[0]].cpuData.slice(
        0,
        x[Object.keys(x)[0]].dims[x[Object.keys(x)[0]].dims.length - 1]
    )
    return clsToken
}

async function generateEmbeddings(patchParams, bitmap) {
    const subPatchTensors = await getSubPatchTensors(patchParams, bitmap)
    const patchEmbeddings = []
    for (const subPatch of subPatchTensors) {
        const patchEmbedding = await getEmbeddingFromTensor(subPatch.tensor)
        patchEmbeddings.push({
            subPatchParams: subPatch.subPatchParams,
            model: self.model.modelName,
            patchEmbedding
        })
    }
    return (patchEmbeddings)
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

async function storeEmbeddings(imageId, patchData) {
    if (!self.db) {
        await initIndexedDB()
    }

    return new Promise((resolve, reject) => {
        const transaction = self.db.transaction(['embeddings'], 'readwrite')
        const store = transaction.objectStore('embeddings')

        patchData.forEach(patch => {
            const addRequest = store.add({
                imageId,
                ...patch.subPatchParams,
                embedding: patch.patchEmbedding
            })
            addRequest.onerror = (e) => console.warn(`Error adding embeddings to IndexedDB:`, addRequest.error)
        })

        transaction.oncomplete = () => resolve()
        transaction.onerror = () => reject(transaction.error)
    })
}

self.onmessage = async function (e) {

    const { imageSource, modelIdentifier, patchParams, bitmap, final, patchSize = 224 } = e.data
    if (!self.session || self.model?.modelName !== modelIdentifier) {
        self.session = createSession(modelIdentifier)
        if (!self.session) {
            self.postMessage({ success: false, error: `Model not found: ${modelIdentifier}` })
        }
    }
    try {
        const result = await generateEmbeddings(patchParams, bitmap)
        self.postMessage({ success: true, data: result, final })
        storeEmbeddings(imageSource, result)
    } catch (error) {
        console.log(eror)
        self.postMessage({ success: false, error: error.message })
    }
};