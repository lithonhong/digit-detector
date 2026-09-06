const canvas = document.getElementById('cv');
const ctx = canvas.getContext('2d');

const canvas2 = document.getElementById('cv2');
const ctx2 = canvas2.getContext('2d');

const detectBtn = document.getElementById('detect-btn');
const clearBtn = document.getElementById('clear-btn');
const resSpan = document.getElementById('detected-digit');

// miscellaneous functions

function getMousePos(event) {
    const rect = canvas.getBoundingClientRect();

    const canvasActualX = canvas.clientWidth;
    const canvasActualY = canvas.clientHeight;

    const canvasDisplayX = canvas.width;
    const canvasDisplayY = canvas.height;

    const x = (event.clientX - rect.left) * (canvasDisplayX / canvasActualX);
    const y = (event.clientY - rect.top) * (canvasDisplayY / canvasActualY);

    return [x, y];
}

const dotProduct = (a, b) => a.reduce((sum, val, i) => sum + val * b[i], 0);
const sigmoid = (z) => {
    return 1 / (1 + Math.exp(-z))};


// loading data

async function loadNeuralNetwork() {
    const response = await fetch('nnet.json');
    const data = await response.json();
    
    return [data.weights, data.biases];
}

let weights = new Array();
let biases = new Array();
let isNetworkLoaded = false;
loadNeuralNetwork().then(
    (data) => {
        weights = data[0];
        biases = data[1];
        isNetworkLoaded = true;
    }
);

// drawing on canvas

let isDrawing = false;
let lastX = 0;
let lastY = 0;
let mouseX = 0;
let mouseY = 0;

ctx.strokeStyle = '#ff0000';
ctx.lineWidth = 16;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = getMousePos(e);
}

function draw(e) {
    if (!isDrawing) {
        return;
    }

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    [mouseX, mouseY] = getMousePos(e);
    ctx.lineTo(mouseX, mouseY);
    ctx.stroke();

    [lastX, lastY] = getMousePos(e);
}

function stopDrawing(e) {
    isDrawing = false;
}

function clearCanvas(e) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();

    ctx2.clearRect(0, 0, canvas2.width, canvas2.height);
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing); 

clearBtn.addEventListener("click", clearCanvas);

// digit detector

function encodeCanvas() {
    canvasRGB = new Array(784);
    for (let x = 0; x < 28; x++) {
        for (let y = 0; y < 28; y++) {
            const pixel = ctx.getImageData(x * 8, y * 8, 8, 8);
            const a = pixel.data.filter((element, index) => index % 8 === 7);
            const a_sum = a.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
            canvasRGB[y * 28 + x] = a_sum / 255 / 32;
        }
    }
    return canvasRGB;
}

function getGuessedDigit(vct) {
    if (!isNetworkLoaded) {
        throw new Error("Data not loaded yet; please wait a moment.");
    }
    for (let i = 0; i < weights.length; i++) {
        vct = weights[i]
            .map((elem, idx) => dotProduct(elem, vct))
            .map((elem, idx) => elem + biases[0][i][0])
            .map((elem) => sigmoid(elem));
    }

    let mx = 0;
    let mxIdx = 0;

    for (let i = 0; i < vct.length; i++) {
        if (vct[i] > mx) {
            mx = vct[i];
            mxIdx = i;
        }
    }
    return mxIdx;
}

// dupe canvas

function setPixel(x, y, r, g, b, a, src) {
    const index = (x + y * 28) * 4;
    src[index]     = r;
    src[index + 1] = g;
    src[index + 2] = b;
    src[index + 3] = a;
}

function dupeCanvas(canv) {
    const cv2ImgData = ctx2.getImageData(0, 0, 28, 28);
    const cv2Data = cv2ImgData.data;

    for (let x = 0; x < 28; x++) {
        for (let y = 0; y < 28; y++) {
            setPixel(x, y, 0, 0, 0, canv[y*28+x] * 255, cv2Data);
        }
    }

    ctx2.putImageData(cv2ImgData, 0, 0);
}

function detectDigit(e){
    if (!isNetworkLoaded) {
        resSpan.innerText = "The network is not loaded yet."
        return
    }
    canvasVct = encodeCanvas();
    dupeCanvas(canvasRGB);
    console.log(canvasRGB)
    resSpan.innerText = getGuessedDigit(canvasVct);
}


detectBtn.addEventListener("click", detectDigit);