// ===============================
// Elements
// ===============================
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");

const startBtn = document.getElementById("startBtn");
const trainBtn = document.getElementById("trainBtn");
const saveBtn = document.getElementById("saveBtn");
const loadBtn = document.getElementById("loadBtn");
const resetBtn = document.getElementById("resetBtn");

// ===============================
// Variables
// ===============================
let mobilenet;  // Image feature extractor
let model;      // Image classifier

// ===============================
// Status helper (sets status message)
// ===============================
function setStatus(message) {
    statusText.innerHTML = `<div class="new">${message}</div>`;
}

// ===============================
// Init / Load the model
// ===============================
async function init() {

    // Set backend (use webgl to speed up processing)
    await tf.setBackend("webgl");

    // Load pretrained MobileNet
    const URL = "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json";
    const loaded = await tf.loadLayersModel(URL);

    // Remove classification head
    const layer = loaded.getLayer("global_average_pooling2d_1");
    mobilenet = tf.model({ inputs: loaded.inputs, outputs: layer.output });

    // Build classifier for MobileNet
    model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [1280], units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: 2, activation: "softmax" }));

    model.compile({
        optimizer: "adam",
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    // Set status and enable the start button
    setStatus("MobileNet loaded!");
    startBtn.disabled = false;

    // Enable load button if saved model exists
    if (localStorage.getItem("image-classifier-labels")) {
        loadBtn.disabled = false;
    }
}
// Call init function
init();


// ===============================
// Start the camera (when startBtn is clicked)
// ===============================
startBtn.onclick = startCamera;
// Function to start the camera
async function startCamera() {
    // Get camera stream
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    // Wait for video metadata to load
    video.addEventListener("loadedmetadata", () => {
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Set status and disable the start button and sample buttons
        setStatus("Camera started!");
        startBtn.disabled = true;
        document.querySelectorAll('.sampleBtn').forEach(btn => btn.disabled = false);

        // Start the detection loop
        runLoop();
    });
};


// ===============================
//  Variables for training
// ===============================
let trainingData = [];
let labels = [];
let labelMap = [];

// ===============================
// Data Preprocessing
// ===============================
// Function to extract features from the video frame
function getFeatures() {
    return tf.tidy(() => {
        const frame = tf.browser.fromPixels(video);
        const resized = tf.image.resizeBilinear(frame, [224, 224]);
        const normalized = resized.div(255);
        return mobilenet.predict(normalized.expandDims()).squeeze();
    });
}

// ===============================
// Add samples for training when a button with .sampleBtn is held down
// ===============================
document.querySelectorAll(".sampleBtn").forEach(btn => {
    btn.onmousedown = () => {
        // Capture a sample every 100ms
        btn.interval = setInterval(() => addSample(btn.dataset.label), 100);
    };
    // Stop capture samples when mouse is released
    btn.onmouseup = () => clearInterval(btn.interval);
    btn.onmouseleave = () => clearInterval(btn.interval);
});
// Function to add a training sample
function addSample(label) {
    const features = getFeatures();
    // Prepare the features for training
    trainingData.push(features);
    labels.push(label);
    // Update the status message
    setStatus(`Added: ${label}`);
    updateCounts();
    // Enable the train button
    trainBtn.disabled = false;
}

// Update sample counts on the buttons
function updateCounts() {
    document.querySelectorAll('button[data-label]').forEach(btn => {
        const label = btn.dataset.label;
        const count = labels.filter(l => l === label).length; // Get the current sample count for the label from the labels array
        btn.textContent = `Add ${label.charAt(0).toUpperCase() + label.slice(1)} (${count} Samples)`;
    });
}

// ===============================
// Train Model (when the #trainBtn button is clicked)
// ===============================
trainBtn.onclick = trainModel;

// Function to train the model
async function trainModel() {
    // Convert training data and labels to tensors
    const xs = tf.stack(trainingData);
    // Normalise the labels to the range [0, 1]
    labelMap = [...new Set(labels)];
    const ys = tf.tensor2d(
        labels.map(l => labelMap.map(x => x === l ? 1 : 0))
    );

    // Train the model (This is where the model learns from the training data)
    await model.fit(xs, ys, { epochs: 20 }); // This trains the model for 20 epochs
 
    // Set the status and enable the save button
    setStatus("Model trained!");
    saveBtn.disabled = false;

    // Clean up tensors
    xs.dispose();
    ys.dispose();
}

// ===============================
// Predict (make predictions and display the result)
// ===============================
// Function to predict the class of an image
function predictClass() {
    // Check if the model is loaded
    if (!model) return null;

    // Get the features from the current video frame
    const features = getFeatures();
    // Make a prediction
    const prediction = model.predict(features.expandDims());
    // Get the prediction data
    const data = prediction.dataSync();
    // Find the index of the highest probability
    const index = data.indexOf(Math.max(...data));
    // Map the index to the corresponding label
    const result = {
        label: labelMap[index],
        confidence: Math.round(data[index] * 100)
    };

    // Clean up tensors
    features.dispose();
    prediction.dispose();

    return result;
}

// Function to draw the video frame and predictions
function draw() {
    
    // Save the current canvas state
    ctx.save();

    // Flip the canvas horizontally (for mirrored camera)
    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Draw the video frame
    ctx.drawImage(video, 0, 0);

    // Check if the model is loaded
    if (model) {
        // Make a prediction
        const prediction = predictClass();

        // Check if the prediction is valid and display it
        if (prediction && prediction.label !== undefined) {
            // Draw label (flipped)
            ctx.save();
            ctx.scale(-1, 1);
            ctx.fillStyle = "#E31C79";
            ctx.font = "30px Arial";
            ctx.fillText(
                prediction.label + " (" + prediction.confidence + "%)",
                10 - canvas.width, // Place label on the left side of the canvas (flipped)
                40 
            );
            ctx.restore();
        }
    }
    ctx.restore();
}

// Main loop to continuously draw the video and make predictions
async function runLoop() {
    draw();
    requestAnimationFrame(runLoop);
}

// ===============================
// Save Model (when saveBtn is clicked)
// ===============================
saveBtn.onclick = saveModel;

// Function to save model to the browser's local storage
async function saveModel() {
    // Save the model in local storage
    await model.save("localstorage://image-classifier-model");

    // Save the label map in local storage
    localStorage.setItem("image-classifier-labels", JSON.stringify(labelMap));

    // Set status message and enable the load button
    setStatus("Model saved!");
    loadBtn.disabled = false;
}

// ===============================
// Load Model (when loadModelBtn is clicked)
// ===============================
loadBtn.onclick = loadModel;

// Function to load model from browser's local storage
async function loadModel() {
    // Load the model from local storage
    model = await tf.loadLayersModel("localstorage://image-classifier-model");

    // Load the label map from local storage
    labelMap = JSON.parse(localStorage.getItem("image-classifier-labels")) || [];

    // Set status message
    setStatus("Model loaded!");
}

// ===============================
// Reset the model and UI (when resetBtn is clicked)
// ===============================
resetBtn.onclick = resetModel;

// Function to reset the model and UI
async function resetModel() {

    // Reset the model data and labels
    trainingData.forEach(t => t.dispose());
    trainingData = [];
    labels = [];
    labelMap = [];
    // Remove model from local storage if one exists
    try {
        await tf.io.removeModel("localstorage://image-classifier-model");
        localStorage.removeItem("image-classifier-labels");
    } catch (e) {}
    // Reset the UI and buttons
    trainBtn.disabled = true;
    saveBtn.disabled = true;
    loadBtn.disabled = true;
    updateCounts();
    // Set status message
    setStatus("Model reset!");
}