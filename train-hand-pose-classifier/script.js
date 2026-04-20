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
let detector;        // Hand detector
let lastHand = null; // Last detected hand

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

    // Load the hand pose detection model
    detector = await handPoseDetection.createDetector(
        handPoseDetection.SupportedModels.MediaPipeHands,
        {
            runtime: "tfjs",
            modelType: "full",
            maxHands: 1
        }
    );

    // Set status and enable the start button
    setStatus("Hand model loaded!");
    startBtn.disabled = false;

    // Enable load button if saved model exists
    if (localStorage.getItem("hand-labels")) {
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
    // Camera flipped/mirror
    video.style.transform = "scaleX(-1)";

    // Wait for video metadata to load
    video.addEventListener("loadedmetadata", () => {
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Set status and disable the start button and sample buttons
        setStatus("Camera started!");
        startBtn.disabled = true;
        document.querySelectorAll(".sampleBtn").forEach(btn => btn.disabled = false);

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
let model;

// ===============================
// Data Preprocessing
// ===============================
// Convert keypoints to array
function keypointsToArray(keypoints) {
    // Use wrist as reference point
    const wrist = keypoints[0];
    return keypoints.flatMap(kp => [
        (kp.x - wrist.x) / video.videoWidth,
        (kp.y - wrist.y) / video.videoHeight
    ]);
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
    if (!lastHand) return;
    // Prepare the training sample with the last detected pose keypoints normalised and the provided label
    trainingData.push(keypointsToArray(lastHand.keypoints));
    labels.push(label);
    // Update the sample counts and status
    setStatus(`Added: ${label}`);
    updateCounts();
    // Enable the train button
    trainBtn.disabled = false;
}

// Update sample counts on the buttons
function updateCounts() {
    document.querySelectorAll("[data-label]").forEach(btn => {
        const label = btn.dataset.label;
        const count = labels.filter(l => l === label).length;
        btn.textContent = `Add ${label} (${count})`;
    });
}

// ===============================
// Train Model (when the #trainBtn button is clicked)
// ===============================
trainBtn.onclick = trainModel;
// Function to train the model
async function trainModel() {
    // Convert training data and labels to tensors
    const xs = tf.tensor2d(trainingData);
    // Normalize the labels to the range [0, 1]
    labelMap = [...new Set(labels)];
    const ys = tf.tensor2d(
        labels.map(l => labelMap.map(x => x === l ? 1 : 0))
    );
    // Define the model architecture
    model = tf.sequential(); // This creates a new sequential model (a sequential model is a linear stack of layers)
    // Input layer with 64 units and ReLU activation (This layer learns to extract features from the input)
    model.add(tf.layers.dense({ inputShape: [xs.shape[1]], units: 64, activation: "relu" }));
    // Hidden layer with 32 units and ReLU activation (This layer learns to combine features)
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    // Output layer with units equal to the number of classes and softmax activation (This layer produces the final class probabilities)
    model.add(tf.layers.dense({ units: labelMap.length, activation: "softmax" }));

    // Compile the model (This configures the model for training)
    model.compile({
        optimizer: "adam", // This is the optimization algorithm used to minimize the loss function
        loss: "categoricalCrossentropy", // This is the loss function used for multi-class classification
        metrics: ["accuracy"] // This measures the accuracy of the model during training
    });

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
// Function to predict the class of a hand pose
function predictClass(keypoints) {
     // Check if the model is loaded
    if (!model) return null;

    // Prepare the input tensor
    const input = tf.tensor2d([keypointsToArray(keypoints)]);
    // Make a prediction
    const prediction = model.predict(input);
    // Get the prediction data
    const data = prediction.dataSync();
    // Find the class with the highest probability
    const index = data.indexOf(Math.max(...data));
    // Map the index to the corresponding label
    const result = {
        label: labelMap[index],
        confidence: Math.round(data[index] * 100)
    };

    // Dispose of the tensors
    input.dispose();
    prediction.dispose();

    return result;
}

// Function to draw the hand keypoints on the canvas
function drawHand(hand) {
    // Save the current context state
    ctx.save();

    // Flip the canvas horizontally
    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0);

    // If hand keypoints are detected
    if (hand) {
        // Draw each keypoint
        hand.keypoints.forEach(kp => {
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, 5, 0, Math.PI * 2);
            ctx.fillStyle = "#E31C79";
            ctx.fill();
        });

        // If a model is loaded
        if (model) {
            // Make a prediction
            const prediction = predictClass(hand.keypoints);

            // Draw the prediction
            ctx.scale(-1, 1);
            ctx.font = "30px Arial";
            ctx.fillText(prediction.label + " (" + prediction.confidence + "%)", -canvas.width + 10, 40);
        }
    }
    // Restore the context
    ctx.restore();
}

// Main loop to estimate poses and draw them on the canvas
async function runLoop() {
    // Estimate hand poses from the video
    const hands = await detector.estimateHands(video);
    // Check if any hands are detected and draw the first detected hand
    if (hands.length > 0) {
        lastHand = hands[0];
    } else {
        lastHand = null;
    }
    drawHand(lastHand);

    // This continues the detection loop
    requestAnimationFrame(runLoop);
}

// ===============================
// Save Model (when saveBtn is clicked)
// ===============================
saveBtn.onclick = saveModel;

// Function to save model to the browser's local storage
async function saveModel() {
    // Save the model in local storage
    await model.save("localstorage://hand-model");

    // Save the label map in local storage
    localStorage.setItem("hand-labels", JSON.stringify(labelMap));

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
    model = await tf.loadLayersModel("localstorage://hand-model");

    // Load the label map from local storage
    labelMap = JSON.parse(localStorage.getItem("hand-labels")) || [];

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
    model = null;
    labelMap = [];
    trainingData = [];
    labels = [];
    // Remove model from local storage if one exists
    try {
        await tf.io.removeModel("localstorage://hand-model");
        localStorage.removeItem("hand-labels");
    } catch (e) {}
    // Reset the UI and buttons
    saveBtn.disabled = true;
    loadBtn.disabled = true;
    updateCounts();
    // Set status message
    setStatus("Model reset!");
}
