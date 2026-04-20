// ===============================
// Elements
// ===============================
const output = document.getElementById("output");
const statusText = document.getElementById("status");

const startBtn = document.getElementById("startBtn");
const trainBtn = document.getElementById("trainBtn");
const saveBtn = document.getElementById("saveBtn");
const loadBtn = document.getElementById("loadBtn");
const resetBtn = document.getElementById("resetBtn");

// ===============================
// Variables
// ===============================
let recognizer;
let model;
let examples = [];
let trainingData = [];
let labels = [];
let labelMap = [];
let mode = "idle"; // "collect", "predict"
let currentLabel = null;

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

    // Create speech recogniser from TensorFlow.js
    recognizer = speechCommands.create("BROWSER_FFT");
    await recognizer.ensureModelLoaded();

    // Set status and enable the start button
    setStatus("Audio model ready!");
    startBtn.disabled = false;

    // Enable load button if saved model exists
    if (localStorage.getItem("audio-labels")) {
        loadBtn.disabled = false;
    }
}
// Call init function
init();

// ===============================
// Start Microphone (when startBtn is clicked)
// ===============================
startBtn.onclick = startMicrophone;
// Function to start the microphone
async function startMicrophone() {
    // Start listening to the microphone and process the audio results
    await recognizer.listen(result => {

        // Get the scores and words from the recogniser
        const scores = result.scores;
        const words = recognizer.wordLabels();

        // ======================
        // Collect training data
        // ======================
        if (mode === "collect" && currentLabel) {
            setStatus(`Recording: ${currentLabel}`);
            // Add the current audio sample to the training data
            addSample(scores, currentLabel);
        }

        // ======================
        // Prediction
        // ======================
        if (mode === "predict") {
            
            // If no trained model, use the pretrained model
            if (!model) {
                setStatus("Using pretrained speech model...");
                // Create a mapping of scores to labels
                const scoresWithLabels = Array.from(scores).map((s, i) => ({
                    score: s,
                    label: words[i]
                }));
                // Sort scores with labels
                scoresWithLabels.sort((a, b) => b.score - a.score);
                // Display the highest score label
                output.innerText =
                    `${scoresWithLabels[0].label} (${Math.round(scoresWithLabels[0].score * 100)}%)`;
            } else {
                // Use the trained or loaded model

                // Create a tensor from the scores
                const input = tf.tensor([scores]);
                // Make a prediction
                const prediction = model.predict(input);
                // Get the prediction data
                const data = prediction.dataSync();
                // Find the index of the highest score
                const index = data.indexOf(Math.max(...data));
                // Display the predicted label
                output.innerText =
                    `${labelMap[index]} (${Math.round(data[index] * 100)}%)`;
                // Dispose of the tensors
                input.dispose();
                prediction.dispose();
            }
            
        }

    }, {
        probabilityThreshold: 0.5 // Set to 0 to include all predictions
    });

    // Set status and enable buttons
    setStatus("Microphone started!");
    startBtn.disabled = true;
    document.querySelectorAll('.sampleBtn').forEach(btn => btn.disabled = false);
};

// ===============================
// Add samples for training when a button with .sampleBtn is clicked
// ===============================
document.querySelectorAll(".sampleBtn").forEach(btn => {
    // When the button is pressed change mode and set current label
    btn.onmousedown = () => {
        mode = "collect";
        currentLabel = btn.dataset.label;
    };
    // When the button is released change mode and clear current label
    btn.onmouseup = () => {
        mode = "idle";
        currentLabel = null;
    };
    btn.onmouseleave = () => {
        mode = "idle";
        currentLabel = null;
    };

});

// Function to add a training sample
function addSample(scores, label) {
    // Add the scores and label to the training data
    trainingData.push(Array.from(scores));
    labels.push(label);
    // Update status and counts
    setStatus(`Added: ${label}`);
    updateCounts();
    // Enable training button
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

    const xs = tf.stack(trainingData.map(d => tf.tensor(d)));

    labelMap = [...new Set(labels)];

    const ys = tf.tensor2d(
        labels.map(l => labelMap.map(x => x === l ? 1 : 0))
    );

    model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [xs.shape[1]],
        units: 64,
        activation: "relu"
    }));

    model.add(tf.layers.dense({
        units: labelMap.length,
        activation: "softmax"
    }));

    model.compile({
        optimizer: "adam",
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    await model.fit(xs, ys, { epochs: 20 });

    setStatus("Model trained!");
    saveBtn.disabled = false;

    xs.dispose();
    ys.dispose();
    
    mode = "predict";
    setStatus("Model trained! Listening for predictions...");
}

// ===============================
// Save Model (when saveBtn is clicked)
// ===============================
saveBtn.onclick = saveModel;

    // Function to save model to the browser's local storage
    async function saveModel() {
    // Save the model in local storage
    await model.save("localstorage://audio-model");

    // Save the label map in local storage
    localStorage.setItem("audio-labels", JSON.stringify(labelMap));

    // Set status message and enable the load button
    setStatus("Model saved!");
    loadBtn.disabled = false;
};

// ===============================
// Load Model (when loadModelBtn is clicked)
// ===============================
loadBtn.onclick = loadModel;

// Function to load model from browser's local storage
async function loadModel() {
    // Load the model from local storage
    model = await tf.loadLayersModel("localstorage://audio-model");

    // Load the label map from local storage
    labelMap = JSON.parse(localStorage.getItem("audio-labels")) || [];

    // Set mode to predict
    mode = "predict";

    // Set status message
    setStatus("Model loaded!");
};

// ===============================
// Reset the model and UI (when resetBtn is clicked)
// ===============================
resetBtn.onclick = resetModel;

// Function to reset the model and UI
async function resetModel() {

    // Reset the model data and labels
    examples = [];
    trainingData = [];
    labels = [];
    labelMap = [];
    model = null;

    // Remove model from local storage if one exists
    try {
        await tf.io.removeModel("localstorage://audio-model");
        localStorage.removeItem("audio-labels");
    } catch (e) {}

    // Reset the UI and buttons
    output.innerText = "Waiting...";
    setStatus("Reset complete");

    trainBtn.disabled = true;
    saveBtn.disabled = true;
    loadBtn.disabled = true;

    // Update Sample Buttons Counts
    updateCounts();
};