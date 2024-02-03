

const STATUS = document.getElementById('status');
const VIDEO = document.getElementById('webcam');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];
const EXPORT_BUTTON = document.getElementById('exportModel');



EXPORT_BUTTON.addEventListener('click', exportModel);
ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

// Just add more buttons in HTML to allow classification of more classes of data!
let dataCollectorButtons = document.querySelectorAll('button.dataCollector');
for (let i = 0; i < dataCollectorButtons.length; i++) {
  dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
  // For mobile.
  dataCollectorButtons[i].addEventListener('touchend', gatherDataForClass);

  // Populate the human readable names for classes.
  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}


let mobilenet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;
let mobileNetBase = undefined;




function customPrint(line) {

  let p = document.createElement('p');
  p.innerText = line;
  document.body.appendChild(p);

}


/**
 * Loads the MobileNet model and warms it up so ready for use.
 **/
async function loadMobileNetFeatureModel() {

  const URL = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json';

  mobilenet = await tf.loadLayersModel(URL);

  STATUS.innerText = 'MobileNet v2 loaded successfully!';

  mobilenet.summary(null, null, customPrint);

  

  const layer = mobilenet.getLayer('global_average_pooling2d_1');

  mobileNetBase = tf.model({inputs: mobilenet.inputs, outputs: layer.output}); 

  mobileNetBase.summary();

  

  // Warm up the model by passing zeros through it once.

  tf.tidy(function () {

    let answer = mobileNetBase.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));

    console.log(answer.shape);

  });

}
loadMobileNetFeatureModel();


let model = tf.sequential();

model.add(tf.layers.dense({inputShape: [1280], units: 64, activation: 'relu'}));

model.add(tf.layers.dense({units: CLASS_NAMES.length, activation: 'softmax'}));

model.summary();

// Compile the model with the defined optimizer and specify a loss function to use.
model.compile({
  // Adam changes the learning rate over time which is useful.
  optimizer: 'adam',
  // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
  // Else categoricalCrossentropy is used if more than 2 classes.
  loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy': 'categoricalCrossentropy', 
  // As this is a classification problem you can record accuracy in the logs too!
  metrics: ['accuracy']  
});


/**
 * Check if getUserMedia is supported for webcam access.
 **/
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}


/**
 * Enable the webcam with video constraints applied.
 **/
function enableCam() {
  if (hasGetUserMedia()) {
    // getUsermedia parameters.
    const constraints = {
      video: true,
      width: 640, 
      height: 480 
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
      VIDEO.srcObject = stream;
      VIDEO.addEventListener('loadeddata', function() {
        videoPlaying = true;
        ENABLE_CAM_BUTTON.classList.add('removed');
      });
    });
  } else {
    console.warn('getUserMedia() is not supported by your browser');
  }
}


/**
 * Handle Data Gather for button mouseup/mousedown.
 **/
function gatherDataForClass() {
  let classNumber = parseInt(this.getAttribute('data-1hot'));
  gatherDataState = (gatherDataState === STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;
  dataGatherLoop();
}


function calculateFeaturesOnCurrentFrame() {

  return tf.tidy(function() {

    // Grab pixels from current VIDEO frame.

    let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);

    // Resize video frame tensor to be 224 x 224 pixels which is needed by MobileNet for input.

    let resizedTensorFrame = tf.image.resizeBilinear(

        videoFrameAsTensor, 

        [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],

        true

    );

 

    let normalizedTensorFrame = resizedTensorFrame.div(255);

 

    return mobileNetBase.predict(normalizedTensorFrame.expandDims()).squeeze();

  });

}

/**
 * When a button used to gather data is pressed, record feature vectors along with class type to arrays.
 **/
function dataGatherLoop() {
  // Only gather data if webcam is on and a relevent button is pressed.
  if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
    // Ensure tensors are cleaned up.
    let imageFeatures = calculateFeaturesOnCurrentFrame();

    trainingDataInputs.push(imageFeatures);
    trainingDataOutputs.push(gatherDataState);
    
    // Intialize array index element if currently undefined.
    if (examplesCount[gatherDataState] === undefined) {
      examplesCount[gatherDataState] = 0;
    }
    // Increment counts of examples for user interface to show.
    examplesCount[gatherDataState]++;

    STATUS.innerText = '';
    for (let n = 0; n < CLASS_NAMES.length; n++) {
      STATUS.innerText += CLASS_NAMES[n] + ' data count: ' + examplesCount[n] + '. ';
    }

    window.requestAnimationFrame(dataGatherLoop);
  }
}


/**
 * Once data collected actually perform the transfer learning.
 **/
async function trainAndPredict() {

  predict = false;

  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

 

  let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');

  let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);

  let inputsAsTensor = tf.stack(trainingDataInputs);

  

  let results = await model.fit(inputsAsTensor, oneHotOutputs, {

    shuffle: true,

    batchSize: 5,

    epochs: 5,

    callbacks: {onEpochEnd: logProgress}

  });
  
  outputsAsTensor.dispose();

  oneHotOutputs.dispose();

  inputsAsTensor.dispose();

  predict = true;

  

  // Make combined model for download.

  let combinedModel = tf.sequential();

  combinedModel.add(mobileNetBase);

  combinedModel.add(model);

  

  combinedModel.compile({

    optimizer: 'adam',

    loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy': 'categoricalCrossentropy'

  });

  

  combinedModel.summary();

  await combinedModel.save('downloads://my-model');

  predictLoop();

}

 

  

  


/**
 * Log training progress.
 **/
function logProgress(epoch, logs) {
  console.log('Data for epoch ' + epoch, logs);
}


/**
 *  Make live predictions from webcam once trained.
 **/
function predictLoop() {
  if (predict) {
    tf.tidy(function() {
      let imageFeatures = calculateFeaturesOnCurrentFrame();
      let prediction = model.predict(imageFeatures.expandDims()).squeeze();
      let highestIndex = prediction.argMax().arraySync();
      let predictionArray = prediction.arraySync();
      STATUS.innerText = 'Prediction: ' + CLASS_NAMES[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence';
    });

    window.requestAnimationFrame(predictLoop);
  }
}


async function exportModel() {
  // Convert the model to JSON format.
  const modelJson = model.toJSON();

  // Convert the JSON to a Blob for download.
  const modelBlob = new Blob([JSON.stringify(modelJson)], { type: 'application/json' });

  // Create a download link and trigger the download.
  const downloadLink = document.createElement('a');
  downloadLink.href = URL.createObjectURL(modelBlob);
  downloadLink.download = 'trained_model.json';
  downloadLink.click();
}

/**
 * Purge data and start over. Note this does not dispose of the loaded 
 * MobileNet model and MLP head tensors as you will need to reuse 
 * them to train a new model.
 **/
function reset() {
  predict = false;
  examplesCount.splice(0);
  for (let i = 0; i < trainingDataInputs.length; i++) {
    trainingDataInputs[i].dispose();
  }
  trainingDataInputs.splice(0);
  trainingDataOutputs.splice(0);
  STATUS.innerText = 'No data collected';
  
  console.log('Tensors in memory: ' + tf.memory().numTensors);
}
