import {MnistData} from "./data.js";

async function showExamples(data) {
    //create container in visor.
    const surface = tfvis.visor().surface({name: "Input Data Examples", tab: "Input Data"});

    //fetch examples.
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    //create a canvas element for each example.
    for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => {
            //reshape image to 28px x 28px.
            return examples.xs
                .slice([i, 0], [1, examples.xs.shape[1]])
                .reshape([28, 28, 1]);
        });

        const canvas = document.createElement("canvas");
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = "margin: 4px;";
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
    }
}

async function run() {
    const data = new MnistData();
    await data.load();
    await showExamples(data);
}

document.addEventListener("DOMContentLoaded", run);

function getModel() {
    const model = tf.sequential();

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    //first layer: specify input shape and convolution parameters.
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling"
    }));

    model.add(tf.layers.maxPooling2d({poolSize: [2,2], strides: [2,2]}));

    //repeat convolute and maxpool.
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
      }));
      model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

      //flatten output into 1D vector.
      model.add(tf.layers.flatten());

      //final layer has 10 output units correpsonding to a digit classification each.
      const NUM_OUTPUT_CLASSES = 10;
      model.add(tf.layers.dense({
          units: NUM_OUTPUT_CLASSES,
          kernelInitializer: "varianceScaling",
          activation: "softmax"
      }));

      const optimizer = tf.train.adam();
      model.compile({
          optimizer: optimizer,
          loss: "categoricalCrossentropy",
          metrics: ["accuracy"]
      });

      return model;
}