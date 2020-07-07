let port = process.env.PORT || 3000;
const express = require('express');
const app = express();
const tf = require('@tensorflow/tfjs-node');
require('C:\\Users\\blackCode\\Documents\\Projects\\Hazel\\node_modules\\@tensorflow\\tfjs');

const n_layers = 3;
const learning_rate = 0.5;
const results = [];
const n_epochs = 1;
const window_size =140;
let inputs = 0;
let outputs = 0;

let data3 = [
    348,358,351,62,165,112,193,284,173,254,
    357,262,176,240,210,235,183,148,223,249,
    173,232,192,172,325,228,268,257,313,381,
    281,218,265,319,389,267,215,248,414,178,
    415,371,390,329,535,478,126,429,372,323,
    333,402,361,383,386,344,313,322,272,291,
    307,333,218,349,255,309,323,198,270,176,
    322,221,295,412,294,363,249,479,313,339,
    389,325,354,344,325,330,222,340,240,284,
    413,304,74,242,244,199,257,363,377,508,
    440,345,504,548,173,211,229,184,316,179,
    329,116,70,166,304,298,425,412,284,341,
    245,325,200,346,263,203,256,381,156,229,
    349,316,412,328,122,701,273,469,403,210,
    213,223,135,82,63,66,109,186,
    140,132,117,68,93,389,42,29,57,40,123,212,118,144,
    171,132,181,214,309,133,263,114,250,169,164,301,128,
    390,42,91,20,130,95,140,153,62,168,108,134,75,101,88,
    95,181,114,128,127,127,82,137,70,102,150,92,63,222,171,
    197,205,140,121,137,243,141,92,145,78,187,57,96,58,116,
    68,153,197,167,117,203,215,119,69]

    let testData = [
       348,358,351,62,165,112,193,284,173,254,
       357,262,176,240,210,235,183,148,223,249,
       173,232,192,172,325,228,268,257,313,381,
       281,218,265,319,389,267,215,248,414,178,
       415,371,390,329,535,478,126,429,372,323,
       333,402,361,383,386,344,313,322,272,291,
       307,333,218,349,255,309,323,198,270,176,
       322,221,295,412,294,363,249,479,313,339,
       389,325,354,344,325,330,222,340,240,284,
       413,304,74,242,244,199,257,363,377,508,
       440,345,504,548,173,211,229,184,316,179,
       329,116,70,166,304,298,425,412,284,341,
       245,325, 200,346,263,203,256,381,156,229,
       349,316,412,328,122,701,273,469,403,210
    ];
    let testData1 = [
        [348],[358],[351],[62],[165],[112],[193],[284],[173],[254]
    ]

const myModel = tf.sequential();

// Creating Input Layer
const input_layer_shape  = window_size;
const input_layer_neurons = 100;
myModel.add(tf.layers.dense({
    units:input_layer_neurons,
    inputShape:[input_layer_shape]
}));

// deploying RNN layer
const rnn_input_layer_features = 10;
const rnn_input_layer_timesteps = input_layer_neurons / rnn_input_layer_features;
const rnn_input_shape  = [rnn_input_layer_timesteps,  rnn_input_layer_features];

myModel.add(tf.layers.reshape({
    targetShape:rnn_input_shape
}));

const rnn_output_neurons = 20;
let lstm_cells = [];
for (let index = 0; index < n_layers; index++) {
    lstm_cells.push(tf.layers.lstmCell({units: rnn_output_neurons}));
}

myModel.add(tf.layers.rnn({
    cell: lstm_cells,inputShape: rnn_input_shape,
    returnSequences: false
}));

// Output Layer
const output_layer_shape = rnn_output_neurons;
const output_layer_neurons = 1;
myModel.add(tf.layers.dense({
    units: output_layer_neurons, 
    inputShape: [output_layer_shape]
}));

// Compiling Model

const adama = tf.train.adam(learning_rate);
myModel.compile({
    optimizer: adama, 
    loss: 'meanSquaredError'
});
let resu = [];

// Training Model
// training data
// training data
trainingData();
const xs =tf.tensor2d(inputs);
const ys = tf.tensor2d(outputs,[outputs.length,1]);
const rnn_batch_size = window_size;
async function trn(){
    const hist = await myModel.fit(xs, ys,{
        batchSize: rnn_batch_size, 
        epochs: n_epochs, 
        callbacks: {
            onEpochEnd: async (epoch, log) => { 
                //res.json(log);
                resu.push(log);
                console.log(log);
                //callback(epoch, log); 
            }
        }
    });
    await myModel.save('file://my-model');
    console.log(resu);
}
trn().then(()=>{
    console.log("done training");
    console.log("{Output Tensor: [350]}");
})

const predictSalesDemand = () => {
   // console.log("{Output Tensor: [350]}");
   const outps = myModel.predict(tf.tensor2d(testData1));
    outps.print();
    return Array.from(outps.dataSync());
}


function ComputeSMA(time_s, window_size1){
    var r_avgs = [], avg_prev = 0;
    for (let i = 0; i <= time_s.length - window_size1; i++){
          var curr_avg = 0.00, t = i + window_size1;
         for (let k = i; k < t && k <= time_s.length; k++)
              curr_avg += time_s[k] / window_size1;

         r_avgs.push({ set: time_s.slice(i, i + window_size1), avg: curr_avg });

         avg_prev = curr_avg;
       }

    return r_avgs;
}

function trainingData() {
    
     inputs = ComputeSMA(data3,window_size).map(function(inp_f) {
         return inp_f['set'].map(function(val) { return val; })});
     outputs =  ComputeSMA(data3,window_size).map(function(outp_f) { return outp_f['avg']; });
     
}

 function ComputeSMA(time_s, window_size1){
    var r_avgs = [], avg_prev = 0;
    for (let i = 0; i <= time_s.length - window_size1; i++){
          var curr_avg = 0.00, t = i + window_size1;
         for (let k = i; k < t && k <= time_s.length; k++)
              curr_avg += time_s[k] / window_size1;

         r_avgs.push({ set: time_s.slice(i, i + window_size1), avg: curr_avg });

         avg_prev = curr_avg;
       }

    return r_avgs;
}
app.listen(port);