import 'fast-text-encoding';
import * as tf from '@tensorflow/tfjs';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';

setWasmPaths('./');
await tf.setBackend('wasm');
await tf.ready();

console.log(" Training house price prediction model...");

// 1. Raw mixed dataset
const rawData = [
  [3974, 3, "Countryside", 256730],
  [1660, 5, "Countryside", 970910],
  [2094, 2, "Downtown",    484681],
  [1930, 4, "Suburb",      249503],
  [1895, 3, "Suburb",      310000],
  [1200, 2, "Downtown",    198000],
  [3400, 4, "Countryside", 610000],
  [1750, 3, "Suburb",      330000],
];

// 2. Encode location (manual one-hot)
const locationMap = {
  Countryside: [1, 0, 0],
  Downtown:    [0, 1, 0],
  Suburb:      [0, 0, 1]
};

// 3. Extract features and labels
const inputs = [];
const prices = [];

rawData.forEach(entry => {
  const [size, bedrooms, location, price] = entry;
  const locationEncoded = locationMap[location];
  inputs.push([size, bedrooms, ...locationEncoded]);
  prices.push([price]);
});

// 4. Normalize numerical features (size and bedrooms only)
const featureTensor = tf.tensor2d(inputs);
const numFeatures = featureTensor.slice([0, 0], [-1, 2]); // size, bedrooms
const catFeatures = featureTensor.slice([0, 2], [-1, -1]); // one-hot locations

const min = numFeatures.min(0);
const max = numFeatures.max(0);
const normNum = numFeatures.sub(min).div(max.sub(min));
const finalFeatures = normNum.concat(catFeatures, 1);

const labelTensor = tf.tensor2d(prices);

// 5. Build and train the model
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [finalFeatures.shape[1]], units: 12, activation: 'relu' }));
model.add(tf.layers.dense({ units: 6, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1 }));

model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

await model.fit(finalFeatures, labelTensor, {
  epochs: 200,
  batchSize: 1,
  verbose: 1
});

// 6. Predict a new example
const newHouse = [2100, 3, ...locationMap["Suburb"]]; // 2100 sqft, 3 bed, Suburb
const newTensor = tf.tensor2d([newHouse]);
const normNewNum = newTensor.slice([0, 0], [-1, 2]).sub(min).div(max.sub(min));
const newCat = newTensor.slice([0, 2], [-1, -1]);
const finalNewInput = normNewNum.concat(newCat, 1);

const prediction = model.predict(finalNewInput);
prediction.print();

prediction.array().then(arr => {
  console.log(` Predicted price: $${Math.round(arr[0][0])}`);
});
