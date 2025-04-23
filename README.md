# 🧠 House Price Prediction with TensorFlow.js + WebAssembly

This project demonstrates a lightweight, browser-compatible machine learning model that predicts house prices using `TensorFlow.js` with the WebAssembly (WASM) backend.

> ✅ No external dependencies, TypeScript, or Node.js — just plain JavaScript, `@tensorflow/tfjs`, and WASM!

---

## 🚀 What It Does

- Trains a neural network to predict house prices based on:
  - 🏠 House size (sqft)
  - 🛏 Number of bedrooms
  - 📍 Location (one-hot encoded)

- Uses the `@tensorflow/tfjs-backend-wasm` for optimal speed in environments like [GraalVM](https://www.graalvm.org/) or in-browser execution.

---

## 📦 Tech Stack

- `@tensorflow/tfjs`
- `@tensorflow/tfjs-backend-wasm`
- `fast-text-encoding` (polyfill for TextEncoder/TextDecoder)

---

## 🛠️ How It Works

1. **Feature Preparation**
   - Extract numerical and categorical values from an array.
   - Apply **min-max normalization** to numerical features.
   - Encode categorical features (locations) using **one-hot encoding**.

2. **Model Architecture**
   - `Dense(12, relu)`
   - `Dense(6, relu)`
   - `Dense(1)` → final house price prediction.

3. **Training**
   - Trained over 200 epochs using `adam` optimizer and `meanSquaredError` loss.

4. **Prediction**
   - You can supply new house data (e.g., `[2100 sqft, 3 bedrooms, Suburb]`) and the model will estimate its price.

---

## Run the Application

To start the demo, simply run:

```bash
./mvnw clean package
```

```bash
./mvnw exec:java
```
