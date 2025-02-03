# Linear Regression with Neural Networks

This project demonstrates a simple linear regression using a neural network built with TensorFlow and Keras. The goal is to train a model that accurately maps input values to their corresponding outputs, making it capable of predicting unseen data with minimal error.

## Project Overview

The task involves training a neural network on a straightforward dataset where the relationship between inputs (`xs`) and outputs (`ys`) is linear. The model aims to achieve a Mean Squared Error (MSE) of less than `1e-05` when predicting new data points.

## Dataset

The dataset consists of six data points with a linear relationship:

```
Inputs (xs):  [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
Outputs (ys): [ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
```

## Model Architecture

The model is a simple Sequential neural network with a single Dense layer. Here's a breakdown of the architecture:

- **Input Layer:** Accepts a single float value.
- **Dense Layer:** One neuron with a linear activation function.

### Model Compilation

- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Loss Function:** Mean Squared Error (MSE)

### Training

The model is trained for 1000 epochs to ensure convergence to a low error rate.

## How to Run

1. **Install Dependencies:**\
   Ensure you have TensorFlow installed:

   ```bash
   pip install tensorflow numpy
   ```

2. **Run the Script:**\
   Execute the Python script to train the model and make predictions:

   ```bash
   python nn_Linear_Regression.py
   ```

3. **Prediction Example:**\
   After training, the model will predict the output for an input value of `10.0`.

   ```
   [[10.999998]]
   ```

4. **Model Saving:**\
   The trained model is saved as `model.h5` for future use.

## Output

- **Loss MSE:** The final loss is printed after training.
- **Prediction:** The model predicts new values with high accuracy.

## License

This project is open-source and available under the MIT License.

---

Feel free to fork this repository, make improvements, and submit pull requests!

---
