# Import necessary libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def solution_model():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    # Build a simple neural network with one Dense layer
    model = Sequential([
        Dense(1, input_shape=[1])
    ])

    # Compile the model with Mean Squared Error loss and Stochastic Gradient Descent optimizer
    model.compile(optimizer='sgd',
                  loss='mse')

    # Train the model for 1000 epochs
    model.fit(xs, ys, epochs=1000, verbose=0)
    
    # Evaluate the model's performance on the training data
    loss = model.evaluate(xs, ys)
    print(f'Loss MSE: {loss}')

    return model


if __name__ == '__main__':
    # Create and train the model
    model = solution_model()

    # Predict a value for a new input
    print(model.predict([10.0]))
    
    # Save the trained model
    model.save("model.h5")