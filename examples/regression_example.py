"""
Simple Linear Regression Example
y = 2x + 1
"""

from neural_network import NeuralNetwork


def main():
    print("=" * 50)
    print("Training Neural Network for Linear Regression")
    print("Function: y = 2x + 1")
    print("=" * 50)
    
    # Create network
    nn = NeuralNetwork([1, 5, 'relu', 1], loss='mse', learning_rate=0.01)
    
    # Training data
    X_train = [[0], [1], [2], [3], [4]]
    y_train = [1, 3, 5, 7, 9]
    
    # Training
    print("\nTraining...")
    for epoch in range(2000):
        total_loss = 0
        for X, y in zip(X_train, y_train):
            output, history = nn.forward(X)
            loss = nn.function_list[nn.funct_list["loss"]][0](y, output)
            total_loss += loss
            nn.backward(y, output, history)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Avg Loss: {total_loss / len(X_train):.4f}")
    
    # Testing
    print("\n" + "=" * 50)
    print("Testing...")
    print("=" * 50)
    test_X = [[5], [6], [7], [10]]
    for X in test_X:
        pred = nn.predict(X)
        true_y = 2 * X[0] + 1
        print(f"X={X[0]}, True={true_y}, Predicted={pred:.2f}")


if __name__ == "__main__":
    main()
