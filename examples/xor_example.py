"""
XOR Problem Example
"""

from neural_network import NeuralNetwork


def main():
    print("=" * 50)
    print("Training Neural Network for XOR Problem")
    print("=" * 50)
    
    # Create network
    nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='mse', learning_rate=0.5)
    
    # Training data
    X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [0, 1, 1, 0]
    
    # Training
    print("\nTraining...")
    for epoch in range(5000):
        total_loss = 0
        for X, y in zip(X_train, y_train):
            output, history = nn.forward(X)
            loss = nn.function_list[nn.funct_list["loss"]][0](y, output)
            total_loss += loss
            nn.backward(y, output, history)
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Avg Loss: {total_loss / len(X_train):.4f}")
    
    # Testing
    print("\n" + "=" * 50)
    print("Testing...")
    print("=" * 50)
    for X, y in zip(X_train, y_train):
        pred = nn.predict(X)
        print(f"Input: {X}, True: {y}, Predicted: {pred:.4f}")


if __name__ == "__main__":
    main()
