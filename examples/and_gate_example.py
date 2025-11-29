"""
AND Gate Example
"""

from neural_network import NeuralNetwork


def main():
    print("=" * 50)
    print("Training Neural Network for AND Gate")
    print("=" * 50)
    
    # Create network
    nn = NeuralNetwork([2, 2, 'sigmoid', 1, 'sigmoid'], loss='bce', learning_rate=0.5)
    
    # Training data
    X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [0, 0, 0, 1]
    
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
    for X, y in zip(X_train, y_train):
        pred = nn.predict(X)
        print(f"{X} AND = {y} (predicted: {pred:.2f})")


if __name__ == "__main__":
    main()
