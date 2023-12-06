import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation_function(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error

# OR 게이트의 입력과 출력 정의
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 1])

# 퍼셉트론 모델 생성 및 훈련
perceptron = Perceptron(input_size=2)
perceptron.train(training_inputs, labels)

# 훈련된 모델로 예측
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for test_input in test_inputs:
    prediction = perceptron.predict(test_input)
    print(f"Input: {test_input}, Prediction: {prediction}")
pre=perceptron.predict([0.3,0.3])
print(pre)
