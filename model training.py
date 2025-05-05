import network
import mnist_loader
import pickle
from config import LAYERS

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network(LAYERS)
net.SGD(training_data, 100, 10, 1.0, test_data=test_data)

with open("trained_weights_2.pkl", "wb") as f:
    pickle.dump((net.biases, net.weights), f)
print("Обучение завершено. Веса сохранены в 'trained_weights_2.pkl'.")