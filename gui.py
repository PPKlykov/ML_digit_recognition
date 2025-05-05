import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageGrab, ImageTk
import numpy as np
import network
import pickle
import os
from digital_draw import DigitDrawApp
from config import LAYERS

WEIGHTS_PATH = "trained_weights_2.pkl"

net = network.Network(LAYERS)
if os.path.exists(WEIGHTS_PATH):
    with open(WEIGHTS_PATH, "rb") as f:
        net.biases, net.weights = pickle.load(f)
    print("Загружены сохранённые веса.")
else:
    print("Весов нет. Сначала обучите сеть через model_training.py")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawApp(root, net)
    root.mainloop()