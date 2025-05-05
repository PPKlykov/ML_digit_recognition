import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageGrab, ImageTk
import numpy as np
from config import IMAGE_SIZE

class DigitDrawApp:
    def __init__(self, master, net):
        self.master = master
        self.net = net
        self.master.title("Нарисуй цифру")

        self.canvas = tk.Canvas(master, width=280, height=280, bg='white')
        self.canvas.pack()

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.btn_clear = tk.Button(master, text="Очистить", command=self.clear)
        self.btn_clear.pack(side=tk.LEFT)

        self.btn_predict = tk.Button(master, text="Узнать цифру", command=self.predict)
        self.btn_predict.pack(side=tk.LEFT)

        self.btn_paste = tk.Button(master, text="Вставить из буфера", command=self.paste_from_clipboard)
        self.btn_paste.pack(side=tk.LEFT)

        self.label_result = tk.Label(master, text="")
        self.label_result.pack()

        self.frame_confirm = tk.Frame(master)
        self.btn_yes = tk.Button(self.frame_confirm, text="Да", command=self.confirm_correct)
        self.btn_no = tk.Button(self.frame_confirm, text="Нет", command=self.correct_mistake)
        self.frame_confirm.pack()
        self.btn_yes.pack(side=tk.LEFT)
        self.btn_no.pack(side=tk.RIGHT)

        self.hide_confirm_buttons()

        self.last_prediction = None
        self.last_img_vector = None

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill=255)
        self.label_result.config(text="")
        self.hide_confirm_buttons()

    def paste_from_clipboard(self):
        try:
            clipboard_image = ImageGrab.grabclipboard()

            if isinstance(clipboard_image, Image.Image):
                clipboard_image = clipboard_image.convert("L")
                clipboard_image = clipboard_image.resize((280, 280))
                self.image.paste(clipboard_image)

                self.tk_img = ImageTk.PhotoImage(clipboard_image)
                self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)

                self.label_result.config(text="Картинка вставлена")
                self.hide_confirm_buttons()
            else:
                self.label_result.config(text="В буфере нет изображения.")
        except Exception as e:
            self.label_result.config(text=f"Ошибка вставки: {str(e)}")

    def predict(self):
        img = self.image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img = ImageOps.invert(img)
        img_data = np.asarray(img) / 255.0
        img_vector = img_data.reshape((IMAGE_SIZE * IMAGE_SIZE, 1))

        output = self.net.feedforward(img_vector)
        prediction = np.argmax(output)

        self.last_prediction = prediction
        self.last_img_vector = img_vector

        self.label_result.config(text=f"Распознано: {prediction}")
        self.show_confirm_buttons()

    def confirm_correct(self):
        self.label_result.config(text="✅ Отлично!")
        self.hide_confirm_buttons()

    def correct_mistake(self):
        self.hide_confirm_buttons()
        correct_window = tk.Toplevel(self.master)
        correct_window.title("Введите правильную цифру")

        tk.Label(correct_window, text="Правильная цифра:").pack()
        entry = tk.Entry(correct_window)
        entry.pack()

        def submit_correction():
            correct_digit = int(entry.get())
            self.train_on_example(self.last_img_vector, correct_digit)
            self.label_result.config(text=f"Переобучено на цифру {correct_digit}")
            correct_window.destroy()

        btn_submit = tk.Button(correct_window, text="OK", command=submit_correction)
        btn_submit.pack()

    def train_on_example(self, img_vector, correct_digit):
        correct_output = np.zeros((10, 1))
        correct_output[correct_digit] = 1.0

        nabla_b, nabla_w = self.net.backprop(img_vector, correct_output)

        eta = 0.5
        self.net.weights = [w - eta * nw for w, nw in zip(self.net.weights, nabla_w)]
        self.net.biases = [b - eta * nb for b, nb in zip(self.net.biases, nabla_b)]

    def show_confirm_buttons(self):
        self.frame_confirm.pack()

    def hide_confirm_buttons(self):
        self.frame_confirm.pack_forget()
