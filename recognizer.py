import cv2
import tkinter as tk
from tkinter import Canvas, Label, Button
from PIL import Image, ImageDraw, ImageGrab
from keras.models import Sequential
from keras.layers import Dense,Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Import this line for correct Matplotlib backend
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


def event_function(event):
    
    x = event.x
    y = event.y
    
    x1 = x - 10
    y1 = y - 10
    x2 = x + 10
    y2 = y + 10

    canvas.create_oval((x1,y1,x2,y2),fill= 'Black')
    img_draw.ellipse((x1,y1,x2,y2),fill= 'White')

def save():
    
    global count

    img_array=np.array(img)
    img_array=cv2.resize(img_array,(28,28))
    cv2.imwrite(str(count)+'.jpg',img_array)
    count=count+1

def clear_canvas():

    global img, img_draw

    canvas.delete("all")
    img=Image.new('RGB', (400, 400), (0,0,0))
    img_draw=ImageDraw.Draw(img)
    update_distribution_chart(np.zeros(10))
    prediction_label.config(text='PREDICTED DIGIT: NONE')

def predict_digit():
    
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array,(28,28))
    
    img_array = img_array/255.0
    img_array = img_array.reshape(1,28,28)
    prediction = model.predict(img_array)
    digit = np.argmax(prediction,axis=1)
    
    prediction_label.config(text='PREDICTED DIGIT:' + str(digit))
    update_distribution_chart(prediction[0])
    return digit, prediction


def update_distribution_chart(prediction):
    global canvas_chart  # Make canvas_chart a global variable

    if canvas_chart is not None:
        canvas_chart.get_tk_widget().pack_forget()  # Remove the previous canvas

    fig, ax = plt.subplots()
    plt.bar(range(10), prediction)
    plt.xticks(range(10))
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title('Digit Probability Distribution')

    canvas_chart = FigureCanvasTkAgg(fig, master=root)
    canvas_chart.get_tk_widget().pack() #Update the Matplotlib canvas


if __name__ == '__main__':
    model = Sequential()
    model.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])
    model.load_weights('model_V1_Adam.keras')

    canvas_chart = None
    # Create the main window
    count = 0
    root = tk.Tk()
    root.title("Handwritten Digit Recognizer")

    # Create a canvas for drawing
    canvas = Canvas(root, width=400, height=400, bg='white')
    canvas.pack()

    # Create a button to recognize the digit
    recognize_button = Button(root, text="Recognize Digit", command=predict_digit)
    recognize_button.pack()

    # Create a button to clear the canvas
    clear_button = Button(root, text="Clear Canvas", command=clear_canvas)
    clear_button.pack()

    # Create a button to save the digit
    button_save=tk.Button(root, text='SAVE',command=save)
    button_save.pack()

    # Create a label for displaying the prediction
    prediction_label=tk.Label(root, text='PREDICTED DIGIT: NONE')
    prediction_label.pack()

    canvas.bind('<B1-Motion>',event_function)
    img=Image.new('RGB',(500,500),(0,0,0))
    img_draw=ImageDraw.Draw(img)

    root.mainloop()