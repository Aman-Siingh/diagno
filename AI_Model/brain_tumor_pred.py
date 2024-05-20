from tensorflow import keras
from PIL import Image
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def filepath():
    root = tk.Tk()
    file_path = filedialog.askopenfilename()
    #root.withdraw()
    print(file_path)
    root.destroy()
    img = Image.open(str(file_path))
    p=pred(img)
    print("type of tumor : ",p)
    root.quit()
    return p

def pred(imgg):
    model = keras.models.load_model(r"D:\django_tutorial\hello\effnet.h5")
    opencvImage = cv2.cvtColor(np.array(imgg), cv2.COLOR_RGB2BGR)
    imgg = cv2.resize(opencvImage,(150,150))
    imgg = imgg.reshape(1,150,150,3)
    p = model.predict(imgg)
    p = np.argmax(p,axis=1)[0]

    if p==0:
       p="Glioma Tumor"
    elif p==1:
        p="no tumor"
    elif p==2:
        p="Meningioma Tumor"
    else:
        p="Pituitary Tumor"
    return p


filepath()