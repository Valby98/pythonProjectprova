# SONO PRESENTI ESEMPI SIA FUNZIONANTI CHE NON NEL CASO SI VOLESSE PROVARE LAVORARCI
# SU PER CORREGGERLI

import PIL.Image
import cv2
import numpy as np
import pandas as pd
from pytesseract import pytesseract
import matplotlib.pyplot as plt

import keras_ocr

from glob import glob
from tqdm.notebook import tqdm
from PIL import Image
import pytesseract
import easyocr


########################### FUNZIONA (ESEMPIO MIGLIORE)
import easyocr

image_path = '002_02.jpg'

reader = easyocr.Reader(['it'], gpu = False)
result = reader.readtext(image_path)
print(result)


img = cv2.imread(image_path)
for detection in result:
    top_left = tuple(int(val) for val in detection[0][0])
    bottom_right = tuple(int(val) for val in detection[0][2])
    text = detection[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.rectangle(img, top_left, bottom_right, (0,255,0), 5)
    img = cv2.putText(img, text, top_left, font, -5, (255,255,255), 2, cv2.LINE_AA)

plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()


'''
####################################### FUNZIONA

import cv2
import easyocr

reader = easyocr.Reader(['en'],gpu = False) # load once only in memory.

image_file_name='002_02.jpg'
image = cv2.imread(image_file_name)

# sharp the edges or image.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
r_easy_ocr=reader.readtext(thresh,detail=0)

print(r_easy_ocr)
'''


'''
################################## FUNZIONA (PRODUCE SOLO ARRAY DI VALORI)

import cv2

def extract_handwritten_text(image_path):
    # Carica l'immagine utilizzando OpenCV
    image = cv2.imread(image_path)

    # Converte l'immagine in scala di grigi
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applica un'operazione di thresholding per migliorare la qualità dell'immagine
    _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Trova i contorni nell'immagine
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Estrai i rettangoli contenenti i contorni
    text_rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        text_rectangles.append((x, y, w, h))

    # Ordina i rettangoli per la coordinata y, per garantire che il testo sia ordinato correttamente
    text_rectangles = sorted(text_rectangles, key=lambda x: x[1])

    # Estrai il testo da ciascun rettangolo
    extracted_text = ""
    for rectangle in text_rectangles:
        x, y, w, h = rectangle
        roi = gray_image[y:y+h, x:x+w]

        # Personalizza ulteriormente l'estrazione del testo da ciascun rettangolo se necessario
        # Ad esempio, utilizzando il riconoscimento di caratteri personalizzato

        # Aggiungi il testo estratto alla stringa finale
        #extracted_text = ",".join(str(row) for row in roi) + "\n"
        extracted_text = ",".join(map(str, roi))


    return extracted_text

if __name__ == "__main__":
    image_path = "002_02.jpg"
    extracted_text = extract_handwritten_text(image_path)
    print("Testo estratto:\n", extracted_text)
'''


'''
############################### FUNZIONA

import pytesseract


def handwriting_recognition(image_path):
    # Carica l'immagine
    image = cv2.imread(image_path)

    # Converte l'immagine in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    pytesseract.pytesseract.tesseract_cmd = r'C:/Users/lenovo/pythonProjectprova/Tesseract-OCR/tesseract.exe'

    # Applica la binarizzazione per migliorare la leggibilità del testo
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Esegui il riconoscimento del testo utilizzando Tesseract OCR
    text = pytesseract.image_to_string(binary, lang='eng')

    # Stampa il testo riconosciuto
    print("Testo riconosciuto:")
    print(text)


# Esegui l'algoritmo sulle immagini di test
#dataset_path = "/percorso/dataset"
image_files = ["Prova.jpg"]

for file in image_files:
    #image_path = dataset_path + "/" + file
    handwriting_recognition(file)
'''


'''
############################### FUNZIONA

myconfig = r"--psm 6 --oem 3"       # --psm da 0 a 13
pytesseract.pytesseract.tesseract_cmd = r'C:/Users/lenovo/pythonProjectprova/Tesseract-OCR/tesseract.exe'
#text = pytesseract.image_to_string(PIL.Image.open("002_02.jpg"), config=myconfig)

#print(text)

#img = cv2.imread('002_02.jpg')
#height, width, _ = img.shape
#boxes = pytesseract.image_to_boxes(img, config=myconfig)
#print(boxes)

#print(pytesseract.image_to_string("002_02.jpg", lang='eng'))

#fig, ax = plt.subplots(figsize=(10,10))
#ax.imshow(plt.imread('002_02.jpg'))
#ax.axis('off')
#plt.show()


reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext('002_02.jpg')
print(results)
'''




'''
############################### NON FUNZIONANTE

def line_detection(image_path):
    # Carica l'immagine
    image = cv2.imread(image_path)

    # Converte l'immagine in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applica un filtro di Canny per rilevare i bordi
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Applica la trasformata di Hough per rilevare le linee
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Disegna le linee rilevate sull'immagine originale
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Mostra l'immagine con le linee rilevate
    cv2.imshow("Line Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Esegui l'algoritmo sulle immagini di test
#dataset_path = "/percorso/dataset"
image_files = ["Prova.jpg"]

line_detection("Prova.jpg")


for file in image_files:
    #image_path = dataset_path + "/" + file
    
    image_path = file
    line_detection(image_path)
'''

'''
################################ NON FUNZIONANTE

def line_detection(image_path, output_path):
    # Carica l'immagine
    image = cv2.imread(image_path)

    # Converte l'immagine in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applica un filtro di Canny per rilevare i bordi
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Applica la trasformata di Hough per rilevare le linee
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Disegna le linee rilevate sull'immagine originale
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Salva l'immagine con le linee rilevate
    cv2.imwrite(output_path, image)


# Esegui l'algoritmo sulle immagini di test
#dataset_path = "pythonProjectprova/DatabaseHandling/LAM/full_pages/002/img"
output_path = "Prove_varie"
image_files = ["002_02.jpg"]

for file in image_files:
    #image_path = dataset_path + "/" + file
    output_image_path = output_path + "/" + file
    line_detection(file, output_image_path)
'''




'''
####################### NON FUNZIONANTE (CAUSA LIBRERIA)

import ocrolib

def handwriting_recognition(image_path):
    # Carica l'immagine
    image = ocrolib.read_image(image_path)

    # Esegui il riconoscimento del testo utilizzando OCRopus
    result = ocrolib.ocr(image)

    # Estrai il testo riconosciuto
    text = ocrolib.ocr_result_to_string(result)

    # Stampa il testo riconosciuto
    print("Testo riconosciuto:")
    print(text)


# Esegui l'algoritmo sulle immagini di test
dataset_path = "/percorso/dataset"
image_files = ["immagine1.jpg", "immagine2.jpg", "immagine3.jpg"]

# Carica i modelli OCR
ocrolib.init()

for file in image_files:
    image_path = dataset_path + "/" + file
    handwriting_recognition(image_path)
'''


'''

############################### NON FUNZIONANTE

import tensorflow as tf
from keras_ocr.detection import Detector
from keras_ocr.recognition import Recognizer


def handwriting_recognition(image_path):
    # Carica l'immagine
    image = cv2.imread(image_path)

    # Inizializza il rilevatore (detector) e il riconoscitore (recognizer)
    detector = Detector()
    recognizer = Recognizer()

    # Esegui la rilevazione dei testi nell'immagine
    boxes = detector.detect(images=[image])[0]

    # Estrai le regioni di interesse contenenti i testi
    regions = [image[y1:y2, x1:x2] for (x1, y1, x2, y2) in boxes]

    # Esegui il riconoscimento dei testi
    predictions = recognizer.recognize(images=regions)

    # Stampa i testi riconosciuti
    print("Testi riconosciuti:")
    for text in predictions:
        print(text)


# Esegui l'algoritmo sulle immagini di test
#dataset_path = "/percorso/dataset"
image_files = ["Prova.jpg"]

for file in image_files:
    #image_path = dataset_path + "/" + file
    handwriting_recognition(file)
'''


'''
####################### NON FUNZIONANTE (CAUSA LIBRERIA)

from PIL import Image
import ocrolib

def recognize_handwritten_text(image_path):
    # Carica l'immagine utilizzando PIL
    image = Image.open(image_path)

    # Converte l'immagine in un'immagine binaria utilizzando OCRopus
    binary_image = ocrolib.adaptive_normalize(image)

    # Riconosce il testo dall'immagine binaria utilizzando OCRopus
    extracted_text = ocrolib.recognize_line(binary_image)

    return extracted_text

if __name__ == "__main__":
    image_path = "path/della/tua/immagine.jpg"
    extracted_text = recognize_handwritten_text(image_path)
    print("Testo estratto:\n", extracted_text)

'''



'''
################################## NON FUNZIONANTE

import keras_ocr
import matplotlib.pyplot as plt
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

pipeline = keras_ocr.pipeline.Pipeline()

images = [ keras_ocr.tools.read(img) for img in ['002_02.jpg', 'Prova.jpg']
           ]

prediction_group = pipeline.recognize(images)

fig, axs = plt.subplots(nrows=len(images), figsize=(10,20))

for ax, image, predictions in zip(axs, images, prediction_group):
    keras_ocr.tools.drawAnnotations(image=image,
                                    predictions=predictions,
                                    ax=ax)

predicted_image = prediction_group[1]
for text, box in predicted_image:
    print(text)
'''

'''
################################### NON FUNZIONANTE

import matplotlib.pyplot as plt
import keras_ocr
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def recognize_handwritten_text(image_path):
    # Carica il pipeline di Keras-OCR
    pipeline = keras_ocr.pipeline.Pipeline()

    # Carica l'immagine
    image = keras_ocr.tools.read(image_path)

    # Esegui il riconoscimento del testo sull'immagine
    result = pipeline.recognize([image])

    # Estrai il testo riconosciuto
    extracted_text = ""
    for detected_text in result[0][1]:
        extracted_text += detected_text[1] + "\n"

    return extracted_text

if __name__ == "__main__":
    image_path = "C:/Users/lenovo/pythonProjectprova/002_02.jpg"
    extracted_text = recognize_handwritten_text(image_path)
    print("Testo estratto:\n", extracted_text)
'''


'''
################################# NON FUNZIONANTE

import cv2

def recognize_handwritten_text(image_path):
    # Carica l'immagine utilizzando OpenCV
    image = cv2.imread(image_path)

    # Carica il modello EAST per la rilevazione del testo
    east_model = "frozen_east_text_detection.pb"
    net = cv2.dnn.readNet(east_model)

    # Dimensioni dell'immagine di input per il modello EAST
    height, width = image.shape[:2]
    new_height, new_width = 320, 320
    scale_x, scale_y = width / new_width, height / new_height

    # Prepara l'immagine per l'inferenza del modello EAST
    blob = cv2.dnn.blobFromImage(image, 1.0, (new_width, new_height), (123.68, 116.78, 103.94), True, False)
    net.setInput(blob)

    # Esegui l'inferenza per rilevare i contorni del testo
    output_layer = net.getUnconnectedOutLayersNames()
    scores, geometry = net.forward(output_layer)

    # Applica il bounding box sulle regioni con testo rilevato
    rectangles = []
    confidences = []
    for i in range(scores.shape[2]):
        score_data = scores[0, 0, i, 0]
        x_data0 = geometry[0, 0, i, 0]
        x_data1 = geometry[0, 0, i, 1]
        x_data2 = geometry[0, 0, i, 2]
        x_data3 = geometry[0, 0, i, 3]
        angles = geometry[0, 0, i, 4]

        if score_data < 0.5:
            continue

        offset_x, offset_y = i * 4, i * 4
        angle = angles * 3.141592653589793 / 180.0
        cos = np.cos(angle)
        sin = np.sin(angle)

        h = x_data0 + x_data2
        w = x_data1 + x_data3

        end_x = int(offset_x + cos * x_data1 + sin * x_data2)
        end_y = int(offset_y - sin * x_data1 + cos * x_data2)
        start_x = int(end_x - w)
        start_y = int(end_y - h)

        rectangles.append((start_x, start_y, end_x, end_y))
        confidences.append(float(score_data))

    # Applica Non-Maximum Suppression (NMS) per rimuovere sovrapposizioni di rettangoli
    indices = cv2.dnn.NMSBoxes(rectangles, confidences, 0.5, 0.4)

    # Estrai il testo dai rettangoli rilevati
    extracted_text = ""
    for i in indices:
        i = i[0]
        start_x, start_y, end_x, end_y = rectangles[i]
        text_roi = image[start_y:end_y, start_x:end_x]
        extracted_text += pytesseract.image_to_string(text_roi) + "\n"

    return extracted_text

if __name__ == "__main__":
    image_path = "C:/Users/lenovo/pythonProjectprova/002_02.jpg"
    extracted_text = recognize_handwritten_text(image_path)
    print("Testo estratto:\n", extracted_text)
'''



'''
################################ NON FUNZIONANTE (RILEVAMENTO ERRATO)

import matplotlib.pyplot as plt
from skimage import io, color, feature
from skimage.transform import probabilistic_hough_line


def line_detection(image_path):
    # Carica l'immagine
    image = io.imread(image_path)

    # Converte l'immagine in scala di grigi
    gray = color.rgb2gray(image)

    # Applica un filtro di Canny per rilevare i bordi
    edges = feature.canny(gray)

    # Esegui la trasformata di Hough probabilistica per rilevare le linee
    lines = probabilistic_hough_line(edges, threshold=10, line_length=50, line_gap=5)

    # Disegna le linee rilevate sull'immagine originale
    fig, ax = plt.subplots()
    ax.imshow(image)

    for line in lines:
        p0, p1 = line
        ax.plot((p0[0], p1[0]), (p0[1], p1[1]), linewidth=2, color='red')

    plt.show()


# Esegui l'algoritmo su un'immagine di test
image_path = "C:/Users/lenovo/pythonProjectprova/002_02.jpg"
line_detection(image_path)
'''

