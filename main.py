import cv2 as cv
import numpy as np
import math

'''
img = cv.imread('Prova2.jpg')


def line_segmentation(image):
    # Conversione dell'immagine in scala di grigi
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Applicazione del filtro di Canny per rilevare i bordi
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    # Rilevamento delle linee utilizzando la trasformata di Hough
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Disegno delle linee rilevate sull'immagine originale
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image


# Caricamento dell'immagine di input
input_image = cv.imread('Prova2.jpg')

# Segmentazione delle linee nell'immagine di input
output_image = line_segmentation(input_image)

# Visualizzazione dell'immagine di output
cv.imshow('Line Segmentation', output_image)
cv.waitKey(0)
cv.destroyAllWindows()


def data_augmentation(image):
    # Esempio di data augmentation: Ruotare l'immagine di un angolo casuale
    angle = np.random.randint(-30, 30)
    rows, cols, _ = image.shape
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    augmented_image = cv.warpAffine(image, M, (cols, rows))

    # Esempio di data augmentation: Aggiungere rumore all'immagine
    noise = np.random.normal(0, 1, image.shape)
    augmented_image = cv.add(augmented_image, noise.astype(np.uint8))

    # Esempio di data augmentation: Applicare uno shift orizzontale
    shift = np.random.randint(-20, 20)
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    augmented_image = cv.warpAffine(augmented_image, M, (cols, rows))

    return augmented_image


# Caricamento dell'immagine di input
#input_image = cv.imread('input_image.jpg')

# Segmentazione delle linee nell'immagine di input
out_img = line_segmentation(output_image)

# Applicazione di data augmentation all'immagine di output
augmented_image = data_augmentation(out_img)

# Visualizzazione dell'immagine di output
cv.imshow('Data Augmentation', augmented_image)
cv.waitKey(0)
cv.destroyAllWindows()
'''

import cv2
import easyocr


def recognize_text(image):
    # Conversione dell'immagine in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Inizializzazione del riconoscitore di EasyOCR
    reader = easyocr.Reader(['en'])

    # Riconoscimento del testo nell'immagine
    results = reader.readtext(gray)

    # Estrazione del testo riconosciuto
    recognized_text = ''
    for result in results:
        recognized_text += result[1] + ' '

    return recognized_text.strip()


# Caricamento dell'immagine di input
input_image = cv2.imread('Prova2.jpg')

rows,cols,channels = input_image.shape

M = cv.getRotationMatrix2D((cols/2, rows/2), 340, 1)
dst = cv.warpAffine(input_image,M,(cols,rows))

# Riconoscimento del testo nell'immagine
recognized_text = recognize_text(dst)

# Stampa del testo riconosciuto
print(recognized_text)



'''

import cv2
import pytesseract


def recognize_text(image):
    # Conversione dell'immagine in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applicazione dell'OCR per riconoscere il testo
    text = pytesseract.image_to_string(gray)

    return text


# Caricamento dell'immagine di input
input_image = cv2.imread('Prova2.jpg')
data = pytesseract.image_to_string(input_image)

# Riconoscimento del testo nell'immagine
recognized_text = recognize_text(input_image)

# Stampa del testo riconosciuto
print(recognized_text)
'''


'''
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)

lines = cv.HoughLines(edges,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()

'''


'''
print(type(img))
#

print(img.shape)
# (225, 400, 3)

img_flip_ud = cv.flip(img, 0)
cv.imwrite('lena_opencv_flip_ud.jpg', img_flip_ud)
# True

cv.imshow('image_flip_ud_lr', img_flip_ud)
cv.waitKey(0)
cv.destroyAllWindows()


img_flip_lr = cv.flip(img, 1)
cv.imwrite('lena_opencv_flip_lr.jpg', img_flip_lr)
# True

cv.imshow('image_flip_ud_lr', img_flip_lr)
cv.waitKey(0)
cv.destroyAllWindows()


img_flip_ud_lr = cv.flip(img, -1)
cv.imwrite('lena_opencv_flip_ud_lr.jpg', img_flip_ud_lr)
# True

cv.imshow('image_flip_ud_lr', img_flip_ud_lr)
cv.waitKey(0)
cv.destroyAllWindows()



img_rotate_90_clockwise = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
cv.imwrite('lena_cv_rotate_90_clockwise.jpg', img_rotate_90_clockwise)
# True
cv.imshow('img_rotate_90_clockwise', img_rotate_90_clockwise)
cv.waitKey(0)
cv.destroyAllWindows()


img_rotate_90_counterclockwise = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
cv.imwrite('lena_cv_rotate_90_counterclockwise.jpg', img_rotate_90_counterclockwise)
# True
cv.imshow('image_rotate_90_counterclockwise', img_rotate_90_counterclockwise)
cv.waitKey(0)
cv.destroyAllWindows()


img_rotate_180 = cv.rotate(img, cv.ROTATE_180)
cv.imwrite('lena_cv_rotate_180.jpg', img_rotate_180)
# True
cv.imshow('image_rotate_180', img_rotate_180)
cv.waitKey(0)
cv.destroyAllWindows()
'''

'''

img = cv.imread('Prova2.jpg',1)

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()

rows,cols,channels = img.shape

M = cv.getRotationMatrix2D((cols/2, rows/2), 340, 1)
dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow('rotation', dst)
cv.waitKey(0)
cv.destroyAllWindows()

#im = cv.imread('lena_opencv_red.jpg')

#cv.imshow('img', im)

#risultato = cv.stylization(im, sigma_s=150, sigma_r=0.25)
'''

'''
# visualizzazione dell'immagine
cv.imshow('Risultato: stilizzazione', risultato)

# visualizzazione della finestra
cv.waitKey(0)

# chiusura della finestra
cv.destroyAllWindows()
'''
#image_new = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#cv.imshow('Grayscaled Flower',image_new)
#cv.waitKey(0)
#cv.destroyAllWindows()
'''
# Edge Detection
# Usiamo la funzione Canny () per riconoscere i bordi dell’immagine.
cv.imwrite('Edge_Flowers.jpg', cv.Canny(im, 200, 300))
# Mostriamo l’immagine dei bordi
cv.imshow('Edges of the Image', cv.imread('Edge_Flowers.jpg'))
cv.waitKey(0)
cv.destroyAllWindows()
'''
'''
import numpy as np

# Load image
img = cv.imread('DatabaseHandling/LAM/full_pages/002/img/002_02.jpg')
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()


# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply binary thresholding
ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)

# Find contours
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Find bounding boxes of contours
boxes = [cv.boundingRect(c) for c in contours]

# Sort boxes by y-coordinate
boxes.sort(key=lambda x: x[1])

# Compute average height of boxes
avg_height = sum([b[3] for b in boxes]) / len(boxes)

# Merge boxes that are close together vertically
merged_boxes = []
current_box = boxes[0]
for box in boxes[1:]:
    if box[1] - current_box[1] < avg_height / 2:
        current_box = (min(current_box[0], box[0]), current_box[1], max(current_box[2], box[2]), max(current_box[3]+current_box[1], box[3]+box[1])-current_box[1])
    else:
        merged_boxes.append(current_box)
        current_box = box
merged_boxes.append(current_box)

# Draw boxes on image
for box in merged_boxes:
    x, y, w, h = box
    cv.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)

# Display image with boxes
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
'''