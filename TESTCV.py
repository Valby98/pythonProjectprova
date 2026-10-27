from DatabaseHandling.LAM import *
from Preprocessing.LAM_backgroundRemover import *

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from torchvision.transforms import ToTensor

print("\n/*** Start program ***/\n")

test_type = 3

match test_type:
    case 0:
        test = "pagina intera"
    case 1:
        test = "una riga"
    case 2:
        test = "canny edge"
    case 3:
        test = "preprocessing"
    case 4:
        test = "remove"
    case _:
        print("Test non ammesso")

lam_path = "DatabaseHandling/LAM"
pages_dir = "full_pages"

match test:
    case "una riga":
        # test della classe LAM_lines: mostra il funzionamento

        # dichiaro il nome della cartella dove trovare i json descrittivi
        json_folder = "basic"
        # dichiaro il nome del file json da utilizzare
        name_set = 'train'

        # dichiaro un oggetto di classe LAM
        # lam_path è il percorso in cui si trova il dataset
        # json_folder è la cartella contenente tutti i .jason descrittivi delle immagini
        #   a singola riga contenute in lines/img
        # ToTensor() è un metodo che permette di trasformare i file immagini (H x W x C) [0, 255]
        #   in un tensore torch.FloatTensor (C x H x W) [0.0, 1.0]
        # name_set è il nome del file (senza estensione) json contenente le descrizioni delle
        #   singole righe
        train = LAM_Lines(lam_path, json_folder, ToTensor(), name_set)
        # questa classe permette solo di ottenere il dataset d'immagini di righe
        #   già croppate

        # carico il dataset con pytorch
        img, text, decade_id, img_name = train.__getitem__(4)
        print(text)

    case "pagina intera":
        train = LAM_FullPages(lam_path, pages_dir, ToTensor())

        images = train.getItems()

        print("Testo:")
        print(images[0].getText())

        img_path = images[0].showImg()

    case "canny edge":
        train = LAM_FullPages(lam_path, pages_dir, ToTensor())

        images = train.getItems()

        img_path = images[0].getImgFullPath()

        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        edges = cv.Canny(img, 100, 200)
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    case "preprocessing":
        train = LAM_FullPages(lam_path, pages_dir, ToTensor())
        start = 100
        end = start + 5
        img_paths = [train.getItem(i).getImgFullPath() for i in range(start, end)]

        image = cv.imread(img_paths[0])
        original = image.copy()
        hh, ww = image.shape[:2]

        '''
        for j in img_paths:
            x = BackGroundRemover(j)

            img = x.removeBackground()

            x.showImg(img)
        '''

        cv.imshow("original", original)
        # visualizzazione della finestra
        cv.waitKey(0)

        # chiusura della finestra
        cv.destroyAllWindows()

        # eseguo blurring: permette di migliorare la binarizzazione dell'immagine
        # andando a eliminare le scritte e rendere visibile maggiormente la pagina
        # rispetto allo sfondo
        ksize = (20, 20)
        # NB: la dimensione del kernel migliora il riconoscimento della pagina, ma peggiora
        # l'ottenimento dei contorni
        image = cv.blur(image, ksize)

        # definisco i boundaries per binarizzazione
        lower = np.array([100, 100, 100])
        upper = np.array([255, 255, 255])

        # ottengo immagine binarizzata
        thresh = cv.inRange(image, lower, upper)

        # ottengo i contorni a partire dall'immagine binarizzata
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # creo un contorno bianco su sfondo nero come maschera
        mask = np.zeros((hh, ww), dtype=np.uint8)
        white = (255, 255, 255)
        cv.drawContours(mask, contours, 0, white, cv.FILLED)
        mask_inv = 255 - mask

        background = np.full_like(image, (0, 0, 0))
        # applico la maschera all'immagine
        image = original.copy()
        image_masked = cv.bitwise_and(image, image, mask=mask)
        background_masked = cv.bitwise_and(background, background, mask=mask_inv)

        result = cv.add(image_masked, background_masked)
        x, y, w, h = cv.boundingRect(contours[0])
        print(x, y, w, h)
        border = 70
        new_w = w + border  # int(w + w*0.05)
        new_h = h + border  # int(h + h*0.05)
        new_x = x - (border >> 1)  # int(x - (new_w-w)*0.5)
        new_y = y - (border >> 1)  # int(y - (new_h-h)*0.5)
        cropped = result[new_y:new_y + new_h, new_x:new_x + new_w]

    case "remove":
        train = LAM_FullPages(lam_path, pages_dir, ToTensor())
        path = train.getItem(5).getImgFullPath()

        x = BackGroundRemover(path)

        img = x.removeBackground()

        x.showImg(img)

    case _:
        print("Test non ammesso!")


print("\n/*** End program ***/\n")