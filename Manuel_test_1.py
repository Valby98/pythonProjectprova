test_nb = 3
test = ''

print("\n/*** Start program ***/\n")

match test_nb:
    case 0:
        test = "hello world!"
    case 1:
        test = "kaggle"
    case 2:
        test = "kaggle1"    #da non considerare
    case 3:
        test = "kaggle2"
    case 4:
        test = "parameters"
    case _:
        assert False, 'test_nb non ammesso'

out_dir = "./OutDir"

match test:
    case "hello world!":
        print(test)

    case "kaggle":
        """
        Note (29/8/23) --> a questo software è necessario:
        1. implementare funzioni che eseguano il calcolo della Character Error Rate (CER) e della Word Error Rate (WER),
            probabilmente è necessario aggiungerli come callback (vedi input callbacks del metodo fit) oppure
            all'interno della ctc (andando ad inserire tipo dei contatori globali) oppure guarda questi due link:
            - https://stackoverflow.com/questions/60285167/tensorflow-callback-as-custom-metric-for-ctc
            - https://github.com/georgeretsi/HTR-ctc/blob/master/train_code/train_htr.py
        2. aggiungere la possibilità di salvare i filtri allenati in una cartella di output. utilizza come percorso ./OutDir/CNNfilters e crea delle 
            sottocartelle per ogni livello di convoluzione (ce ne sono 4), nel caso dividile anche per data in modo tale da poter rivedere 
            output precedenti 
        3. la possibilità di plottare (anche al termine del training) l'andamento della loss di training e di validation
        4. non sono riuscito a trovare il modo per ottenere i parametri allenati dalla rete, prova a vedere se è
            possibile fare qualcosa guardando anche keras.callbacks.TensorBoard e in generale keras.callbacks
        5. (SE RIMANE TEMPO) vedere se è possibile in maniera automatica generare una grafica che mostri la rete 
            https://www.google.com/search?client=safari&sca_esv=561082500&q=vggnet&tbm=isch&source=lnms&sa=X&ved=2ahUKEwi4is2D1IKBAxXMzgIHHZtvDisQ0pQJegQIDBAB&biw=1440&bih=632&dpr=2#imgrc=TK0-YkID_TjM9M
            qualcosa tipo l'immagine sopra
        Sentiamoci domani 30/8/23 se ho dimenticato qualcosa o se non riesci a compilare/eseguire, il codice è stato testato ed è
            funzionante, la loss ottenuta alla 5 epoca è di circa 120
        """


        import numpy as np
        import matplotlib.pyplot as plt
        import cv2 as cv
        from DatabaseHandling.LAM import LAM_Lines
        from torchvision.transforms import ToTensor
        from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
        from keras import backend as K
        from keras.models import Model
        from keras.optimizers import Adam
        from tensorflow import keras

        #permette di plottare 6 immagini con la rispettiva label/testo
        def show_imgs(imgs, labels):
            if len(imgs) > 6:
                imgs = imgs[0:6]
            for i in range(6):
                plt.subplot(2, 3, i+1)
                plt.imshow(cv.rotate(imgs[i], cv.ROTATE_90_COUNTERCLOCKWISE), cmap='gray')
                plt.title(labels[i])
                plt.axis('off')
            plt.show()

        #le dimensioni sono molto piccole per andare ad eseguire i test sulla compilazione
        #il numero di elementi ottenibile è circa 19000
        train_size = 20
        valid_size = 10

        json_folder = "basic"
        name_set = "train"
        lam_path = "./DatabaseHandling/LAM"
        #uso la classe LAM_lines solo per ottenere il percorso delle immagini (vedi full_path e imgs_dir dopo)
        train_dataset = LAM_Lines(lam_path, json_folder, ToTensor(), name_set)
        full_path = './DatabaseHandling/LAM/lines/img/'

        #creo una lista di percorsi associati alle immagini e prendo le label associate
        imgs_dir = [full_path + i['img'] for i in train_dataset.samples]
        imgs_labels = [i['text'] for i in train_dataset.samples]
        max_size = len(imgs_dir)
        assert (train_size + valid_size) <= max_size

        #questa funzione permette di eseguire il padding dell'immagine ad una dimensione prestabilita
        #se una immagine è più piccola della dimensione viene eseguito padding altrimenti (se più grande), 
        #   la funzione resize la riporta alla dimensione voluta
        def preprocess(img):
            WHITE = [255, 255, 255]
            new_shape = (256, 64)
            h = img.shape[0]
            if h < new_shape[1]:
                img = cv.copyMakeBorder(img, 0, new_shape[1] - h, 0, 0, cv.BORDER_CONSTANT, value=WHITE)
            w = img.shape[1]
            if w < new_shape[0]:
                img = cv.copyMakeBorder(img, 0, 0, 0,  new_shape[0] - w, cv.BORDER_CONSTANT, value=WHITE)
            img = cv.resize(img, new_shape, interpolation = cv.INTER_AREA)
            img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            return img

        #questo ciclo permette di creare la lista di immagini da elaborare (questa funzione riempie la RAM quando si usa tutto il dataset)
        train_x = []
        train_labels = []
        for i in range(train_size):
            img_dir = imgs_dir[i]
            train_labels.append(imgs_labels[i])
            image = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)
            image = preprocess(image)
            image = image / 255
            train_x.append(image)
        
        #stessa cosa di prima ma con le immagini di validazione, più avanti sarà trasformata in funzione
        valid_x = []
        valid_labels = []
        for i in range(valid_size):
            img_dir = imgs_dir[-i-1]
            valid_labels.append(imgs_labels[-i-1])
            image = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)
            image = preprocess(image)
            image = image / 255
            valid_x.append(image)

        #l'implementazione da cui è preso questo codice dice di invertire la width con la height, NB: il canale è solo 1, quindi scala di grigi
        train_x = np.array(train_x).reshape(-1, 256, 64, 1)
        valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)

        #qui si viene a creare un nuovo set di caratteri, questo perchè ho voluto ridurre la dimensione 
        #dell'alfabeto possibile andando a eliminare le maiuscole (vedi il metodo lower() sotto)
        new_set = {}
        idx = 0
        for k, v in train_dataset.char_to_idx.items():
            if k.lower() in new_set.keys():
                continue
            new_set[k.lower()] = idx
            idx += 1

        #stampa il set di caratteri che si possono predire dalla rete
        #print(new_set)

        max_str_len = max([len(i) for i in train_labels + valid_labels])
        num_of_characters = len(new_set) + 1
        num_of_timestamps = 128

        #questa funzione trasforma la stringa/label ground truth in tensore, dato il set di caratteri
        def label_to_num(charset, label):
            ret = []
            for ch in label.lower():
                val = charset.get(ch)
                assert val != None, f'val = {val}, ch = {ch}\nlabel = {label}'
                ret.append(val)
            return ret
        
        #questa funzione trasforma un tensore in stringa, dato il set di caratteri
        def num_to_label(charset, num):
            ret = ""
            temp_set = {v:k for k, v in charset.items()} #NB funziona solo se i values non si ripetono mai
            for ch in num:
                val = temp_set.get(ch)
                assert val != None, f'val = {val}, ch = {ch}\num = {num}'
                ret += val
            return ret
        
        #mostra un esempio di conversione stringa-tensore
        display_label_to_num = False
        if display_label_to_num:
            idx = 2
            print(new_set)
            print(train_labels[idx])
            print("length = ", len(train_labels[idx]))
            print(label_to_num(new_set, train_labels[idx]))

        #mostra un esempio di comversione tensore-stringa
        display_num_to_label = False
        if display_num_to_label:
            idx = 2
            #print(new_set)
            temp = train_labels[idx]
            print(temp)
            num = label_to_num(new_set, temp)
            print(f"len = {len(num)} --> ", num)
            label = num_to_label(new_set, num)
            print(label)
        
        #creazione di alcuni array di appoggio per la fase di training
        train_y = np.ones([train_size, max_str_len]) * -1
        train_label_len = np.zeros([train_size, 1])
        train_input_len = np.ones([train_size, 1]) * (num_of_timestamps - 2)
        train_output = np.zeros([train_size])

        for i in range(train_size):
            train_label_len[i] = len(train_labels[i])
            train_y[i, 0:len(train_labels[i])] = label_to_num(new_set, train_labels[i])

        #mostra solo i vettori creati
        display_train_vect = False
        if display_train_vect:
            print("train_y = ", train_y)
            print("train_label_len = ", train_label_len)
            print("train_input_len = ", train_input_len)
            print("train_output = ", train_output)

        #creazione di alcuni array di appoggio per la fase di validation
        valid_y = np.ones([valid_size, max_str_len]) * -1
        valid_label_len = np.zeros([valid_size, 1])
        valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps - 2)
        valid_output = np.zeros([valid_size])

        for i in range(valid_size):
            valid_label_len[i] = len(valid_labels[i])
            valid_y[i, 0:len(valid_labels[i])] = label_to_num(new_set, valid_labels[i])

        #mostra solo i vettori creati
        display_valid_vect = False
        if display_valid_vect:
            print("valid_y = ", valid_y)
            print("valid_label_len = ", valid_label_len)
            print("valid_input_len = ", valid_input_len)
            print("valid_output = ", valid_output)

        # /************** RETE NEURALE *******************/

        #se la dimensione dell'input cambia, va cambiato anche questa riga
        input_data = Input(shape=(256, 64, 1), name='input') #256 sarà poi la lunghezza della sequenza temporale a meno dei maxpooling
        #NB: la lunghezza (es 256) deve essere almeno più lunga della lunghezza massima del ground truth NB: --> 256 passa per i maxpooling quindi la lunghezza è in realtà 256 / 4

        inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner) # dim = 128 x 32

        inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner) #dim = 64 x 16
        inner = Dropout(0.3)(inner)

        inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner) #dim = 32 x 16 --> 32 è la sequenza temporale --> quindi è (1,2) per evitare di ridurre la lunghezza
        inner = Dropout(0.3)(inner)
        #gli iperparametri di questi strati convolutivi non dipendono dall'input

        # CNN to RNN
        #512 dipende dallo shape dell'input dopo i maxpooling
        inner = Reshape(target_shape=((128, 512)), name='reshape')(inner) #128 = numero di canali dalla canvolutiva
        inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

        ## RNN
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner) #256 = dimensione (altezza) della sequenza tempoerale, la lunghezza della sequenza temporale qua non è mostrata

        ## OUTPUT
        inner = Dense(num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
        y_pred = Activation('softmax', name='softmax')(inner) #dimensione di uscita = sequenza temporale * num of characters = 32 * 62
        #ciascuna riga contiene le 62 probabilità per ogni lettera

        model = Model(inputs=input_data, outputs=y_pred)
        #model.summary()


        # Compilazione del modello con la funzione di perdita CTC
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='ctc', optimizer=optimizer)


        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            # the 2 is critical here since the first couple outputs of the RNN
            # tend to be garbage
            y_pred = y_pred[:, 2:, :]
            return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

        labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)

        #passare metrica CER o WER per valutare correttezza rete
        model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate = 0.001))

        train_en = True
        if train_en:
            model_final.fit(x=[train_x, train_y, train_input_len, train_label_len], y=train_output, 
                            validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
                            epochs=5, batch_size=5)

    case 'kaggle1':

        import numpy as np
        import matplotlib.pyplot as plt
        import cv2 as cv
        from DatabaseHandling.LAM import LAM_Lines
        from torchvision.transforms import ToTensor
        from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
        from keras import backend as K
        from keras.models import Model
        from keras.optimizers import Adam
        from tensorflow import keras
        from datetime import datetime

        #permette di plottare 6 immagini con la rispettiva label/testo
        def show_imgs(imgs, labels):
            if len(imgs) > 6:
                imgs = imgs[0:6]
            for i in range(6):
                plt.subplot(2, 3, i+1)
                plt.imshow(cv.rotate(imgs[i], cv.ROTATE_90_COUNTERCLOCKWISE), cmap='gray')
                plt.title(labels[i])
                plt.axis('off')
            plt.show()

        #le dimensioni sono molto piccole per andare ad eseguire i test sulla compilazione
        #il numero di elementi ottenibile è circa 19000
        train_size = 20
        valid_size = 10

        json_folder = "basic"
        name_set = "train"
        lam_path = "./DatabaseHandling/LAM"
        #uso la classe LAM_lines solo per ottenere il percorso delle immagini (vedi full_path e imgs_dir dopo)
        train_dataset = LAM_Lines(lam_path, json_folder, ToTensor(), name_set)
        full_path = './DatabaseHandling/LAM/lines/img/'

        #creo una lista di percorsi associati alle immagini e prendo le label associate
        imgs_dir = [full_path + i['img'] for i in train_dataset.samples]
        imgs_labels = [i['text'] for i in train_dataset.samples]
        max_size = len(imgs_dir)
        assert (train_size + valid_size) <= max_size

        #questa funzione permette di eseguire il padding dell'immagine ad una dimensione prestabilita
        #se una immagine è più piccola della dimensione viene eseguito padding altrimenti (se più grande),
        #   la funzione resize la riporta alla dimensione voluta
        def preprocess(img):
            WHITE = [255, 255, 255]
            new_shape = (256, 64)
            h = img.shape[0]
            if h < new_shape[1]:
                img = cv.copyMakeBorder(img, 0, new_shape[1] - h, 0, 0, cv.BORDER_CONSTANT, value=WHITE)
            w = img.shape[1]
            if w < new_shape[0]:
                img = cv.copyMakeBorder(img, 0, 0, 0,  new_shape[0] - w, cv.BORDER_CONSTANT, value=WHITE)
            img = cv.resize(img, new_shape, interpolation = cv.INTER_AREA)
            img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            return img

        #questo ciclo permette di creare la lista di immagini da elaborare (questa funzione riempie la RAM quando si usa tutto il dataset)
        train_x = []
        train_labels = []
        for i in range(train_size):
            img_dir = imgs_dir[i]
            train_labels.append(imgs_labels[i])
            image = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)
            image = preprocess(image)
            image = image / 255
            train_x.append(image)

        #stessa cosa di prima ma con le immagini di validazione, più avanti sarà trasformata in funzione
        valid_x = []
        valid_labels = []
        for i in range(valid_size):
            img_dir = imgs_dir[-i-1]
            valid_labels.append(imgs_labels[-i-1])
            image = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)
            image = preprocess(image)
            image = image / 255
            valid_x.append(image)

        #l'implementazione da cui è preso questo codice dice di invertire la width con la height, NB: il canale è solo 1, quindi scala di grigi
        train_x = np.array(train_x).reshape(-1, 256, 64, 1)
        valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)

        #qui si viene a creare un nuovo set di caratteri, questo perchè ho voluto ridurre la dimensione
        #dell'alfabeto possibile andando a eliminare le maiuscole (vedi il metodo lower() sotto)
        new_set = {}
        idx = 0
        for k, v in train_dataset.char_to_idx.items():
            if k.lower() in new_set.keys():
                continue
            new_set[k.lower()] = idx
            idx += 1

        #stampa il set di caratteri che si possono predire dalla rete
        #print(new_set)

        max_str_len = max([len(i) for i in train_labels + valid_labels])
        num_of_characters = len(new_set) + 1
        num_of_timestamps = 128

        #questa funzione trasforma la stringa/label ground truth in tensore, dato il set di caratteri
        def label_to_num(charset, label):
            ret = []
            for ch in label.lower():
                val = charset.get(ch)
                assert val != None, f'val = {val}, ch = {ch}\nlabel = {label}'
                ret.append(val)
            return ret

        #questa funzione trasforma un tensore in stringa, dato il set di caratteri
        def num_to_label(charset, num):
            ret = ""
            temp_set = {v:k for k, v in charset.items()} #NB funziona solo se i values non si ripetono mai
            for ch in num:
                val = temp_set.get(ch)
                assert val != None, f'val = {val}, ch = {ch}\num = {num}'
                ret += val
            return ret

        #mostra un esempio di conversione stringa-tensore
        display_label_to_num = False
        if display_label_to_num:
            idx = 2
            print(new_set)
            print(train_labels[idx])
            print("length = ", len(train_labels[idx]))
            print(label_to_num(new_set, train_labels[idx]))

        #mostra un esempio di comversione tensore-stringa
        display_num_to_label = False
        if display_num_to_label:
            idx = 2
            #print(new_set)
            temp = train_labels[idx]
            print(temp)
            num = label_to_num(new_set, temp)
            print(f"len = {len(num)} --> ", num)
            label = num_to_label(new_set, num)
            print(label)

        #creazione di alcuni array di appoggio per la fase di training
        train_y = np.ones([train_size, max_str_len]) * -1
        train_label_len = np.zeros([train_size, 1])
        train_input_len = np.ones([train_size, 1]) * (num_of_timestamps - 2)
        train_output = np.zeros([train_size])

        for i in range(train_size):
            train_label_len[i] = len(train_labels[i])
            train_y[i, 0:len(train_labels[i])] = label_to_num(new_set, train_labels[i])

        #mostra solo i vettori creati
        display_train_vect = False
        if display_train_vect:
            print("train_y = ", train_y)
            print("train_label_len = ", train_label_len)
            print("train_input_len = ", train_input_len)
            print("train_output = ", train_output)

        #creazione di alcuni array di appoggio per la fase di validation
        valid_y = np.ones([valid_size, max_str_len]) * -1
        valid_label_len = np.zeros([valid_size, 1])
        valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps - 2)
        valid_output = np.zeros([valid_size])

        for i in range(valid_size):
            valid_label_len[i] = len(valid_labels[i])
            valid_y[i, 0:len(valid_labels[i])] = label_to_num(new_set, valid_labels[i])

        #mostra solo i vettori creati
        display_valid_vect = False
        if display_valid_vect:
            print("valid_y = ", valid_y)
            print("valid_label_len = ", valid_label_len)
            print("valid_input_len = ", valid_input_len)
            print("valid_output = ", valid_output)

        # /************** RETE NEURALE *******************/

        #se la dimensione dell'input cambia, va cambiato anche questa riga
        input_data = Input(shape=(256, 64, 1), name='input') #256 sarà poi la lunghezza della sequenza temporale a meno dei maxpooling
        #NB: la lunghezza (es 256) deve essere almeno più lunga della lunghezza massima del ground truth NB: --> 256 passa per i maxpooling quindi la lunghezza è in realtà 256 / 4

        inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner) # dim = 128 x 32

        inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner) #dim = 64 x 16
        inner = Dropout(0.3)(inner)

        inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner) #dim = 32 x 16 --> 32 è la sequenza temporale --> quindi è (1,2) per evitare di ridurre la lunghezza
        inner = Dropout(0.3)(inner)
        #gli iperparametri di questi strati convolutivi non dipendono dall'input

        # CNN to RNN
        #512 dipende dallo shape dell'input dopo i maxpooling
        inner = Reshape(target_shape=((128, 512)), name='reshape')(inner) #128 = numero di canali dalla canvolutiva
        inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

        ## RNN
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner) #256 = dimensione (altezza) della sequenza tempoerale, la lunghezza della sequenza temporale qua non è mostrata

        ## OUTPUT
        inner = Dense(num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
        y_pred = Activation('softmax', name='softmax')(inner) #dimensione di uscita = sequenza temporale * num of characters = 32 * 62
        #ciascuna riga contiene le 62 probabilità per ogni lettera

        model = Model(inputs=input_data, outputs=y_pred)
        #model.summary()

        tensor_board = {"histogram_freq" : 1}
        model_prefix = "test_prefix"
        model_path = f"{out_dir}/{model_prefix}.h5"
        callbacks = keras.callbacks.TensorBoard(log_dir=f"{out_dir}/logs", histogram_freq= 1)


        # Compilazione del modello con la funzione di perdita CTC
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='ctc', optimizer=optimizer, metrics=['accuracy'])



        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            # the 2 is critical here since the first couple outputs of the RNN
            # tend to be garbage
            y_pred = y_pred[:, 2:, :]
            return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

        labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)

        #passare metrica CER o WER per valutare correttezza rete
        model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate = 0.001),
                            metrics=['accuracy'])

        import tensorflow as tf
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        valid_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

        # Define our metrics
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
        test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

        def train_step(model, optimizer, x_train, y_train):
            with tf.GradientTape() as tape:
                predictions = model(x_train, training=True)
                loss = loss_object(y_train, predictions)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss(loss)
            train_accuracy(y_train, predictions)


        def test_step(model, x_test, y_test):
            predictions = model(x_test)
            loss = loss_object(y_test, predictions)

            test_loss(loss)
            test_accuracy(y_test, predictions)


        EPOCHS = 4
        test_dataset = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))

        for epoch in range(EPOCHS):
            for (x_train, y_train) in train_dataset:
                train_step(model_final, optimizer, x_train, y_train)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

            for (x_test, y_test) in test_dataset:
                test_step(model, x_test, y_test)
            with valid_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  train_accuracy.result() * 100,
                                  test_loss.result(),
                                  test_accuracy.result() * 100))

            # Reset metrics every epoch
            train_loss.reset_states()
            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()

        train_en = True
        if train_en:
            model_final.fit(x=[train_x, train_y, train_input_len, train_label_len], y=train_output,
                            validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
                            epochs=5, batch_size=5, callbacks=[callbacks])

    case 'kaggle2':
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2 as cv
        from DatabaseHandling.LAM import LAM_Lines
        from torchvision.transforms import ToTensor
        from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
        from keras import backend as K
        from keras.models import Model
        from keras.optimizers import Adam
        from tensorflow import keras
        import tensorflow as tf
        #permette di plottare 6 immagini con la rispettiva label/testo
        def show_imgs(imgs, labels):
            if len(imgs) > 6:
                imgs = imgs[0:6]
            for i in range(6):
                plt.subplot(2, 3, i+1)
                plt.imshow(cv.rotate(imgs[i], cv.ROTATE_90_COUNTERCLOCKWISE), cmap='gray')
                plt.title(labels[i])
                plt.axis('off')
            plt.show()

        #le dimensioni sono molto piccole per andare ad eseguire i test sulla compilazione
        #il numero di elementi ottenibile è circa 19000
        train_size = 20
        valid_size = 10

        json_folder = "basic"
        name_set = "train"
        lam_path = "./DatabaseHandling/LAM"
        #uso la classe LAM_lines solo per ottenere il percorso delle immagini (vedi full_path e imgs_dir dopo)
        train_dataset = LAM_Lines(lam_path, json_folder, ToTensor(), name_set)
        full_path = './DatabaseHandling/LAM/lines/img/'

        #creo una lista di percorsi associati alle immagini e prendo le label associate
        imgs_dir = [full_path + i['img'] for i in train_dataset.samples]
        imgs_labels = [i['text'] for i in train_dataset.samples]
        max_size = len(imgs_dir)
        assert (train_size + valid_size) <= max_size

        #questa funzione permette di eseguire il padding dell'immagine ad una dimensione prestabilita
        #se una immagine è più piccola della dimensione viene eseguito padding altrimenti (se più grande),
        #   la funzione resize la riporta alla dimensione voluta
        def preprocess(img):
            WHITE = [255, 255, 255]
            new_shape = (256, 64)
            h = img.shape[0]
            if h < new_shape[1]:
                img = cv.copyMakeBorder(img, 0, new_shape[1] - h, 0, 0, cv.BORDER_CONSTANT, value=WHITE)
            w = img.shape[1]
            if w < new_shape[0]:
                img = cv.copyMakeBorder(img, 0, 0, 0,  new_shape[0] - w, cv.BORDER_CONSTANT, value=WHITE)
            img = cv.resize(img, new_shape, interpolation = cv.INTER_AREA)
            img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            return img

        #questo ciclo permette di creare la lista di immagini da elaborare (questa funzione riempie la RAM quando si usa tutto il dataset)
        train_x = []
        train_labels = []
        for i in range(train_size):
            img_dir = imgs_dir[i]
            train_labels.append(imgs_labels[i])
            image = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)
            image = preprocess(image)
            image = image / 255
            train_x.append(image)

        #stessa cosa di prima ma con le immagini di validazione, più avanti sarà trasformata in funzione
        valid_x = []
        valid_labels = []
        for i in range(valid_size):
            img_dir = imgs_dir[-i-1]
            valid_labels.append(imgs_labels[-i-1])
            image = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)
            image = preprocess(image)
            image = image / 255
            valid_x.append(image)

        #l'implementazione da cui è preso questo codice dice di invertire la width con la height, NB: il canale è solo 1, quindi scala di grigi
        train_x = np.array(train_x).reshape(-1, 256, 64, 1)
        valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)

        #qui si viene a creare un nuovo set di caratteri, questo perchè ho voluto ridurre la dimensione
        #dell'alfabeto possibile andando a eliminare le maiuscole (vedi il metodo lower() sotto)
        new_set = {}
        idx = 0
        for k, v in train_dataset.char_to_idx.items():
            if k.lower() in new_set.keys():
                continue
            new_set[k.lower()] = idx
            idx += 1

        #stampa il set di caratteri che si possono predire dalla rete
        #print(new_set)

        max_str_len = max([len(i) for i in train_labels + valid_labels])
        num_of_characters = len(new_set) + 1
        num_of_timestamps = 128

        #questa funzione trasforma la stringa/label ground truth in tensore, dato il set di caratteri
        def label_to_num(charset, label):
            ret = []
            for ch in label.lower():
                val = charset.get(ch)
                assert val != None, f'val = {val}, ch = {ch}\nlabel = {label}'
                ret.append(val)
            return ret

        #questa funzione trasforma un tensore in stringa, dato il set di caratteri
        def num_to_label(charset, num):
            ret = ""
            temp_set = {v:k for k, v in charset.items()} #NB funziona solo se i values non si ripetono mai
            for ch in num:
                val = temp_set.get(ch)
                assert val != None, f'val = {val}, ch = {ch}\num = {num}'
                ret += val
            return ret

        #mostra un esempio di conversione stringa-tensore
        display_label_to_num = False
        if display_label_to_num:
            idx = 2
            print(new_set)
            print(train_labels[idx])
            print("length = ", len(train_labels[idx]))
            print(label_to_num(new_set, train_labels[idx]))

        #mostra un esempio di comversione tensore-stringa
        display_num_to_label = False
        if display_num_to_label:
            idx = 2
            #print(new_set)
            temp = train_labels[idx]
            print(temp)
            num = label_to_num(new_set, temp)
            print(f"len = {len(num)} --> ", num)
            label = num_to_label(new_set, num)
            print(label)

        #creazione di alcuni array di appoggio per la fase di training
        train_y = np.ones([train_size, max_str_len]) * -1
        train_label_len = np.zeros([train_size, 1])
        train_input_len = np.ones([train_size, 1]) * (num_of_timestamps - 2)
        train_output = np.zeros([train_size])

        for i in range(train_size):
            train_label_len[i] = len(train_labels[i])
            train_y[i, 0:len(train_labels[i])] = label_to_num(new_set, train_labels[i])

        #mostra solo i vettori creati
        display_train_vect = False
        if display_train_vect:
            print("train_y = ", train_y)
            print("train_label_len = ", train_label_len)
            print("train_input_len = ", train_input_len)
            print("train_output = ", train_output)

        #creazione di alcuni array di appoggio per la fase di validation
        valid_y = np.ones([valid_size, max_str_len]) * -1
        valid_label_len = np.zeros([valid_size, 1])
        valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps - 2)
        valid_output = np.zeros([valid_size])

        for i in range(valid_size):
            valid_label_len[i] = len(valid_labels[i])
            valid_y[i, 0:len(valid_labels[i])] = label_to_num(new_set, valid_labels[i])

        #mostra solo i vettori creati
        display_valid_vect = False
        if display_valid_vect:
            print("valid_y = ", valid_y)
            print("valid_label_len = ", valid_label_len)
            print("valid_input_len = ", valid_input_len)
            print("valid_output = ", valid_output)

        # /************** RETE NEURALE *******************/

        #se la dimensione dell'input cambia, va cambiato anche questa riga
        input_data = Input(shape=(256, 64, 1), name='input') #256 sarà poi la lunghezza della sequenza temporale a meno dei maxpooling
        #NB: la lunghezza (es 256) deve essere almeno più lunga della lunghezza massima del ground truth NB: --> 256 passa per i maxpooling quindi la lunghezza è in realtà 256 / 4

        inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner) # dim = 128 x 32

        inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner) #dim = 64 x 16
        inner = Dropout(0.3)(inner)

        inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner) #dim = 32 x 16 --> 32 è la sequenza temporale --> quindi è (1,2) per evitare di ridurre la lunghezza
        inner = Dropout(0.3)(inner)
        #gli iperparametri di questi strati convolutivi non dipendono dall'input

        # CNN to RNN
        #512 dipende dallo shape dell'input dopo i maxpooling
        inner = Reshape(target_shape=((128, 512)), name='reshape')(inner) #128 = numero di canali dalla canvolutiva
        inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

        ## RNN
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner) #256 = dimensione (altezza) della sequenza tempoerale, la lunghezza della sequenza temporale qua non è mostrata

        ## OUTPUT
        inner = Dense(num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
        y_pred = Activation('softmax', name='softmax')(inner) #dimensione di uscita = sequenza temporale * num of characters = 32 * 62
        #ciascuna riga contiene le 62 probabilità per ogni lettera

        model = Model(inputs=input_data, outputs=y_pred)
        #model.summary()


        # Compilazione del modello con la funzione di perdita CTC
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='ctc', optimizer=optimizer)


        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            # the 2 is critical here since the first couple outputs of the RNN
            # tend to be garbage
            y_pred = y_pred[:, 2:, :]
            return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

        labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        from keras.callbacks import TensorBoard

        tensor_board = TensorBoard(log_dir=f'{out_dir}/logs', histogram_freq=0, write_graph=True, write_images=True)

        ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)

        #passare metrica CER o WER per valutare correttezza rete
        model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate = 0.001))

        train_en = True
        if train_en:
            model_final.fit(x=[train_x, train_y, train_input_len, train_label_len], y=train_output,
                            validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
                            epochs=5, batch_size=5, shuffle=True, callbacks=tensor_board)

        #DOPO L'ESECUZIONE INSERIRE NEL TERMINALE IL COMANDO tensorboard --logdir ./OutDir/logs

    case 'parameters':
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2 as cv
        from DatabaseHandling.LAM import LAM_Lines
        from torchvision.transforms import ToTensor
        from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
        from keras import backend as K
        from keras.models import Model
        from keras.optimizers import Adam
        from tensorflow import keras

        #permette di plottare 6 immagini con la rispettiva label/testo
        def show_imgs(imgs, labels):
            if len(imgs) > 6:
                imgs = imgs[0:6]
            for i in range(6):
                plt.subplot(2, 3, i+1)
                plt.imshow(cv.rotate(imgs[i], cv.ROTATE_90_COUNTERCLOCKWISE), cmap='gray')
                plt.title(labels[i])
                plt.axis('off')
            plt.show()

        #le dimensioni sono molto piccole per andare ad eseguire i test sulla compilazione
        #il numero di elementi ottenibile è circa 19000
        train_size = 20
        valid_size = 10

        json_folder = "basic"
        name_set = "train"
        lam_path = "./DatabaseHandling/LAM"
        #uso la classe LAM_lines solo per ottenere il percorso delle immagini (vedi full_path e imgs_dir dopo)
        train_dataset = LAM_Lines(lam_path, json_folder, ToTensor(), name_set)
        full_path = './DatabaseHandling/LAM/lines/img/'

        #creo una lista di percorsi associati alle immagini e prendo le label associate
        imgs_dir = [full_path + i['img'] for i in train_dataset.samples]
        imgs_labels = [i['text'] for i in train_dataset.samples]
        max_size = len(imgs_dir)
        assert (train_size + valid_size) <= max_size

        #questa funzione permette di eseguire il padding dell'immagine ad una dimensione prestabilita
        #se una immagine è più piccola della dimensione viene eseguito padding altrimenti (se più grande),
        #   la funzione resize la riporta alla dimensione voluta
        def preprocess(img):
            WHITE = [255, 255, 255]
            new_shape = (256, 64)
            h = img.shape[0]
            if h < new_shape[1]:
                img = cv.copyMakeBorder(img, 0, new_shape[1] - h, 0, 0, cv.BORDER_CONSTANT, value=WHITE)
            w = img.shape[1]
            if w < new_shape[0]:
                img = cv.copyMakeBorder(img, 0, 0, 0,  new_shape[0] - w, cv.BORDER_CONSTANT, value=WHITE)
            img = cv.resize(img, new_shape, interpolation = cv.INTER_AREA)
            img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            return img

        #questo ciclo permette di creare la lista di immagini da elaborare (questa funzione riempie la RAM quando si usa tutto il dataset)
        train_x = []
        train_labels = []
        for i in range(train_size):
            img_dir = imgs_dir[i]
            train_labels.append(imgs_labels[i])
            image = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)
            image = preprocess(image)
            image = image / 255
            train_x.append(image)

        #stessa cosa di prima ma con le immagini di validazione, più avanti sarà trasformata in funzione
        valid_x = []
        valid_labels = []
        for i in range(valid_size):
            img_dir = imgs_dir[-i-1]
            valid_labels.append(imgs_labels[-i-1])
            image = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)
            image = preprocess(image)
            image = image / 255
            valid_x.append(image)

        #l'implementazione da cui è preso questo codice dice di invertire la width con la height, NB: il canale è solo 1, quindi scala di grigi
        train_x = np.array(train_x).reshape(-1, 256, 64, 1)
        valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)

        #qui si viene a creare un nuovo set di caratteri, questo perchè ho voluto ridurre la dimensione
        #dell'alfabeto possibile andando a eliminare le maiuscole (vedi il metodo lower() sotto)
        new_set = {}
        idx = 0
        for k, v in train_dataset.char_to_idx.items():
            if k.lower() in new_set.keys():
                continue
            new_set[k.lower()] = idx
            idx += 1

        #stampa il set di caratteri che si possono predire dalla rete
        #print(new_set)

        max_str_len = max([len(i) for i in train_labels + valid_labels])
        num_of_characters = len(new_set) + 1
        num_of_timestamps = 128

        #questa funzione trasforma la stringa/label ground truth in tensore, dato il set di caratteri
        def label_to_num(charset, label):
            ret = []
            for ch in label.lower():
                val = charset.get(ch)
                assert val != None, f'val = {val}, ch = {ch}\nlabel = {label}'
                ret.append(val)
            return ret

        #questa funzione trasforma un tensore in stringa, dato il set di caratteri
        def num_to_label(charset, num):
            ret = ""
            temp_set = {v:k for k, v in charset.items()} #NB funziona solo se i values non si ripetono mai
            for ch in num:
                val = temp_set.get(ch)
                assert val != None, f'val = {val}, ch = {ch}\num = {num}'
                ret += val
            return ret

        #mostra un esempio di conversione stringa-tensore
        display_label_to_num = False
        if display_label_to_num:
            idx = 2
            print(new_set)
            print(train_labels[idx])
            print("length = ", len(train_labels[idx]))
            print(label_to_num(new_set, train_labels[idx]))

        #mostra un esempio di comversione tensore-stringa
        display_num_to_label = False
        if display_num_to_label:
            idx = 2
            #print(new_set)
            temp = train_labels[idx]
            print(temp)
            num = label_to_num(new_set, temp)
            print(f"len = {len(num)} --> ", num)
            label = num_to_label(new_set, num)
            print(label)

        #creazione di alcuni array di appoggio per la fase di training
        train_y = np.ones([train_size, max_str_len]) * -1
        train_label_len = np.zeros([train_size, 1])
        train_input_len = np.ones([train_size, 1]) * (num_of_timestamps - 2)
        train_output = np.zeros([train_size])

        for i in range(train_size):
            train_label_len[i] = len(train_labels[i])
            train_y[i, 0:len(train_labels[i])] = label_to_num(new_set, train_labels[i])

        #mostra solo i vettori creati
        display_train_vect = False
        if display_train_vect:
            print("train_y = ", train_y)
            print("train_label_len = ", train_label_len)
            print("train_input_len = ", train_input_len)
            print("train_output = ", train_output)

        #creazione di alcuni array di appoggio per la fase di validation
        valid_y = np.ones([valid_size, max_str_len]) * -1
        valid_label_len = np.zeros([valid_size, 1])
        valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps - 2)
        valid_output = np.zeros([valid_size])

        for i in range(valid_size):
            valid_label_len[i] = len(valid_labels[i])
            valid_y[i, 0:len(valid_labels[i])] = label_to_num(new_set, valid_labels[i])

        #mostra solo i vettori creati
        display_valid_vect = False
        if display_valid_vect:
            print("valid_y = ", valid_y)
            print("valid_label_len = ", valid_label_len)
            print("valid_input_len = ", valid_input_len)
            print("valid_output = ", valid_output)

        # /************** RETE NEURALE *******************/

        #se la dimensione dell'input cambia, va cambiato anche questa riga
        input_data = Input(shape=(256, 64, 1), name='input') #256 sarà poi la lunghezza della sequenza temporale a meno dei maxpooling
        #NB: la lunghezza (es 256) deve essere almeno più lunga della lunghezza massima del ground truth NB: --> 256 passa per i maxpooling quindi la lunghezza è in realtà 256 / 4

        inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner) # dim = 128 x 32

        inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner) #dim = 64 x 16
        inner = Dropout(0.3)(inner)

        inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner) #dim = 32 x 16 --> 32 è la sequenza temporale --> quindi è (1,2) per evitare di ridurre la lunghezza
        inner = Dropout(0.3)(inner)
        #gli iperparametri di questi strati convolutivi non dipendono dall'input

        # CNN to RNN
        #512 dipende dallo shape dell'input dopo i maxpooling
        inner = Reshape(target_shape=((128, 512)), name='reshape')(inner) #128 = numero di canali dalla canvolutiva
        inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

        ## RNN
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner) #256 = dimensione (altezza) della sequenza tempoerale, la lunghezza della sequenza temporale qua non è mostrata

        ## OUTPUT
        inner = Dense(num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
        y_pred = Activation('softmax', name='softmax')(inner) #dimensione di uscita = sequenza temporale * num of characters = 32 * 62
        #ciascuna riga contiene le 62 probabilità per ogni lettera

        model = Model(inputs=input_data, outputs=y_pred)
        #model.summary()


        # Compilazione del modello con la funzione di perdita CTC
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='ctc', optimizer=optimizer)


        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            # the 2 is critical here since the first couple outputs of the RNN
            # tend to be garbage
            y_pred = y_pred[:, 2:, :]
            return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

        labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)

        #passare metrica CER o WER per valutare correttezza rete
        model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate = 0.001))

        train_en = True
        if train_en:
            model_final.fit(x=[train_x, train_y, train_input_len, train_label_len], y=train_output,
                            validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
                            epochs=5, batch_size=5)

        # Ottenere i pesi allenati del modello
        trained_weights = model_final.get_weights()
        #print(trained_weights)
        # Ora puoi salvare i pesi su disco o fare altre operazioni con essi
        # Ad esempio, puoi salvarli in un file numpy
        #import numpy as np

        #np.save('trained_weights.npy', trained_weights)

        #model_final.summary()


        #loss = model_final.evaluate(x=[train_x, train_y, train_input_len, train_label_len], y=train_output)
        #print('LOSS', loss)

        #training_loss = model_final.train_on_batch(x=[train_x, train_y, train_input_len, train_label_len], y=train_output)
        #print(training_loss)

        #configurazione completa del modello, inclusi i livelli, gli iperparametri e altre impostazioni
        #config = model_final.get_config()
        #print(config)

        #Per la funzione di perdita e l'accuratezza, è comune ottenerli durante l'addestramento utilizzando
        #il metodo train_on_batch o il metodo evaluate su dati di validation o test
        accuracy = model_final.train_on_batch(x=[train_x, train_y, train_input_len, train_label_len], y=train_output)
        str_accuracy = str(accuracy)
        print('Accuracy: ', str_accuracy[1])

        bias = model_final.get_weights()[1]
        print('Bias:', bias)

    case _:
        assert False, 'test non ammesso'


print("\n/*** End program ***/\n")


