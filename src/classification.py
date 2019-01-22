########################################################################################################################

#Wczytanie danych treningowych i rozdzielenie ich na dwie listy
#Lista train_question_text będzie zawierać pytania, a lista train_labels oznaczenie tych pytań (0 lub 1)
import csv
train_question_text = []
train_labels = []
first = True
with open('../input/train.csv', newline='', encoding="utf8") as csvfile:
    r = csv.reader(csvfile, delimiter=',')
    for row in r:
        if first:#Pomijam pierwszy wiersz, ponieważ zawiera nagłówki kolumn
            first = False
        else:
            train_question_text.append(row[1])#Dodanie do listy pytania
            train_labels.append(int(row[2]))#Dodanie do listy oznaczenia pytania (0 lub 1)

			
#Wczytanie pytań ze zbioru testowego
#Lista test_question_text będzie zawierać pytania
test_question_text = []
test_qid = []
first = True
with open('../input/test.csv', newline='', encoding="utf8") as csvfile:
    r = csv.reader(csvfile, delimiter=',')
    for row in r:
        if first:#Pomijam pierwszy wiersz, ponieważ zawiera nagłówki kolumn
            first = False
        else:
            test_question_text.append(row[1])#Dodanie do listy pytania
            test_qid.append(row[0])


#Wczytanie oznaczeń pytań zbioru testowego
#Lista test_labels będzie zawierać oznaczenia pytań zbioru testowego (0 lub 1)			
test_labels = []
first = True
with open('../input/sample_submission.csv', newline='', encoding="utf8") as csvfile:
    r = csv.reader(csvfile, delimiter=',')
    for row in r:
        if first:#Pomijam pierwszy wiersz, ponieważ zawiera nagłówki kolumn
            first = False
        else:
            test_labels.append(row[1])#Dodanie do listy oznaczenia pytania zbioru treningowego(0 lub 1)


#Zamiana słów w pytaniach na liczby całkowite i stworzenie słownika		
word_index = {}#Słownik
train_data = []
pom = 1;#wartość 1 dla pierwszego klucza (słowa) w słowniku
for row in train_question_text:
    list_word = row.split( )#Podzielenie pytania na listę słów
    liczby = []#Lista pomocnicza, która będzie zawierała ciąg liczb całkowitych reprezentujących słowa w danym pytaniu.
    for l in list_word:
        if l in word_index:
            liczby.append(word_index.get(l) + 3)#Dodanie do listy pomocniczej liczby całkowitej odpowiadającej danemu słowu w słowniku. 
        else:									#Wartość ta zwiększona jest o 3, ponieważ później do słownika dodawane są 4 klucze o wartości 0, 1, 2 i 3
            word_index[l] = pom
            liczby.append(pom + 3)
            pom = pom + 1
    train_data.append(liczby)#Dodanie listy pomocniczej do głównej listy

	
#Zamiana słów w pytaniach na liczby całkowite zbioru testowego
test_data = []
for row in test_question_text:
    list_word = row.split( )
    liczby = []
    for l in list_word:
        if l in word_index:
            liczby.append(word_index.get(l) + 3)
        else:
            word_index[l] = pom
            liczby.append(pom + 3)
            pom = pom + 1
    test_data.append(liczby)
	
########################################################################################################################


#### based on: https://www.tensorflow.org/tutorials/keras/basic_text_classification

import tensorflow as tf
from tensorflow import keras


### Get data
#imdb = keras.datasets.imdb#####################################################################
#(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)#########


### Convert the integers to words
# A dictionary mapping words to an integer index
#word_index = imdb.get_word_index()#############################################################

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


	
### Our code

print("Training entries length: {}, labels length: {}".format(len(train_data), len(train_labels)))
print("Test entries length: {}, labels length: {}".format(len(test_data), len(test_labels)))
print("Example record")
print("raw: {}".format(train_data[0]))
print("as word: {}".format(decode_review(train_data[0])))
print("label: {}".format(train_labels[0]))
print("Legend: 0 is a negative review, and 1 is a positive review")


### Data preprocessing
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

### Metryki - ocena sieci
K = keras.backend

## Precyzja
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

## Czułość
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

## Specyficzność
def specificity(y_pred, y_true):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    false_positive = K.sum(neg_y_true * y_pred)
    true_negative = K.sum(neg_y_true * neg_y_pred)
    specificity = true_negative / (true_negative + false_positive + K.epsilon())
    return specificity

### Model
vocab_size = len(word_index) #600000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mean_squared_error',
              metrics=['accuracy',precision, recall, specificity])

			  
### Training and testing	

splitFactor =int(len(train_labels)*0.3) #30% to validate data

		  
x_val = train_data[:splitFactor]
partial_x_train = train_data[splitFactor:]

y_val = train_labels[:splitFactor]
partial_y_train = train_labels[splitFactor:]

import numpy as np
test_labelsNpArray = np.array(test_labels).astype(int)


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
					
results = model.evaluate(test_data, test_labelsNpArray)

prediction= model.predict_classes(test_data, batch_size=32, verbose=1)

predictionVector = prediction.flatten()

import pandas as pd
submission= pd.DataFrame(
        {'qid': test_qid, 'prediction':predictionVector},
        columns = ['qid', 'prediction'])
submission.to_csv('submission.csv', index=False)

### Miary jakości
loss = round(results[0]*100,2)
acc = round(results[1]*100,2)
prec = round(results[2]*100,2)
rec = round(results[3]*100,2)
spec = round(results[4]*100,2)
print('\nLoss: ', loss, '%') # strata
print('Accuracy: ', acc, '%') # dokładność
print('Precision: ', prec, '%') # precyzja
print('Recall: ', rec, '%') # czułość
print('Specificity: ', spec, '%') # specyficzność