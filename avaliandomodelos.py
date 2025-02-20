import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("embeddings.csv")
#print(df)

X = np.array(df.drop(columns=["Unnamed: 0", "target"]))

#print(X.shape)

y = np.array(df.target)

#print(y)

#print(y.shape)

# MISTURANDO TUDO

from sklearn.utils import shuffle

trainX, trainY = shuffle(X, y, random_state=0)

# print(trainY)

# TRATAR LABELS

from sklearn.preprocessing import LabelEncoder

out_encoder = LabelEncoder()

out_encoder.fit(trainY)

trainY = out_encoder.transform(trainY) # VAI TRANSFORMAR NOMES EM NÚMEROS -> DISCRETIZAÇÃO OU BINARIZAÇÃO

#print(trainY)

## VALIDAÇÃO

df_val = pd.read_csv("embeddings-test.csv") # CSV DA VALIDAÇÃO

#print(df_val)

valX = np.array(df_val.drop(columns=["Unnamed: 0", "target"]))

#print(valX.shape)

valY = np.array(df_val.target)

# print(valY)

out_encoder.fit(valY)

valY = out_encoder.transform(valY)

# print(valY)

# AVALIAÇÃO DE ALGORITMOS

# Avaliando o KNN

# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=5)

# knn.fit(trainX, trainY)

# Resultados do treino

# yhat_train = knn.predict(trainX)
# yhat_val = knn.predict(valX)

# print(yhat_val)

# from sklearn.metrics import confusion_matrix

# def print_confusion_matrix(model1_name, valY, yhat_val):

  #cm = confusion_matrix(valY, yhat_val)
  #total = sum(sum(cm))
 # acc = (cm[0, 0] + cm[1, 1]) / total
#  sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
#  specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

#  print("MODELO : {}".format(model1_name))
#  print("Acurácia: {:.4f}".format(acc))
#  print("Sensitividade: {:.4f}".format(sensitivity))
 # print("Especificidade: {:.4f}".format(specificity))

  # from mlxtend.plotting import plot_confusion_matrix
  # fig, ax = plot_confusion_matrix(conf_mat=cm , figsize = (5, 5))
  # plt.show()

# print_confusion_matrix("KNN", valY, yhat_val)

# TESTANDO O SVM

#from sklearn import svm

#svm = svm.SVC()

#svm.fit(trainX, trainY)

# Treinando o SVM

#yhat_train = svm.predict(trainX)
#yhat_val = svm.predict(valX)

# print_confusion_matrix("SVM", valY, yhat_val)

# USANDO O KERAS

from tensorflow.keras.utils import to_categorical

trainY = to_categorical(trainY)

valY = to_categorical(valY)

from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(128,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation="softmax"))

model.summary()

model.compile(optimizer="adam",
              loss= "categorical_crossentropy",
              metrics=['accuracy'])

model.fit(trainX, trainY, epochs=100, batch_size=8)

yhat_train = model.predict(trainX)
yhat_val = model.predict(valX)

# TRANSFORMANDO A PROBABILIDADE EM UM VALOR ABSOLUTO

yhat_val = np.argmax(yhat_val, axis=1)
print(yhat_val)

valY = np.argmax(valY, axis=1)
print(valY)

model.save('facesv1.h5')
