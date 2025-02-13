from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray, expand_dims
import numpy as np
import pandas as pd

def load_face(filename):
  # carregando imagem de arquivos
  image = Image.open(filename)

  # converter para RGB
  image = image.convert("RGB")

  return asarray(image)

# Carregando imagens de face de um diretório
def load_faces(directory_src):

  faces = list()

  # iterando arquivos
  for filename in listdir(directory_src):

    path = directory_src + filename

    try:
      faces.append(load_face(path))
    except:
      print("Erro na imagem {}".format(path))

  return faces

## Carregando todo o dataset de imagens de faces

def load_fotos(directory_src):

  X, y = list(), list()

  # iterar pastas por classes
  for subdir in listdir(directory_src):

    #path
    path = directory_src + subdir + '\\'

    if not isdir(path):
      continue

    faces = load_faces(path)

    labels = [subdir for _ in range(len(faces))]

    # sumarizar progresso
    print('>Carregadas %d faces da classe: %s' % (len(faces), subdir))

    X.extend(faces)
    y.extend(labels)
  
  return asarray(X), asarray(y)

## CARREGANDO TODAS AS IMAGENS

trainX, trainy = load_fotos(directory_src = "C:\\Users\\tmamo\\Documents\\Projeto2-Novo\\IA\\arquivos\\Faces\\")

from tensorflow.keras.models import load_model

model = load_model('C:\\Users\\tmamo\\Documents\\Projeto2-Novo\\IA\\facenet\\facenet_keras_3.h5')

model.summary()

# FUNÇÃO GERADORA DE EMBEDDINGS

def get_embedding(model, face_pixels):

  # PADRONIZAÇÃO
  mean, std = face_pixels.mean(), face_pixels.std()
  face_pixels = (face_pixels - mean)/std

  # TRANSFORMAR A FACE EM 1 ÚNICO EXEMPLO

  # (160,160) -> (1,160,160)

  samples = expand_dims(face_pixels, axis=0)

  # REALIZAR A PREDIÇÃO GERANDO O EMBEDDING
  yhat = model.predict(samples)

  # [[1,2...128], [1,2..128]]

  return yhat[0]

# GERANDO TODAS AS EMBEDDINGS

newTrainX = list()

for face in trainX:
  embedding = get_embedding(model, face)
  newTrainX.append(embedding)

newTrainX = asarray(newTrainX)

newTrainX.shape

# DataFrame

df = pd.DataFrame(data=newTrainX)
print(df)
df['target'] = trainy
print(df)

df.to_csv('faces.csv')




      
