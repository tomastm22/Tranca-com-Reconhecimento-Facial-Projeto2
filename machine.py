import os
import shutil

# COPIA A IMAGEM RECEBIDA PARA A PASTA "IDENTIFICAR"
def copiar_imagem():
    origem = r"C:\Users\tmamo\Documents\Projeto2-Novo\IA\betateste\arquivos\Fotos\tomas" # FUNCIONAL
    destino = r"C:\Users\tmamo\Documents\Projeto2-Novo\IA\betateste\arquivos\Identificar" # FUNCIONAL
    
    try:
        for arquivo in os.listdir(origem):
            if arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                caminho_origem = os.path.join(origem, arquivo)
                caminho_destino = os.path.join(destino, arquivo)
                shutil.copy2(caminho_origem, caminho_destino)
                print(f"Imagem '{arquivo}' copiada para '{destino}'.")
                criar_copias(caminho_origem, origem)
    except Exception as e:
        print(f"Erro ao copiar a imagem: {e}")

# CRIA 12 COPIAS DA IMAGEM
def criar_copias(arquivo_origem, pasta_destino):
    nome_base, extensao = os.path.splitext(os.path.basename(arquivo_origem))
    
    try:
        for i in range(1, 13):
            novo_nome = f"{nome_base}_copia{i}{extensao}"
            caminho_copia = os.path.join(pasta_destino, novo_nome)
            shutil.copy2(arquivo_origem, caminho_copia)
            print(f"Cópia '{novo_nome}' criada em '{pasta_destino}'.")
    except Exception as e:
        print(f"Erro ao criar cópias: {e}")

# CÓDIGO ESTRATOR DE FACES
from mtcnn import MTCNN  # Reconhece Faces
from PIL import Image  # Manipular imagem
from os import listdir, makedirs  # Listar diretório e criar diretórios
from os.path import isdir, exists  # Confirmar se é diretório e verificar se existe
from numpy import asarray  # Converter uma imagem PIL em array

detector = MTCNN()  # Inicializa o detector de rostos


def extrair_face(arquivo, size=(160, 160)):
    try:
        img = Image.open(arquivo)  # Abre a imagem
        img = img.convert('RGB')  # Converte para RGB
        array = asarray(img)  # Converte para array numpy
        results = detector.detect_faces(array)

        if not results:  # Verifica se encontrou um rosto
            print(f"Nenhum rosto detectado em {arquivo}")
            return None

        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height

        face = array[y1:y2, x1:x2]  # Recorta a face
        image = Image.fromarray(face)  # Converte de volta para imagem PIL
        image = image.resize(size)  # Redimensiona a imagem
        return image

    except Exception as e:
        print(f"Erro ao processar {arquivo}: {e}")
        return None


def load_fotos(directory_src, directory_target):
    if not exists(directory_target):
        makedirs(directory_target)  # Cria diretório se não existir

    for filename in listdir(directory_src):
        path = f"{directory_src}/{filename}"
        path_tg = f"{directory_target}/{filename}"

        if not isdir(path):  # Evita processar diretórios dentro da pasta
            try:
                face = extrair_face(path)
                if face:
                    face.save(path_tg, "JPEG", quality=100, optimize=True, progressive=True)
                    print(f"Face salva em {path_tg}")
            except Exception as e:
                print(f"Erro ao salvar {filename}: {e}")


def load_dir(directory_src, directory_target):
    for subdir in listdir(directory_src):
        path = f"{directory_src}/{subdir}"
        path_tg = f"{directory_target}/{subdir}"

        if isdir(path):  # Apenas processa diretórios
            if not exists(path_tg):
                makedirs(path_tg)  # Cria diretório de destino
            load_fotos(path, path_tg)


if __name__ == "__main__":
    # COPIA A IMAGEM PARA A PASTA "IDENTIFICAR"
    # CRIA 12 COPIAS DA IMAGEM
    copiar_imagem()
    # CARREGA O EXTRATOR DE FACES
    load_dir(
        "C:/Users/tmamo/Documents/Projeto2-Novo/IA/betateste/arquivos/Fotos",
        "C:/Users/tmamo/Documents/Projeto2-Novo/IA/betateste/arquivos/Faces"
    )

# GERADOR DE EMBEDDINGS
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
                                            #C:\Users\tmamo\Documents\Projeto2-Novo\IA\betateste\arquivos
trainX, trainy = load_fotos(directory_src = "C:\\Users\\tmamo\\Documents\\Projeto2-Novo\\IA\\betateste\\arquivos\\Faces\\")

from tensorflow.keras.models import load_model
                    #C:\Users\tmamo\Documents\Projeto2-Novo\IA\betateste\codigo
model = load_model('C:\\Users\\tmamo\\Documents\\Projeto2-Novo\\IA\\betateste\\codigo\\facenet_keras.h5')

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
# print(df)
df['target'] = trainy
# print(df)

df.to_csv('embeddingsbetateste.csv')

# TREINANDO COM OS EMBEDDINGS

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Carregar dados de treinamento
df = pd.read_csv("embeddingsbetateste.csv")
print(df)

X = np.array(df.drop(columns=["Unnamed: 0", "target"]))
print(X.shape)

y = np.array(df.target)
print(y)
print(y.shape)

# Misturando os dados
from sklearn.utils import shuffle

trainX, trainY = shuffle(X, y, random_state=0)
print(trainY)

# Tratar labels
from sklearn.preprocessing import LabelEncoder

out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)  # Discretização ou binarização
print(trainY)

# Usando o Keras
from tensorflow.keras.utils import to_categorical

trainY = to_categorical(trainY)

from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(128,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="softmax"))

model.summary()

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['accuracy'])

model.fit(trainX, trainY, epochs=100, batch_size=8)

model.save('modelobetateste.h5')

# RECONHECENDO COM O MODELO TREINADO

import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
import cv2
import os

caminho_pessoa = r"C:/Users/tmamo/Documents/Projeto2-Novo/IA/betateste/arquivos/Fotos"
pessoa = [nome for nome in os.listdir(caminho_pessoa) if os.path.isdir(os.path.join(caminho_pessoa, nome))]
# pessoa = ["ALISON", "JULIO", "TOMAS"]
num_classes = len(pessoa)
            #C:\Users\tmamo\Documents\Projeto2-Novo\IA\betateste\arquivos\Identificar
image_dir = "C:/Users/tmamo/Documents/Projeto2-Novo/IA/betateste/arquivos/Identificar"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

if not image_files:
    print("Erro: Nenhuma imagem encontrada na pasta.")
    exit()

detector = MTCNN()
facenet = load_model("C:/Users/tmamo/Documents/Projeto2-Novo/IA/betateste/codigo/facenet_keras.h5")
model = load_model("C:/Users/tmamo/Documents/Projeto2-Novo/IA/betateste/modelobetateste.h5")

def extract_face(image, box, required_size=(160, 160)):
    pixels = np.asarray(image)
    x1, y1, width, height = box
    x2, y2 = x1 + width, y1 + height
    face = pixels[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    return np.asarray(image)

def get_embedding(facenet, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    if std == 0:
        std = 1
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = facenet.predict(samples)
    return yhat[0]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem {image_file}.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)

    user = "DESCONHECIDO"
    for face in faces:
        confidence = face['confidence'] * 100
        x1, y1, w, h = face['box']

        if confidence >= 98:
            face_pixels = extract_face(image_rgb, face['box'])
            face_pixels = face_pixels.astype("float32") / 255
            emb = get_embedding(facenet, face_pixels)
            tensor = np.expand_dims(emb, axis=0)
            predictions = model.predict(tensor)
            classe = np.argmax(predictions)
            prob = predictions[0][classe] * 100

            if prob > 80:
                user = str(pessoa[classe]).upper()
            else:
                user = "DESCONHECIDO"
    
    print(f"Imagem: {image_file} - Identificado: {user}")
