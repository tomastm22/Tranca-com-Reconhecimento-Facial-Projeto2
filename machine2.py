import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
import cv2
import os

caminho_pessoa = r"C:/Users/tmamo/Documents/Projeto2-Novo/IA/betateste/arquivos/Fotos"
#pessoa = [nome for nome in os.listdir(caminho_pessoa) if os.path.isdir(os.path.join(caminho_pessoa, nome))]
pessoa = ["ALISON", "JULIO", "TOMAS"]
num_classes = len(pessoa)
            #C:\Users\tmamo\Documents\Projeto2-Novo\IA\betateste\arquivos\Fotos\tomas
image_dir = "C:/Users/tmamo/Documents/Projeto2-Novo/IA/betateste/arquivos/Fotos/tomas"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

if not image_files:
    print("Erro: Nenhuma imagem encontrada na pasta.")
    exit()

detector = MTCNN()
facenet = load_model("C:/Users/tmamo/Documents/Projeto2-Novo/IA/betateste/codigo/facenet_keras.h5")
                    #C:\Users\tmamo\Documents\Projeto2-Novo\IA\codigo\facesv1
model = load_model("C:/Users/tmamo/Documents/Projeto2-Novo/IA/codigo/facesv1.h5")

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

            if prob > 98:
                user = str(pessoa[classe]).upper()
            else:
                user = "DESCONHECIDO"
    
    #print(f"Imagem: {image_file} - Identificado: {user}")
    print(user)