import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
import cv2


pessoa = ["ALISON", "JULIO", "TOMAS"]
num_classes = len(pessoa)
cap = cv2.VideoCapture(0)


detector = MTCNN()
facenet = load_model("facenet_keras.h5")
model = load_model("facesv1.h5")

def extract_face(image, box, required_size=(160,160)):

  pixels = np.asarray(image)

  x1, y1, width, height = box

  x2, y2 = x1 + width, y1 + height
  # extract the face
  face = pixels[y1:y2, x1:x2]
  # resize pixels to the model size
  image = Image.fromarray(face)
  image = image.resize(required_size)
  return np.asarray(image)

def get_embedding(facenet, face_pixels):

  face_pixels = face_pixels.astype('float32')

  mean, std = face_pixels.mean(), face_pixels.std()
  face_pixels = (face_pixels - mean) / std

  samples = np.expand_dims(face_pixels, axis=0)

  yhat = facenet.predict(samples)
  return yhat[0]

while True:

  _, frame = cap.read()

  faces = detector.detect_faces(frame)

  for face in faces:

    confidence = face['confidence']*100

    if confidence>=98:

      x1, y1, w, h = face['box']
      face = extract_face(frame, face['box'])

      face = face.astype("float32")/255

      emb = get_embedding(facenet, face)

      tensor = np.expand_dims(emb, axis=0)

      classe = model.predict_classes(tensor)[0]
      prob = model.predict_proba(tensor)
      prob = prob[0][classe]*100

      # PEGANDO O USUARIO
      user = str(pessoa[classe]).upper()

      # COR DO QUADRADO
      color = (192, 255, 119)

      cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), color, 2)
      # FONTE DO TEXTO
      font = cv2.FONT_HERSHEY_SIMPLEX
      # TAMANHO DA FONTE
      font_scale = 0.5

      cv2.putText(frame, user, (x1, y1-10), font, fontScale=font_scale, color=color,
                  thickness=1)
  
  cv2.imshow("FACE RECOGNITION", frame)

  key = cv2.waitKey(1)

  if key==27: #ESC
    break

cap.release()
cv2.destroyAllWindows()









