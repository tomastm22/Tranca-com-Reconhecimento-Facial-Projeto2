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
    load_dir(
        "C:/Users/tmamo/Documents/Projeto2-Novo/IA/arquivos/Fotos",
        "C:/Users/tmamo/Documents/Projeto2-Novo/IA/arquivos/Faces"
    )
