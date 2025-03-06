# SecurityLevel

**Descrição:** O **SecurityLevel** é um projeto de automação residencial composto por três módulos principais: sensoriamento, visão computacional e sistema web. Seu principal objetivo é proporcionar um sistema de segurança doméstico, oferecendo uma fechadura inteligente que utiliza reconhecimento facial como mecanismo de autenticação para liberar o acesso a determinados ambientes.

Este repositório, em específico, refere-se exclusivamente ao módulo de **visão computacional**, responsável pelo processamento e análise das imagens para autenticação do usuário.

---

## Índice

1. [Sobre o Projeto](#sobre-o-projeto)
2. [Tecnologias Utilizadas](#tecnologias-utilizadas)
3. [Instalação e Configuração](#instala%C3%A7%C3%A3o-e-configura%C3%A7%C3%A3o)
4. [Como Usar](#como-usar)
5. [Agradecimentos](#agradecimentos)
6. [Contato](#contato)

---

## Sobre o Módulo de Visão Computacional

O módulo é formado por códigos em Python que, juntos, são responsáveis por treinar um modelo de inteligência artificial (IA) capaz de realizar o reconhecimento facial. A IA realiza dois processos: treinamento e reconhecimento. O processo de treinamento gera um modelo capaz de reconhecer as faces para as quais foi treinado. Com esse modelo, ocorre o processo de reconhecimento do rosto em tempo real.

---

## Tecnologias Utilizadas

- Python - Versão 3.7.0
- TensorFlow - Versão 2.2.0
- Windows 11 (com C++ redistribuível) ([https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170))
- Facenet\_keras.h5 ([https://github.com/a-m-k-18/Face-Recognition-System/blob/master/facenet\_keras.h5](https://github.com/a-m-k-18/Face-Recognition-System/blob/master/facenet_keras.h5))

---

## Instalação e Configuração

### Pré-requisitos

- Python - Versão 3.7.0
- TensorFlow - Versão 2.2.0 ([https://www.tensorflow.org/install?hl=pt-br](https://www.tensorflow.org/install?hl=pt-br))
- Facenet\_keras.h5 ([https://github.com/a-m-k-18/Face-Recognition-System/blob/master/facenet\_keras.h5](https://github.com/a-m-k-18/Face-Recognition-System/blob/master/facenet_keras.h5))

---

## Como Usar

Instruções básicas para utilização do sistema.

### **Processo de Treinamento**

1. **Recortando as faces das fotos:** Crie uma pasta chamada `fotos` e, dentro dela, crie uma subpasta com o nome da pessoa que deseja treinar a IA para reconhecer. Dentro dessa subpasta, adicione as fotos da pessoa. Em seguida, crie também uma pasta chamada `faces`, onde ficarão armazenadas as faces recortadas das fotos. Altere os endereços no código **extract\_face.py** e execute-o.

2. **Gerando os embeddings das faces:** No código **facenet.py**, altere o endereço da **pasta de faces** e do **arquivo facenet\_keras.h5**. Após a execução, o código gerará um arquivo `.csv` com os embeddings das faces.

3. **Treinando e salvando o modelo de IA:** No código **avaliandomodelos.py**, altere o endereço do arquivo `.csv`. Ao fim da execução, será gerado um **modelo** com a extensão `.h5`. Caso necessário, remova o conjunto de validação.

### **Processo de Reconhecimento**

1. **Reconhecendo rostos com a WebCam:** No código **reconhecimentowebcanv1.py**, altere o nome da variável `pessoa` e os endereços dos arquivos **facenet\_keras.h5** e **modelo de treinamento** (atualmente chamado de `facesv1.h5`). Esse código realiza o reconhecimento em tempo real utilizando a webcam disponível.

2. **Melhorias:**

   - **Segunda versão do código:** No código **reconhecimentowebcanv2.py**, foram tratados os casos de **falso-positivo**, aumentando a porcentagem de confiança.

3. **Reconhecimento de imagem sem WebCam:**

   - O código **reconhecerImagem.py** usa o mesmo modelo de IA criado no processo de treinamento, mas realiza o reconhecimento apenas de imagens dentro de uma pasta específica, sem o uso da webcam.

4. **Machine:**

   - O código **machine.py** é uma compilação das etapas de treinamento e reconhecimento de imagem. Esse código executa todo o processo automaticamente para um conjunto de imagens.
   - O código **machine2.py** realiza apenas o reconhecimento de uma imagem específica e retorna o nome da face correspondente. Ele utiliza como base o modelo gerado anteriormente.

5. **Conexão com o Sistema Web:**

   - **API:** O código **api.py** recebe uma imagem do servidor web, executa o script **machine2.py** e retorna o nome da face correspondente ao servidor.

---

## Agradecimentos

O sistema de reconhecimento foi baseado nas aulas da playlist **Reconhecimento Facial Descomplicado** do canal **Sandeco** no YouTube. Expresso aqui minha gratidão pelo compartilhamento de conhecimento por parte do professor Sandeco.

---

## Contato

- **Nome:** Tomás Amorim
- **Email:** [tomasamorim46@gmail.com](mailto\:tomasamorim46@gmail.com)

