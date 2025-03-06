from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import subprocess

app = FastAPI()

# Diretórios e arquivos
UPLOAD_DIR = r"C:\Users\tmamo\Documents\Projeto2-Novo\IA\betateste\arquivos\Fotos\tomas"
SCRIPT_PATH = r"C:\Users\tmamo\Documents\Projeto2-Novo\IA\betateste\codigo\machine2.py"
OUTPUT_FILE = r"C:\Users\tmamo\Documents\Projeto2-Novo\IA\betateste\codigo\saida.txt"

os.makedirs(UPLOAD_DIR, exist_ok=True)  # Garante que o diretório existe

@app.post("/reconhecer")
async def reconhecer_imagem(file: UploadFile = File(...)):
    try:
        # Define o caminho completo para salvar o arquivo
        file_path = Path(UPLOAD_DIR) / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Executa o script machine.py e redireciona a saída para um arquivo
        with open(OUTPUT_FILE, "w") as output:
            subprocess.run(["python", SCRIPT_PATH], stdout=output, stderr=subprocess.STDOUT, text=True)

        # Lê a última linha do arquivo de saída
        with open(OUTPUT_FILE, "r") as output:
            lines = output.readlines()
            last_line = lines[-1].strip() if lines else "Nenhuma saída gerada"
        
        # Apagar a imagem salva após o processamento
        if file_path.exists():
            os.remove(file_path)

        return JSONResponse(content={"mensagem": last_line}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a imagem: {str(e)}")