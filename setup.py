# setup.py
import subprocess
import sys
import shutil

def run_cmd(cmd):
    """Executa comando no shell, mostrando saída em tempo real."""
    print(f"\n>>> Executando: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("=== Instalando dependências Python ===")

    # Remover pacotes conflitantes
    run_cmd("pip uninstall -y augmentation")

    # Instalar pacotes do requirements.txt
    run_cmd("pip install timm==0.5.4 fire validators lmdb pillow natsort torchvision tqdm wand opencv-python nltk six scikit-image")

    print("\n✅ Setup concluído! Você pode rodar 'python run.py' agora.")

if __name__ == "__main__":
    main()
