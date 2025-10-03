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

    print("\n=== Verificando ImageMagick ===")
    if shutil.which("magick") is None:
        print("\n⚠️ ImageMagick não encontrado no PATH.")
        print("Baixe e instale a versão correta para Windows (64-bit Q16 DLL):")
        print("https://imagemagick.org/script/download.php#windows")
        print("Durante a instalação, marque:")
        print(" - Add application directory to your system PATH")
        print(" - Install legacy utilities (e.g., convert)")
    else:
        print("✅ ImageMagick encontrado no PATH.")

    print("\n✅ Setup concluído! Você pode rodar 'python run.py' agora.")

if __name__ == "__main__":
    main()
