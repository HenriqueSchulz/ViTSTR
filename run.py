# run.py
import os
import sys

# Importar função para criar LMDB
from create_lmdb_dataset import createDataset

def main():
    # Caminho do projeto (pasta atual do script)
    project_path = os.path.abspath(os.path.dirname(__file__))

    # Dataset
    dataset_folder = os.path.join(project_path, "dataset_files", "testA")
    labels_file = os.path.join(dataset_folder, "labels.txt")
    lmdb_output = os.path.join(dataset_folder, "test_a_lmdb")
    
    '''
    # Limpar labels.txt
    if os.path.exists(labels_file):
        cleaned_lines = []
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # pula linhas vazias
                parts = line.split(" ", 1)  # separa no primeiro espaço
                if len(parts) != 2:
                    print(f"⚠ Linha ignorada: {line}")
                    continue
                image_name, text = parts
                # Remove "images/" se presente e adiciona tab
                image_name = image_name.replace("images/", "")
                cleaned_lines.append(f"{image_name}\t{text}")
        with open(labels_file, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned_lines))
    '''

    # Criar LMDB se não existir
    if not os.path.exists(lmdb_output):
        print(f"📦 Criando LMDB em: {lmdb_output}")
        createDataset(
            inputPath=os.path.join(dataset_folder, "images"),
            gtFile=labels_file,
            outputPath=lmdb_output
        )
    else:
        print(f"✅ LMDB já existe em: {lmdb_output}")

    # Rodar modelo ViTSTR
    sys.path.append(project_path)  # garante que imports do projeto funcionem
    import test  # o script test.py do projeto

    print(lmdb_output)

    # Ajuste os argumentos diretamente
    args = [
        "--eval_data", lmdb_output,
        "--Transformation", "None",
        "--FeatureExtraction", "None",
        "--SequenceModeling", "None",
        "--Prediction", "None",
        "--Transformer",
        "--TransformerModel", "vitstr_base_patch16_224",
        "--sensitive",
        "--data_filtering_off",
        "--imgH", "224",
        "--imgW", "224",
        "--saved_model", os.path.join(project_path, "saved_models", "vitstr_tiny_patch16_224_aug.pth")
    ]

    # Test.py espera sys.argv
    sys.argv = ["test.py"] + args
    test.main()  # chama a função principal do test.py

if __name__ == "__main__":
    main()
