import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Launcher para Streamlit com variáveis dinâmicas.")

    parser.add_argument(
        "--ffmpeg",
        required=True,
        help="Caminho completo para o ffmpeg.exe"
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Arquivo .py do Streamlit a ser executado (relativo ou absoluto)"
    )

    args = parser.parse_args()

    # Transformar caminho para absoluto
    target_file = os.path.abspath(args.file)

    if not os.path.isfile(target_file):
        print(f"\n❌ ERRO: O arquivo não existe:\n{target_file}\n")
        sys.exit(1)

    # Define a variável de ambiente
    os.environ["FFMPEG_PATH"] = args.ffmpeg

    # Comando final do Streamlit
    command = [
        sys.executable,
        "-m", "streamlit",
        "run",
        target_file
    ]

    print("\n=== Executando Streamlit ===")
    print(f"Arquivo: {target_file}")
    print(f"FFmpeg: {args.ffmpeg}\n")

    subprocess.run(command)

if __name__ == "__main__":
    main()
