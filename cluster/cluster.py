#!/usr/bin/env python3
"""
Gera imagens chamando ./a.out com seeds aleatórias, converte PGM->PNG
e salva pares 00034.png / 00034.txt contendo o Z exibido pelo executável.

Uso: python cluster.py [--start 34] [--count 4] [--aout ./a.out]
"""
import argparse
import os
import re
import subprocess
import sys
import shutil


def gen_seed():
    return int.from_bytes(os.urandom(4), "little")


def extract_pgm_name(output):
    m = re.search(r"Salvo:\s*([^\s]+\.pgm)", output)
    if m:
        return m.group(1)
    # fallback: look for any .pgm filename
    m2 = re.search(r"(\S+\.pgm)", output)
    return m2.group(1) if m2 else "saida.pgm"


def extract_z(output):
    # try common patterns: Z = [..] or Z: [..]
    m = re.search(r"Z\s*[:=]\s*(\[[^\]]+\])", output, re.IGNORECASE)
    if m:
        return m.group(1)
    # find a long bracketed list (likely the latent vector)
    m2 = re.findall(r"\[[^\]]+\]", output)
    for cand in m2:
        nums = re.findall(r"-?\d+\.?\d*", cand)
        if len(nums) >= 8:
            return cand
    # last resort: return whole stdout/stderr trimmed
    return output.strip()


def run_once(aout_path, index, convert_cmd):
    seed = index
    # choose an explicit output pgm per run to avoid races / reuse
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    pgm = os.path.join(out_dir, f"saida_{index:05d}.pgm")
    cmd = [aout_path, pgm, str(seed)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise RuntimeError(f"Executável não encontrado: {aout_path}")

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

    # extract Z from program output (if present)
    z = extract_z(combined)

    out_base = f"{index:05d}"
    png_name = os.path.join(out_dir, out_base + ".png")
    txt_name = os.path.join(out_dir, out_base + ".txt")

    if not os.path.exists(pgm):
        # if the expected file wasn't created, try the default saida.pgm
        if os.path.exists('saida.pgm'):
            shutil.copyfile('saida.pgm', pgm)
        else:
            # fall back to most recent .pgm
            candidates = [f for f in os.listdir(out_dir) if f.lower().endswith('.pgm')]
            if candidates:
                candidates.sort(key=lambda p: os.path.getmtime(os.path.join(out_dir, p)), reverse=True)
                pgm = os.path.join(out_dir, candidates[0])
            else:
                raise RuntimeError(f"Arquivo PGM esperado não existe: {pgm}\nPrograma output:\n{combined}")

    # convert to png
    conv_cmd = [convert_cmd, pgm, png_name]
    subprocess.run(conv_cmd, check=True)

    # save Z
    with open(txt_name, "w") as f:
        f.write(z + "\n")

    return png_name, txt_name, seed


def main():
    # Apaga arquivos .pgm da pasta data se solicitado
    import glob
    if '--delete-pgm' in sys.argv:
        pgm_files = glob.glob(os.path.join('data', '*.pgm'))
        for f in pgm_files:
            try:
                os.remove(f)
                print(f"Removido: {f}")
            except Exception as e:
                print(f"Erro ao remover {f}: {e}")
        return
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=34, help="Índice inicial (ex: 34 -> 00034.png)")
    p.add_argument("--count", type=int, default=4, help="Número de imagens a gerar")
    p.add_argument("--aout", default="./a.out", help="Caminho para o executável a.out")
    p.add_argument("--convert", default="convert", help="Comando de conversão (ImageMagick)")
    args = p.parse_args()

    cur = args.start
    for i in range(args.count):
        try:
            png, txt, seed = run_once(args.aout, cur + i, args.convert)
            print(f"Gerado: {png}, {txt} (seed={seed})")
        except Exception as e:
            print("Erro ao gerar imagem:", e, file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
