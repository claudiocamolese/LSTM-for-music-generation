import os
import torch
import random
import subprocess

# =========================
# Seed per riproducibilità
# =========================
def set_seed(number):
    torch.manual_seed(number)
    random.seed(number)

# =========================
# Salva una sequenza in formato ABC
# =========================
def save_song_to_abc(song, output_path, filename="generated_music"):
    # Crea la cartella se non esiste
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Percorso completo del file
    file_path = os.path.join(output_path, filename + ".abc")
    
    # Salva il file
    with open(file_path, "w") as f:
        f.write(song)
    
    return file_path

# =========================
# Converte un file ABC in WAV
# =========================


def abc2wav(abc_file, midi_dir="music_lstm/outputs/file/", wav_dir="music_lstm/outputs/audio"):
    # crea le cartelle se non esistono
    if not os.path.exists(midi_dir):
        os.makedirs(midi_dir)
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)

    # converte abc -> midi
    midi_file = os.path.join(midi_dir, os.path.basename(abc_file).replace(".abc", ".mid"))
    ret = os.system(f"abc2midi {abc_file} -o {midi_file}")
    if ret != 0 or not os.path.exists(midi_file):
        print("Conversion abc->midi failed")
        return None

    # converte midi -> wav
    wav_file = os.path.join(wav_dir, os.path.basename(abc_file).replace(".abc", ".wav"))
    ret = os.system(f"timidity {midi_file} -Ow -o {wav_file}")
    if ret != 0 or not os.path.exists(wav_file):
        print("Conversion midi->wav failed")
        return None

    return wav_file
