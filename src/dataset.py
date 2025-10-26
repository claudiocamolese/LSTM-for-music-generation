import os
import regex
import numpy as np
import torch
import random
from torch.utils.data import Dataset

class Datasets(Dataset):
    def __init__(self, dataset_name, seq_len=100, percentage= None, device='cpu',):
        super().__init__()
        self.path = os.path.dirname(__file__)
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.device = device
        self.percentage = percentage

        if self.dataset_name == 'irish':
            self.songs = self._load_irish_data()
        elif self.dataset_name == 'pop':
            self.songs = self._load_pop_data()     
        elif self.dataset_name == "mix":
            self.songs = self._load_mix(percentage= self.percentage) 
        else:
            raise ValueError(f"Dataset '{dataset_name}' non supportato.")

        # total number of token concatenated in my dataset
        self.vectorized_songs = self._processing() 
        
    # -------------------------------------------------------
    # Upload data
    # -------------------------------------------------------
    def _load_irish_data(self):
        """Upload the irish dataset in ./data/irish/irish.abc path

        Returns:
            (function) _extract_song: list of strings
        """
        file_path = os.path.join(self.path, "..", "data", "irish", "irish.abc")
        with open(file_path) as f:
            text = f.read()
        return self._extract_song_irish(text)
    
    def _load_pop_data(self):
        file_path = os.path.join(self.path, "..", "data", "pop", "pop.txt")
        with open(file_path) as f:
            text = f.read()
        return self._extract_song_pop(text)

    def _extract_song_irish(self, dataset :str):
        """Devide the dataset .abc in songs for the training.
        Since there is only one .abc file that represents the dataset,
        we need to divide in songs using the pattern delimitator.

        Args:
            (dataset: str): dataset file in .abc format

        Returns:
            (list of strings): one string for each song
        """
        pattern = "(^|\n\n)(.*?)\n\n"
        search_results = regex.findall(pattern, dataset, overlapped=True, flags=regex.DOTALL)
        songs = [song[1] for song in search_results]
        print(f"Found {len(songs)} {self.dataset_name} songs")
        return songs
    
    def _extract_song_pop(self, dataset: str):
        """Devide the dataset .abc in songs for the training.
        Since there is only one .abc file that represents the dataset,
        we need to divide in songs using the pattern delimitator.

        Args:
            (dataset: str): dataset file in .abc format

        Returns:
            (list of strings): one string for each song
        """
        pattern = r'(?=^X:\s*\d+)'
        songs = regex.split(pattern, dataset, flags=regex.MULTILINE)
        songs = [s.strip() for s in songs if s.strip()]  # remove empty entries
        print(f"Found {len(songs)} {self.dataset_name} songs")
        return songs
    
    def _load_mix(self, percentage : float):
        irish_songs = self._load_irish_data()
        lenght_irish_songs = len(irish_songs)
        
        pop_songs = self._load_pop_data()
        lenght_pop_songs = len(pop_songs)
        
        sampled_irish_songs = random.sample(irish_songs, k= min(lenght_irish_songs, lenght_pop_songs)//2)
        sampled_pop_songs = random.sample(pop_songs, k= min(lenght_irish_songs, lenght_pop_songs)//2)
        songs = sampled_irish_songs + sampled_pop_songs
        random.shuffle(songs)        
        return songs
    # -------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------
    def _processing(self):
        """This function creates a dictionary of all possible notes in the songs, mapping char:index.
        Then all the characters are concatened togheter in the one single array.
        

        Returns:
            (np.array): numeric array with codified characters 
        """
        self.vocab = sorted({c for song in self.songs for c in song})
        self.char2idx = {u: i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)
        dtype = np.uint8 if len(self.vocab) <= 256 else np.int32
        encoded_songs = [np.fromiter((self.char2idx[c] for c in song), dtype=dtype)
                         for song in self.songs]
        vectorized = np.concatenate(encoded_songs)
        print(f"Vocab size: {len(self.vocab)} | Total tokens: {len(vectorized)}")
        return vectorized

    # -------------------------------------------------------
    # Dataset interface per DataLoader
    # -------------------------------------------------------
    def __len__(self):
        """returns number of available songs.
        The bottom formula describes the behaviour of the LSTM. We need to slide with stride of 1,
        which is the standard approach for character-by-character generation.

        Returns:
            (int): number of sequence of lenght self.seq_len in the dataset
        """
        return len(self.vectorized_songs) - self.seq_len - 1

    def __getitem__(self, idx):
        """Every time the dataloader calls dataset[idx], it receives a sliding window of self.seq_len size.
         

        Args:
            idx (int): starting point of the concatenated array.

        Returns:
            train_data, test_data (torch.tensor): returns the arrays for training and their next characters
        """
        start = idx
        end = start + self.seq_len
        x = self.vectorized_songs[start:end]
        y = self.vectorized_songs[start+1:end+1]
        return torch.tensor(x, dtype=torch.long, device= self.device), torch.tensor(y, dtype=torch.long, device =self.device)


if __name__ == "__main__":
    dataset = Datasets('mix')
    print(f"Number of songs: {dataset.__len__()}")
    print("Sample:\n", dataset[0][:10])  
    print("Total number of tokens:", len(dataset.vectorized_songs))
    print("Lenght of a sequence:", dataset.seq_len)
    print(f"Total number of different vocab", len(dataset.vocab))