import os
import numpy as np
import torch
import scipy
import random
from torch.utils.data import Dataset


class AudioFeatureDataset(Dataset):
    """
    A custom Dataset for loading featrues.

    Args:
        audio_folder (str): Path to the directory containing feature files.
        length (int): The length of feature vectors (optional, for padding/trimming).
        is_test (bool): Whether the dataset is for testing.
    """

    def __init__(self, audio_folder: str, length: int = None, is_test: bool = False):
        self.audio_folder = audio_folder
        self.length = length
        self.is_test = is_test
        
        # List all audio files
        self.audio_files = sorted([x for x in os.listdir(audio_folder) if x.endswith(".npy")])

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load feature files
        audio_path = os.path.join(self.audio_folder, self.audio_files[idx])
        
        label = int(self.audio_files[idx].split('-')[-1][0])
            
        spectrogram=np.load(audio_path)
        spectrogram = np.array(spectrogram)
        
        # Padding
        if spectrogram.shape[0]<61:
            pad = 61-spectrogram.shape[0]
            spectrogram = np.pad(spectrogram, ((0,pad), (0,0)), mode='constant', constant_values=0)
        
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        
        # No need to prepare the label for the test set
        if self.is_test:
            return spectrogram

        # Extract the label from the file name
        label = torch.tensor(np.array(label), dtype=torch.int)
        
        
        # Get sample name (optional, split on the expected format)
        sample_name = self.audio_files[idx].split('-')[0]
        
        return spectrogram, label, sample_name
