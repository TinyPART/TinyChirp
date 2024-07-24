import torchaudio
from torchaudio import transforms as T
from torch.utils.data import Dataset
import os
import torch.nn.functional as F
import config as cfg



def load_and_preprocess_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    #print(waveform.shape)
    transformer = T.MelSpectrogram(sample_rate=sr, n_mels=64)
    mel_spec = transformer(waveform)
    mel_spec_db = T.AmplitudeToDB()(mel_spec)
    mel_spec_db = mel_spec_db.permute(0,2,1)
    return mel_spec_db, waveform, sr

def load_and_preprocess_raw_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    return waveform, sr


class AudioDataset(Dataset):
    # AudioDataset class for MelSpectrogram dataset
    def __init__(self, target_folder, non_target_folder, transform = None, fixed_length_spect = None, train = True):
        self.target_folder = target_folder
        self.non_target_folder = non_target_folder
        self.target_files = os.listdir(target_folder)
        self.non_target_files = os.listdir(non_target_folder)
        self.transform = transform
        self.fixed_length_spect = fixed_length_spect
        self.train = train
        #self.fixed_length_wave = fixed_length_wave
    
    def __len__(self):
        return len(self.target_files) + len(self.non_target_files)
    
    def __getitem__(self, idx):
        if idx < len(self.target_files):
            filename = self.target_files[idx]
            label = 0
            filepath = os.path.join(self.target_folder,filename)
        else:
            filename = self.non_target_files[idx-len(self.target_files)]
            label = 1
            filepath = os.path.join(self.non_target_folder, filename)
        mel_spec,_, _ = load_and_preprocess_audio(filepath)
        """if self.train: 
            chunks = analyze.getRawAudioFromFile(filepath, 0, cfg.FILE_SPLITTING_DURATION)"""
        # was used for birdnet compatibility
        if self.fixed_length_spect is not None:
            if mel_spec.shape[1] < self.fixed_length_spect:
                mel_spec = F.pad(mel_spec, (0,0,self.fixed_length_spect - mel_spec.shape[1],0), 'constant', 0)
            elif mel_spec.shape[1] > self.fixed_length_spect:
                mel_spec = mel_spec[:, : self.fixed_length_spect, : ]
        
            
        """if self.train : 
            return chunks, mel_spec, label
        else:
            return mel_spec, label"""
        return mel_spec, label


class RawAudioDataset(Dataset):
    # Class for audio dataset with waveforms
    def __init__(self, target_folder, non_target_folder, fixed_length_wave = None, transform = None):
        self.target_folder = target_folder
        self.non_target_folder = non_target_folder
        self.target_files = os.listdir(target_folder)
        self.non_target_files = os.listdir(non_target_folder)
        self.transform = transform
        self.fixed_length_wave = fixed_length_wave
    
    def __len__(self):
        return len(self.target_files) + len(self.non_target_files)
    
    def __getitem__(self, idx):
        if idx < len(self.target_files):
            filename = self.target_files[idx]
            label = 0
            filepath = os.path.join(self.target_folder,filename)
        else:
            filename = self.non_target_files[idx-len(self.target_files)]
            label = 1
            filepath = os.path.join(self.non_target_folder, filename)
        waveform, sample_rate = load_and_preprocess_raw_audio(filepath)
        if self.transform:
            waveform = self.transform(waveform)
        if self.fixed_length_wave:
            current_length = waveform.shape[1]
            if current_length > self.fixed_length_wave:
                waveform = waveform[:,:self.fixed_length_wave]
            else:
                waveform = F.pad(waveform, (0, self.fixed_length_wave - current_length), mode='constant', value=0)
        return waveform, label

