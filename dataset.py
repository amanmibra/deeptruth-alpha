import os

import torch
from torch.utils.data import Dataset
import torchaudio

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_TIME_LIMIT = 5
DEFAULT_DEVICE="cpu"
DEFAULT_TRANSFORMATION = torchaudio.transforms.MelSpectrogram(
    sample_rate=DEFAULT_SAMPLE_RATE,
    n_fft=2048,
    hop_length=512,
    n_mels=128
)

class VoiceDataset(Dataset):

    def __init__(
            self,
            pure_data_directories,
            clone_data_directories,
            transformation=DEFAULT_TRANSFORMATION,
            device=DEFAULT_DEVICE,
            target_sample_rate=DEFAULT_SAMPLE_RATE,
            time_limit_in_secs=DEFAULT_TIME_LIMIT,
        ):
        # file processing
        self._pure_paths = [os.path.join(dir) for dir in pure_data_directories]
        self._clone_paths = [os.path.join(dir) for dir in clone_data_directories]
        self.labels = ["pure", "clone"]
        self.label_mapping = {label: i for i, label in enumerate(self.labels)}
        self.audio_files_labels = self._join_audio_files()

        self.device = device

        # audio processing
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = time_limit_in_secs * self.target_sample_rate

        # preprocess all wavs
        self.wavs = self._process_wavs()

    def __len__(self):
        return len(self.audio_files_labels)

    def __getitem__(self, index):
        return self.wavs[index]

    def _process_wavs(self):
        wavs = []
        for filepath, label in self.audio_files_labels:
            # load wav
            wav, sr = torchaudio.load(filepath)

            # modify wav file, if necessary
            wav = wav.to(self.device)
            wav = self._resample(wav, sr)
            wav = self._mix_down(wav)
            wav = self._cut_or_pad(wav)
            
            # apply transformation
            wav = self.transformation(wav)

            wavs.append((wav, self.label_mapping[label]))
        
        return wavs


    def _join_audio_files(self):
        """Join all the audio file names and labels into one single dimenional array"""
        audio_files_labels = []
        files = {"pure": self._pure_paths, "clone": self._clone_paths}

        for label in ["pure", "clone"]:
            for path in files[label]:
                for f in os.listdir(path):
                    audio_filepath = os.path.join(path, f)
                    audio_files_labels.append((audio_filepath, label))

        return audio_files_labels

    def _resample(self, wav, current_sample_rate):
        """Resample audio to the target sample rate, if necessary"""
        if current_sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(current_sample_rate, self.target_sample_rate)
            wav = resampler(wav)
        
        return wav

    def _mix_down(self, wav):
        """Mix down audio to a single channel, if necessary"""
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        return wav

    def _cut_or_pad(self, wav):
        """Modify audio if number of samples != target number of samples of the dataset.

        If there are too many samples, cut the audio.
        If there are not enough samples, pad the audio with zeros.
        """

        length_signal =  wav.shape[1]
        if length_signal > self.num_samples:
            wav = wav[:, :self.num_samples]
        elif length_signal < self.num_samples:
            num_of_missing_samples = self.num_samples - length_signal
            pad = (0, num_of_missing_samples)
            wav = torch.nn.functional.pad(wav, pad)

        return wav
