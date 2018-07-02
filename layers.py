import torch
import torch.nn.functional as F
import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
from .audio_processing import dynamic_range_compression
from .audio_processing import dynamic_range_decompression
from .audio_processing import amplitude_to_db_torch, db_to_amplitude_torch
from .stft import STFT


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class TacotronSTFT(torch.nn.Module):
    """
    ->  adapted from https://github.com/pseeth/pytorch-stft/blob/master/mel_spectrogram.py
        Example:
            audio, sr = librosa.load("mixture.mp3", sr=None)
            audio = Variable(torch.FloatTensor(audio), requires_grad=False).unsqueeze(0)
            mel_transform = MelSpectrogram(sample_rate=sr, filter_length=1024, num_mels=150)
            mel_spectrogram = mel_transform(audio).squeeze(0).data.numpy()
        """
    def __init__(self, sample_rate=22050, filter_length=1024,
                 hop_length=256, win_length=None, win_func="hann",
                 num_mels=80, ref_dB=-30, max_dB=120, preemph=0.95):
        super().__init__()
        if win_length is None:
            win_length = filter_length
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.sample_rate = sample_rate  
        
        self.stft = STFT(filter_length=self.filter_length, hop_length=self.hop_length,
                         win_length=win_length, window=win_func)
        mel_filters = librosa.filters.mel(self.sample_rate, self.filter_length, self.num_mels)
        self.mel_filter_bank = torch.nn.Parameter(torch.FloatTensor(mel_filters), requires_grad=False)

        self.ref_dB = ref_dB
        self.max_dB = max_dB
        self.preemph = preemph

    def _normalize(self, s):
        return torch.clamp((s + self.ref_dB + self.max_dB) / self.max_dB, min=0.0, max=1.0)

    def spectral_normalize(self, magnitudes):
        output = amplitude_to_db_torch(magnitudes)
        return output

    def spectral_de_normalize(self, db):
        output = db_to_amplitude_torch(db)
        return output

    def mel_spectrogram(self, input_data):
        if self.preemph > 0:
            input_data = torch.cat((input_data[:, :1], input_data[:, 1:] - self.preemph * input_data[:, :-1]), dim=1)
        magnitude, phase = self.stft.transform(input_data)
        mel_spectrogram = F.linear(magnitude.transpose(-1, -2), self.mel_filter_bank)
        mel_spectrogram = self.spectral_normalize(mel_spectrogram)
        return self._normalize(mel_spectrogram).transpose(-1, -2)

    def named_parameters(self, memo=None, prefix=''):
        # no parameters!
        pass
