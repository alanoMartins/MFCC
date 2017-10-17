import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct


class MFCC_Sanderson:

    def __init__(self):
        self.num_filters = 40
        self.nfft = 2048

    def execute(self, signal, rate):
        # signal = self.apply_pre_emphasis(signal)
        frames = self.framing(signal, rate)
        f = self.fourier_transform(frames)
        filters = self.filters(f, self.banks(rate))
        mfcc = self.mfcc(filters)
        return mfcc

    def execute_by_path(self, path):
        rate, signal = scipy.io.wavfile.read(path)
        return self.execute(signal, rate)

    def apply_pre_emphasis(self, signal):
        pre_emphasis = 0.97
        return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    def framing(self, signal, rate):
        frame_size = (20 / 1000) * rate
        frame_advance = (10 / 1000) * rate

        frame_size = int(round(frame_size))
        frame_advance = int(round(frame_advance))

        signal_len = len(signal)

        # Revisar com Thelmo
        # exceed = np.remainder(signal_len - frame_size, frame_advance)
        # z = np.zeros(int(exceed))
        # pad_signal = np.append(signal, z)

        num_frames = int(np.ceil(
            float(np.abs(signal_len - frame_size)) / frame_advance))  # Make sure that we have at least 1 frame

        # Adicionando zeros no fim do sinal
        pad_signal_length = num_frames * frame_advance + frame_size
        z = np.zeros((int(pad_signal_length) - signal_len))
        pad_signal = np.append(signal, z)

        #Aplicando a janela de Hamming
        indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + np.tile(
            np.arange(0, num_frames * frame_advance, frame_advance), (frame_size, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames *= np.hamming(frame_size)
        return frames

    def fourier_transform(self, frames):
        mag_frames = np.absolute(np.fft.fft(frames, self.nfft))
        # Deve normalizar
        return self.normilize(mag_frames)

    def normilize(self, arr):
        arr -= (np.mean(arr, axis=0) + 1e-8)
        return arr

    def filters(self, power, fbank):
        filters = fbank
        filter_banks = np.dot(power, filters.T)
        # Estabilizar
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        return np.log10(filter_banks)

    def mfcc(self, filters):
        # num_ceps = 12
        return dct(filters, type=2, axis=1, norm='ortho') # [:, 1: (num_ceps + 1)]

    def banks(self, rate):
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.num_filters + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((self.nfft + 1) * hz_points / rate)

        fbank = np.zeros((self.num_filters, int(np.floor(self.nfft / 2 + 1))))
        for m in range(1, self.num_filters + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        return fbank
