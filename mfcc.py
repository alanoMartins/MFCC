import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class MFCC_Sanderson:

    def __init__(self):
        self.num_filters = 40
        self.nfft = 2048
        self.cep = 12
        self.debug = False

    def execute(self, signal, rate):
        # signal = self.apply_pre_emphasis(signal)
        self.rate = rate
        frames = self.framing(signal, rate)
        filters = self.filter_banks(rate)
        pow_frames = self.fourier_transform(frames)
        signal_filtered = self.run_filters(pow_frames, filters)
        mfcc = self.mfcc(signal_filtered)

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
        Y = np.fft.rfft(frames, self.nfft)
        Z = abs(Y)
        NUP = int(np.ceil((len(Y) + 1) / 2))
        Z = Z[0:NUP]

        pow_frames = ((1.0 / self.nfft) * ((Z) ** 2))

        if self.debug:
            self.plotFFT(frames[0], self.rate)

        return pow_frames


    def plotFFT(self, frame, rate):
        Ws = 0.020
        Ns = Ws * rate
        lin = np.linspace(0, 1, Ns)
        plt.plot(lin, frame)
        plt.show()
        Y = np.fft.fft(frame, self.nfft) / len(frame)
        Z = abs(Y)
        NUP = int(np.ceil((len(Y) + 1) / 2))
        Z = Z[0:NUP]
        f = rate * np.linspace(-0.5, 0.5, NUP)
        plt.plot(f, Z)
        plt.show()

    def plotFilter(self, filters, rate):
        f = np.linspace(0, rate / 2, np.ceil((self.nfft + 1) / 2))
        for idx in range(0, len(filters)):
            plt.figure(idx)
            plt.title('Filter %d (partial)' % idx)
            plt.plot(f[0:50], filters[idx][0:50])
        plt.show()

    def normilize(self, arr):
        arr = [(1 / len(arr) * v ** 2) for v in arr]
        return arr

    def run_filters(self, power, fbank):
        filters = fbank
        filter_banks = np.dot(power, filters.T)
        # Estabilizar
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        return np.log10(filter_banks)


    def filter_banks(self, rate):

        sanderson_filters = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1149, 1320, 1516, 1741, 2000, 2297, 2639, 3031, 3482, 4000])
        central_freqs = sanderson_filters[1:-1] # Remove o inicial e final, mantendo apenas os centrais
        nfilt = len(central_freqs)
        bin = np.floor((self.nfft + 1) * sanderson_filters / rate)

        fbank = np.zeros((nfilt, int(np.floor(self.nfft / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # inicial
            f_m = int(bin[m])  # central
            f_m_plus = int(bin[m + 1])  # final

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        if self.debug:
            self.plotFilter(fbank, rate) #Plota todos os filtros
        return fbank



    def mfcc(self, filter_banks):
        cep_lifter = 22

        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (self.cep + 1)]  # Keep 2-13
        (nframes, ncoeff) = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift
        plt.show()
        return mfcc