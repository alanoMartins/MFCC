import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt


class MFCC_Sanderson:

    def __init__(self):
        self.num_filters = 40
        self.nfft = 512

    def execute(self, signal, rate):
        # signal = self.apply_pre_emphasis(signal)
        self.rate = rate
        frames = self.framing(signal, rate)
        fast_fourier = self.stff(signal, rate)
        banks_alt = self.filter_banks(signal, rate)
        mfcc = self.mfcc(banks_alt)
        frames_fourier = self.fourier_transform_sanderson(frames)
        #fourier = self.fourier_transform(frames, rate)
        banks = self.fbanks(frames_fourier[0])


        # filters = self.filters(f, self.banks(rate))
        # mfcc = self.mfcc(filters)
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

    def fourier_transform(self, frames, rate):
        self.plotFFT(frames[0], rate)
        mag_frames = np.absolute(np.fft.fft(frames, self.nfft))

        # Deve normalizar
        return self.normilize(mag_frames)

    def fourier_transform_sanderson(self, frames):
        Y = np.fft.rfft(frames, self.nfft)
        Z = abs(Y)
        NUP = int(np.ceil((len(Y) + 1) / 2))
        Z = Z[0:NUP]
        f = self.rate * np.linspace(-0.5, 0.5, NUP)
        #plt.plot(f, Z)
        #plt.show()

        NFFT = 512

        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # Power Spectrum

        pow_frames_sanderson =  ((1.0 / NFFT) * ((Z) ** 2))


        return pow_frames_sanderson


    def plotFFT(self, frame, rate):
        Ws = 0.020
        Ns = Ws * rate
        lin = np.linspace(0, 1, Ns)
        plt.plot(lin, frame)
        plt.show()
        b = 512
        Y = np.fft.fft(frame, self.nfft) / len(frame)
        Z = abs(Y)
        NUP = int(np.ceil((len(Y) + 1) / 2))
        Z = Z[0:NUP]
        f = rate * np.linspace(-0.5, 0.5, NUP)
        plt.plot(f, Z)
        plt.show()



    def normilize(self, arr):
        arr = [(1 / len(arr) * v ** 2) for v in arr]
        return arr

    def filters(self, power, fbank):
        filters = fbank
        filter_banks = np.dot(power, filters.T)
        # Estabilizar
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        return np.log10(filter_banks)

    def mfcc_sanderson(self, filters):
        # num_ceps = 12
        return dct(filters, type=2, axis=1, norm='ortho') # [:, 1: (num_ceps + 1)]



    def single_filter(self, value, initial, center, final):
        if value < initial or value > final:
            return 0
        elif value < center:
            return (value - initial) / (center - initial)
        else:
            return (final - value) / (final - center)

    def function_filter(self, initial, center, final):
        return lambda x: self.single_filter(x, initial, center, final)

    def fbanks(self, freqs):
        hz_points = lambda x: (2595 * np.log10(1 + (x / 2) / 700))

        centers = [300, 400, 500, 600, 700, 800, 900, 1000, 1149, 1320, 1516, 1741, 2000, 2297, 2639, 3031, 3482]
        initial = hz_points(200)
        final = hz_points(4000)



        centers = list(map(hz_points, centers))

        filters = []
        for i in range(0, len(centers)):
            if i == 0:
                filters.append(self.function_filter(initial, centers[i], centers[i + 1]))
            elif i == len(centers) - 1:
                filters.append(self.function_filter(centers[i- 1], centers[i], final))
            else:
                filters.append(self.function_filter(centers[i - 1], centers[i], centers[i + 1]))

        banks = []
        for f in filters:
            banks.append([f(fr) for fr in freqs])

        return banks

        # Short time fourier transform
    def stff(self, signal, rate):
        NFFT = 512

        frames = self.framing(signal, rate)
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # Power Spectrum

        return mag_frames, pow_frames


    def filter_banks(self, signal, rate):
        NFFT = 512
        nfilt = 40

        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((NFFT + 1) * hz_points / rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        mag_frames, pow_frames = self.stff(signal, rate)
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        return 20 * np.log10(filter_banks)

    def mfcc(self, filter_banks):
        num_ceps = 12
        cep_lifter = 22

        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13
        (nframes, ncoeff) = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift  # *
        return mfcc