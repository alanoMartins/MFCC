from mfcc import MFCC_Sanderson
import matplotlib.pyplot as plt
import numpy
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank


class Extractor:

    DEBUG = False

    def __init__(self, extractor):
        if extractor == 'sanderson':
            self.extractor = MFCC_Sanderson()
        elif extractor == 'lib':
            self.extractor = self.feature_lib()
        else:
            raise Exception('Define a extractor')

    def mfcc(self, signal, rate):
        return self.extractor.execute(signal, rate)

    def mfcc(self, path):
        return self.extractor.execute_by_path(path)

    def plot(self, arr):

        if self.DEBUG:
            plt.plot(arr)
            plt.show()

    def feature_lib(self, path):
        import scipy.io.wavfile as wav

        (rate, sig) = wav.read(path)
        mfcc_feat = mfcc(sig, rate, nfft=2048, nfilt=17, ceplifter=12)
        #d_mfcc_feat = delta(mfcc_feat, 2)
        #fbank_feat = logfbank(sig, rate)





        return mfcc_feat

