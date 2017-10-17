from mfcc import MFCC
from mfcc_sanderson import MFCC_Sanderson
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
        elif extractor != '':
            self.extractor = MFCC()
        else:
            raise Exception('Define a extractor')

    def mfcc(self, signal, rate):
        return self.extractor.execute(signal, rate)

    def mfcc(self, path):
        return self.extractor.execute_by_path(path)

    # def summary(self, arr):
    #     mean = numpy.mean(arr)
    #     median = numpy.median(arr)
    #     std = numpy.std(arr)
    #     var = numpy.var(arr)
    #
    #     return [mean]
    #
    # def mfcc_feature(self, path):
    #     mfcc, bank = self.mfcc_by_path(path)
    #
    #     mfcc = mfcc[1:400, :]
    #
    #     summaries = map(self.summary, mfcc)
    #     summaries = list(summaries)
    #     flatten = [y for x in summaries for y in x]
    #
    #     return flatten

    def plot(self, arr):

        if self.DEBUG:
            plt.plot(arr)
            plt.show()

    def feature_lib(self, path):
        import scipy.io.wavfile as wav

        (rate, sig) = wav.read(path)
        mfcc_feat = mfcc(sig, rate, nfft=1200)
        #d_mfcc_feat = delta(mfcc_feat, 2)
        #fbank_feat = logfbank(sig, rate)

        mfcc_feat = mfcc_feat[1:300, :]

        summaries = map(self.summary, mfcc_feat)
        flatten = [y for x in summaries for y in x]

        #print(mfcc_feat[1:3, :])

        return flatten

