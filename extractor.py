from mfcc import MFCC
import matplotlib.pyplot as plt
import numpy
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank


class Extractor:

    DEBUG = False

    def __init__(self, extractor):
        if extractor != '':
            self.extractor = MFCC()
        else:
            raise Exception('Define a extractor')

    def mfcc(self, signal, rate):
        return self.extractor.execute(signal, rate)

    def mfcc_by_path(self, path):
        return self.extractor.execute_by_path(path)

    def summary(self, arr):
        mean = numpy.mean(arr)
        median = numpy.median(arr)
        std = numpy.std(arr)
        var = numpy.var(arr)

        l = len(arr)

        return [mean, median, std, var]

    def mfcc_feature(self, path):
        mfcc, bank = self.mfcc_by_path(path)

        summaries = map(self.summary, mfcc)
        summaries = list(summaries)
        flatten = [y for x in summaries for y in x]

        l = len(flatten)

        return flatten

    def plot(self, arr):

        if self.DEBUG:
            plt.plot(arr)
            plt.show()

    def feature_lib(self, path):
        import scipy.io.wavfile as wav

        (rate, sig) = wav.read(path)
        mfcc_feat = mfcc(sig, rate)
        d_mfcc_feat = delta(mfcc_feat, 2)
        fbank_feat = logfbank(sig, rate)

        summaries = map(self.summary, fbank_feat)
        flatten = [y for x in summaries for y in x]

        print(fbank_feat[1:3, :])

        return flatten

