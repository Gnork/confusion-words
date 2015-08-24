__author__ = 'Christoph Jansen, HTW Berlin'

from enum import Enum

CorpusOptions = Enum('CorpusOptions', 'TRAIN CV CV2 TEST')

class RuleResults:
    def __init__(self):
        self.train = None
        self.train_error = None
        self.cv = None
        self.cv_error = None
        self.cv2 = None
        self.cv2_error = None
        self.test = None
        self.test_error = None
        self.common = None

    def set_common(self, data):
        self.common = data

    def set_data(self, data: (int, int), corpus: CorpusOptions, error: bool):
        if corpus == CorpusOptions.TRAIN:
            if not error:
                self.train = data
            else:
                self.train_error = data
        elif corpus == CorpusOptions.CV:
            if not error:
                self.cv = data
            else:
                self.cv_error = data
        elif corpus == CorpusOptions.CV2:
            if not error:
                self.cv2 = data
            else:
                self.cv2_error = data
        elif corpus == CorpusOptions.TEST:
            if not error:
                self.test = data
            else:
                self.test_error = data

    def get_data(self, corpus: CorpusOptions):
        if corpus == CorpusOptions.TRAIN:
            if not self.train_error:
                return self.train + self.train
            return self.train + self.train_error
        elif corpus == CorpusOptions.CV:
            if not self.cv:
                return (0,0,0,0)
            elif not self.cv_error:
                return self.cv + self.cv
            return self.cv + self.cv_error
        elif corpus == CorpusOptions.CV2:
            if not self.cv2:
                return (0,0,0,0)
            elif not self.cv2_error:
                return self.cv2 + self.cv2
            return self.cv2 + self.cv2_error
        elif corpus == CorpusOptions.TEST:
            if not self.test_error:
                return self.test + self.test
            return self.test + self.test_error