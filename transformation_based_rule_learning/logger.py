__author__ = 'Christoph Jansen, HTW Berlin'

import os
import scores
from data import CorpusOptions
import json

def log_exp_settings(log_path, exp_settings):
    open_log = open(log_path, 'w')
    data = json.dumps(exp_settings, sort_keys=True)
    print(data, file=open_log)
    open_log.close()
    print(data)

class TrainingLogger:
    def __init__(self, log_path):
        self.open_log = open(log_path, 'w')
        self.common_fields = ['index',
                              'rule',
                              'origin',
                              'replace',
                              'skips']

        self.individual_fields = ['tn',
                                  'fp',
                                  'tp',
                                  'fn',
                                  'accuracy',
                                  'precision',
                                  'recall',
                                  'w_accuracy',
                                  'w_precision',
                                  'w_recall',
                                  'f_1_score',
                                  'f_05_score',
                                  'f_025_score']

        print(self._headline(), file=self.open_log)

    def _headline(self):
        fields = self._headline_fields()
        line = ''
        for field in fields:
            line += '%s;' % field
        return line.rstrip(';')

    def _headline_fields(self):
        train_fields = ['train_%s' % f for f in self.individual_fields]
        cv_fields = ['cv_%s' % f for f in self.individual_fields]
        cv2_fields = ['cv2_%s' % f for f in self.individual_fields]
        test_fields = ['test_%s' % f for f in self.individual_fields]
        return self.common_fields + train_fields + test_fields + cv_fields + cv2_fields

    def _dataline(self, fields, d):
        line = ''
        for field in fields:
            val = d.get(field, '')
            line += '%s;' % str(val)
        return line.rstrip(';')

    def _common_data(self, common):
        cd = {}
        for f in self.common_fields:
            cd[f] = common.get(f, '')
        return cd

    def _individual_data(self, d, p, e, w):
        data = {'%s_tn' % p: d[0],
                '%s_fp' % p: d[1],
                '%s_tp' % p: d[2],
                '%s_fn' % p: d[3],
                '%s_accuracy' % p: scores.accuracy(scores.w_error_corpus(d, w)),
                '%s_precision' % p: scores.precision(scores.w_error_corpus(d, w)),
                '%s_recall' % p: scores.recall(scores.w_error_corpus(d, w)),
                '%s_w_accuracy' % p: scores.accuracy(scores.w_flags(scores.w_error_corpus(d, w), e)),
                '%s_w_precision' % p: scores.precision(scores.w_flags(scores.w_error_corpus(d, w), e)),
                '%s_w_recall' % p: scores.recall(scores.w_flags(scores.w_error_corpus(d, w), e)),
                '%s_f_1_score' % p: scores.f_score(scores.w_flags(scores.w_error_corpus(d, w), e)),
                '%s_f_05_score' % p: scores.f_score(scores.w_flags(scores.w_error_corpus(d, w), e), beta=0.5),
                '%s_f_025_score' % p: scores.f_score(scores.w_flags(scores.w_error_corpus(d, w), e), beta=0.25)}
        return data


    def out(self, data, artificial_error, do_weighting):
        combined_data = self._common_data(data.common)
        combined_data.update(self._individual_data(data.get_data(CorpusOptions.TRAIN), 'train', artificial_error, do_weighting))
        combined_data.update(self._individual_data(data.get_data(CorpusOptions.CV), 'cv', artificial_error, do_weighting))
        combined_data.update(self._individual_data(data.get_data(CorpusOptions.CV2), 'cv2', artificial_error, do_weighting))
        combined_data.update(self._individual_data(data.get_data(CorpusOptions.TEST), 'test', artificial_error, do_weighting))

        line = self._dataline(self._headline_fields(), combined_data)
        print(line)
        print(line, file = self.open_log)
        self._flush()

    def close(self):
        self.open_log.close()

    def _flush(self):
        self.open_log.flush()
        os.fsync(self.open_log.fileno())

class TestLogger(TrainingLogger):
    def out(self, data, artificial_error, do_weighting):
        combined_data = self._individual_data(data.get_data(CorpusOptions.TEST), 'test', artificial_error, do_weighting)

        line = self._dataline(self._headline_fields(), combined_data)
        print(line)
        print(line, file = self.open_log)
        self._flush()