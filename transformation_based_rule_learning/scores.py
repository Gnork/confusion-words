__author__ = 'Christoph Jansen, HTW Berlin'

from data import CorpusOptions

def accuracy(flags):
    c = correct(flags)
    w = wrong(flags)
    if c == 0:
        return 0
    return c / (c + w)

def correct(flags):
    tn, fp, tp, fn = flags
    return tp + tn

def wrong(flags):
    tn, fp, tp, fn = flags
    return fp + fn

def precision(flags):
    tn, fp, tp, fn = flags
    if tp == 0:
        return 0
    return tp / (tp + fp)

def recall(flags):
    tn, fp, tp, fn = flags
    if tp == 0:
        return 0
    return tp / (tp + fn)

def f_score(flags, beta=1):
    p = precision(flags)
    r = recall(flags)
    b = beta
    if p == 0 or r == 0:
        return 0
    return (1 + b*b) * (p*r / (b*b*p + r))

def eval(intend_token, predict_token):
    if predict_token == intend_token:
        return (1, 0)
    return (0, 1)

def w_flags(flags, artificial_error):
    e = artificial_error
    tn, fp, tp, fn = flags
    w_tn = tn * (1.0-e)
    w_fp = fp * (1.0-e)
    w_tp = tp * e
    w_fn = fn * e
    return (w_tn, w_fp, w_tp, w_fn)
    
def w_error_corpus(flags, do_weighting):
    tn, fp, tp, fn = flags
    if do_weighting:
        if tp + fn == 0 or tn + fp == 0:
            error_corpus_weight = 1
        else:
            error_corpus_weight = (tn + fp) / (tp + fn)
        tp = tp * error_corpus_weight
        fn = fn * error_corpus_weight
    return (tn, fp, tp, fn)

class AccuracyScore:
    def __init__(self, do_weighting=False):
        self.w = do_weighting
        pass

    def score(self, flags):
        return accuracy(w_error_corpus(flags, self.w))

    def __str__(self):
        return 'accuracy'

class WAccuracyScore:
    def __init__(self, artificial_error, do_weighting=False):
        self.e = artificial_error
        self.w = do_weighting

    def score(self, flags):
        return accuracy(w_flags(w_error_corpus(flags, self.w), self.e))

    def __str__(self):
        return 'w_accuracy'

class WPrecisionScore:
    def __init__(self, artificial_error, do_weighting=False):
        self.e = artificial_error
        self.w = do_weighting

    def score(self, flags):
        return precision(w_flags(w_error_corpus(flags, self.w), self.e))

    def __str__(self):
        return 'w_precision'

class F1Score:
    def __init__(self, artificial_error, do_weighting=False):
        self.e = artificial_error
        self.w = do_weighting

    def score(self, flags):
        return f_score(w_flags(w_error_corpus(flags, self.w), self.e))

    def __str__(self):
        return 'f_1_score'

class F05Score:
    def __init__(self, artificial_error, do_weighting=False):
        self.e = artificial_error
        self.w = do_weighting
        self.b = 0.5

    def score(self, flags):
        return f_score(w_flags(w_error_corpus(flags, self.w), self.e), beta=self.b)

    def __str__(self):
        return 'f_05_score'

class F025Score:
    def __init__(self, artificial_error, do_weighting=False):
        self.e = artificial_error
        self.w = do_weighting
        self.b = 0.25

    def score(self, flags):
        return f_score(w_flags(w_error_corpus(flags, self.w), self.e), beta=self.b)

    def __str__(self):
        return 'f_025_score'

def sort_rules_by_train_results(all_train_results, rules, score, last_results):
    l = score.score(last_results.get_data(CorpusOptions.TRAIN))
    calculated_scores = []
    for i, ar in enumerate(all_train_results):
        s = score.score(ar.get_data(CorpusOptions.TRAIN))
        t = (s, i)
        calculated_scores.append(t)
    calculated_scores.sort(reverse=True)

    sorted_rules = []
    for s, i in calculated_scores:
        if s <= l:
            break
        sorted_rules.append(rules[i])
    return sorted_rules

def select_rule_by_cv_results(all_cv_results, score, last_results):
    l = score.score(last_results.get_data(CorpusOptions.CV))
    selected_index = -1
    for i, ar in enumerate(all_cv_results):
        s = score.score(ar.get_data(CorpusOptions.CV))
        if s > l:
            selected_index = i
            break

    return selected_index