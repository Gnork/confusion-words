__author__ = 'Christoph Jansen, HTW Berlin'

import os
import normalization
import training
from datetime import datetime
import scores
import rule_templates
import logger
import shutil

home_dir = os.path.expanduser('~')

### BEGIN SETTINGS ###

# confusion_set must consist of token_a and token_b
# order matters: algorithm will generate rules for occurrences of token_a
# token_b will be used to for artificial error corpus generation
token_a = 'than'
token_b = 'then'

# absolute path to corpus file 
# corpus must contain one sentence per line 
# line must consist of token|POS pairs
# POS tag-set does not matter
# example line:
# He|PPS did|DOD not|* say|VB by|IN how|QL much|AP .|.
corpus_file = os.path.join(home_dir, 'brown_tagged.txt')

# many temporary and result files will be generated in work directory
work_dir = os.path.join(home_dir, 'training_tbrl')

# error rate is weighting for results of artificial error corpus -> True Positive, False Negative
# correct corpus results will be weighted with (1 - artificial_error) -> True Negative, False Positive
# can be set to 0.5 for no weighting
artificial_error = 0.01

# define baseline prediction 
# identity baseline prediction is placeholder for doing nothing (recommended setting)
# alternative is simple baseline prediction in rule_templates
bp = rule_templates.IdentityBaselinePrediction()

# define evaluation criteria (F05Score is recommended)
# alternatives can be found in scores
# do_weighting enables weighting for skewed data (token_a and token_b not equally distributed in corpus)
# do_weighting is necessary for applying correct artificial_error (can be set to False if real error corpus is used and artificial_error is 0.5)
do_weighting = True
score = scores.F05Score(artificial_error, do_weighting=do_weighting)

### END SETTINGS ###

# init
test=3
cv=4
confusion_set = (token_a, token_b)
confusion_set_string = '%s_%s' % confusion_set
set_dir = os.path.join(work_dir, confusion_set_string)
paths = {'normalized': os.path.join(work_dir, 'normalized.txt'),
    'filtered_tsv': os.path.join(set_dir, 'filtered.tsv'),
    'train_intend_tsv': os.path.join(set_dir, 'train_intend.tsv'),
    'cv_intend_tsv': os.path.join(set_dir, 'cv_intend.tsv'),
    'test_intend_tsv': os.path.join(set_dir, 'test_intend.tsv'),
    'train_error_tsv': os.path.join(set_dir, 'train_error.tsv'),
    'cv_error_tsv': os.path.join(set_dir, 'cv_error.tsv'),
    'test_error_tsv': os.path.join(set_dir, 'test_error.tsv'),
    'tmp': os.path.join(set_dir, 'tmp')}
    
if not os.path.exists(paths['tmp']):
    os.makedirs(paths['tmp'])

# prepare corpus
normalization.normalize_acrotagged_corpus(corpus_file, paths['normalized'])
normalization.filter_corpus_to_tsv(paths['normalized'], paths['filtered_tsv'], confusion_set)
normalization.dynamically_split_tsv(paths, confusion_set, test=test, cv=cv)

# if training with seperate error corpus is performed
use_seperate_error_corpus = True
normalization.directed_error_tsv(paths, confusion_set, cv=cv)

experiment = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
experiment_dir = os.path.join(set_dir, experiment)

exp_settings = {'baseline_prediction': bp.__str__(),
                'confusion_set': confusion_set,
                'score_method': score.__str__(),
                'artificial_error': artificial_error,
                'cv': cv,
                'test': test,
                'corpus': corpus_file}

paths['train_intend_predict'] = os.path.join(experiment_dir, 'train_intend_predict.txt')
paths['train_intend_predict_tmp'] = os.path.join(experiment_dir, 'train_intend_predict.tmp')
paths['train_error_predict'] = os.path.join(experiment_dir, 'train_error_predict.txt')
paths['train_error_predict_tmp'] = os.path.join(experiment_dir, 'train_error_predict.tmp')

paths['cv_intend_predict'] = os.path.join(experiment_dir, 'cv_intend_predict.txt')
paths['cv_intend_predict_tmp'] = os.path.join(experiment_dir, 'cv_intend_predict.tmp')
paths['cv_error_predict'] = os.path.join(experiment_dir, 'cv_error_predict.txt')
paths['cv_error_predict_tmp'] = os.path.join(experiment_dir, 'cv_error_predict.tmp')

paths['test_intend_predict'] = os.path.join(experiment_dir, 'test_intend_predict.txt')
paths['test_intend_predict_tmp'] = os.path.join(experiment_dir, 'test_intend_predict.tmp')
paths['test_error_predict'] = os.path.join(experiment_dir, 'test_error_predict.txt')
paths['test_error_predict_tmp'] = os.path.join(experiment_dir, 'test_error_predict.tmp')

log_path = os.path.join(experiment_dir, 'results')

paths['training_log'] = os.path.join(log_path, 'training.csv')
paths['exp_log'] = os.path.join(log_path, 'experiment.json')
paths['rules'] = os.path.join(log_path, 'rules.pickle')

if not os.path.exists(log_path):
    os.makedirs(log_path)

logger.log_exp_settings(paths['exp_log'], exp_settings)

# begin training
training.perform(paths, confusion_set, score, [bp], artificial_error, cv, use_seperate_error_corpus=use_seperate_error_corpus, do_weighting=do_weighting)

try:
    os.remove(paths['train_intend_predict'])
    os.remove(paths['train_intend_predict_tmp'])
except:
    pass
try:
    os.remove(paths['train_error_predict'])
    os.remove(paths['train_error_predict_tmp'])
except:
    pass
try:
    os.remove(paths['cv_intend_predict'])
    os.remove(paths['cv_intend_predict_tmp'])
except:
    pass
try:
    os.remove(paths['cv_error_predict'])
    os.remove(paths['cv_error_predict_tmp'])
except:
    pass
try:
    os.remove(paths['test_intend_predict'])
    os.remove(paths['test_intend_predict_tmp'])
except:
    pass
try:
    os.remove(paths['test_error_predict'])
    os.remove(paths['test_error_predict_tmp'])
except:
    pass