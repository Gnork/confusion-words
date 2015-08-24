__author__ = 'Christoph Jansen, HTW Berlin'

from io_wrapper import TSVReader
from io_wrapper import export_rule_set
import scores
import rule_templates
import multiprocessing
from data import CorpusOptions
from data import RuleResults
from multiprocessing import Pool
from logger import TrainingLogger
import shutil

def scores_per_line(inp):
    tsv_data, predict_token, rules = inp
    tsv_data.tokens[tsv_data.index] = predict_token
    scorings = []

    for rule in rules:
        new_predict_token = rule.apply(tsv_data.tokens, tsv_data.pos, tsv_data.index)
        scorings.append(tsv_data.label == new_predict_token)

    return scorings

def parallel_scoring(corpus, last_predictions, rules, pool, chunk_size):
    counters = []
    for rule in rules:
        counters.append([0, 0])
    data = data_generator(corpus, last_predictions, rules)
    results = pool.imap_unordered(scores_per_line, data, chunksize=chunk_size)
    for result in results:
        for counter, is_correct in zip(counters, result):
            if is_correct:
                counter[0] += 1
            else:
                counter[1] += 1
    return counters

def data_generator(corpus, last_predictions, rules):
    tsv_reader = TSVReader(corpus)
    last_prediction_reader = open(last_predictions, 'r')
    for tsv_data, predict_token in zip(tsv_reader, last_prediction_reader):
        predict_token = predict_token.strip()
        yield (tsv_data, predict_token, rules)
    last_prediction_reader.close()

def _apply_rule_set_to_tokens(author_tokens, author_pos, index, rule_set):
    apply_tokens = list(author_tokens)
    apply_token = apply_tokens[index]
    for rule in rule_set:
        apply_tokens[index] = apply_token
        apply_token = rule.apply(apply_tokens, author_pos, index)

    return apply_token

def _initial_prediction(tsv_path: str, predict_path: str, confusion_set: (str, str), rule_set, confuse_tokens: bool = False):
    reader = TSVReader(tsv_path)
    open_predict = open(predict_path, 'w')
    for d in reader:
        if confuse_tokens:
            t = d.tokens[d.index]
            if t == confusion_set[0]:
                d.tokens[d.index] = confusion_set[1]
            else:
                d.tokens[d.index] = confusion_set[0]

        predict_token = _apply_rule_set_to_tokens(d.tokens, d.pos, d.index, rule_set)
        print(predict_token, file=open_predict)
    open_predict.close()
    
def _incremental_prediction(tsv_path: str, predict_path: str, predict_tmp_path, confusion_set: (str, str), rule):
    reader = TSVReader(tsv_path)
    open_predict = open(predict_path, 'r')
    open_predict_tmp = open(predict_tmp_path, 'w')
    
    for d, predict_token in zip(reader, open_predict):
        d.tokens[d.index] = predict_token.strip()
        apply_token = _apply_rule_set_to_tokens(d.tokens, d.pos, d.index, [rule])
        print(apply_token, file=open_predict_tmp)
    open_predict.close()   
    open_predict_tmp.close()
    shutil.copy(predict_tmp_path, predict_path)

def score_existing_prediction(tsv_path, predict_path):
    tsv_reader = TSVReader(tsv_path)
    predict_open = open(predict_path, 'r')
    r0 = 0
    r1 = 0
    for tsv_data in tsv_reader:
        predict_token = predict_open.readline().strip()
        scoring = scores.eval(tsv_data.label, predict_token)
        r0 += scoring[0]
        r1 += scoring[1]
        
    predict_open.close()
    return (r0, r1)

def _apply_initial_prediction(intend_tsv_path, intend_predict_path, error_tsv_path, error_predict_path, confusion_set, rule_set, corpus_option, use_seperate_error_corpus, use_baseline_prediction, selected_results):
    _initial_prediction(intend_tsv_path, intend_predict_path, confusion_set, rule_set)
    selected_results.set_data(score_existing_prediction(intend_tsv_path, intend_predict_path), corpus_option, False)

    if use_seperate_error_corpus:
        _initial_prediction(error_tsv_path, error_predict_path, confusion_set, rule_set)
        selected_results.set_data(score_existing_prediction(error_tsv_path, error_predict_path), corpus_option, True)

    elif not use_baseline_prediction:
        _initial_prediction(intend_tsv_path, error_predict_path, confusion_set, rule_set, confuse_tokens=True)
        selected_results.set_data(score_existing_prediction(intend_tsv_path, error_predict_path), corpus_option, True)

def _apply_incremental_prediction(intend_tsv_path, intend_predict_path, intend_predict_tmp_path, error_tsv_path, error_predict_path, error_predict_tmp_path, confusion_set, selected_rule, corpus_option, use_seperate_error_corpus, use_baseline_prediction, selected_results):
    _incremental_prediction(intend_tsv_path, intend_predict_path, intend_predict_tmp_path, confusion_set, selected_rule)
    selected_results.set_data(score_existing_prediction(intend_tsv_path, intend_predict_path), corpus_option, False)

    if use_seperate_error_corpus:
        _incremental_prediction(error_tsv_path, error_predict_path, error_predict_tmp_path, confusion_set, selected_rule)
        selected_results.set_data(score_existing_prediction(error_tsv_path, error_predict_path), corpus_option, True)

    elif not use_baseline_prediction:
        _incremental_prediction(intend_tsv_path, error_predict_path, error_predict_tmp_path, confusion_set, selected_rule)
        selected_results.set_data(score_existing_prediction(intend_tsv_path, error_predict_path), corpus_option, True)

def perform(paths, confusion_set, score, rule_set, artificial_error, do_cv, use_seperate_error_corpus=True, do_weighting=True):

    rule_generation_funcs = [rule_templates.generate_cooccurences,
                             rule_templates.generate_collocations]

    # multiprocessing settings
    chunk_size = 10
    cpus = multiprocessing.cpu_count()
    pool = Pool(cpus)

    # misc init
    use_baseline_prediction = False
    if type(rule_set[0]) is rule_templates.SimpleBaselinePrediction:
        use_baseline_prediction = True

    logger = TrainingLogger(paths['training_log'])
    selected_results = RuleResults()

    # apply previous rules to training data
    _apply_initial_prediction(paths['train_intend_tsv'], paths['train_intend_predict'], paths['train_error_tsv'], paths['train_error_predict'], confusion_set, rule_set, CorpusOptions.TRAIN, use_seperate_error_corpus, use_baseline_prediction, selected_results)
    if do_cv:
        _apply_initial_prediction(paths['cv_intend_tsv'], paths['cv_intend_predict'], paths['cv_error_tsv'], paths['cv_error_predict'], confusion_set, rule_set, CorpusOptions.CV, use_seperate_error_corpus, use_baseline_prediction, selected_results)
    _apply_initial_prediction(paths['test_intend_tsv'], paths['test_intend_predict'], paths['test_error_tsv'], paths['test_error_predict'], confusion_set, rule_set, CorpusOptions.TEST, use_seperate_error_corpus, use_baseline_prediction, selected_results)

    selected_rule = rule_set[-1]
        
    common_data = {'index': len(rule_set) - 1,
                   'rule': selected_rule.__str__(),
                   'origin': selected_rule.origin_token,
                   'replace': selected_rule.replace_token,
                   'skips': 0}

    selected_results.set_common(common_data)
    logger.out(selected_results, artificial_error, do_weighting)
    
    while(True):
        # training
        generated_rules = _generate_rules_from_prediction_efficient(paths, rule_generation_funcs, confusion_set, use_baseline_prediction, use_seperate_error_corpus)
        
        print('generated rules: %d' % len(generated_rules))
        if len(generated_rules) == 0:
            break

        scorings = parallel_scoring(paths['train_intend_tsv'], paths['train_intend_predict'], generated_rules, pool, chunk_size)
        all_results = []
        for scoring in scorings:
            rule_results = RuleResults()
            rule_results.set_data(scoring, CorpusOptions.TRAIN, False)
            all_results.append(rule_results)

        if use_seperate_error_corpus:
            scorings = parallel_scoring(paths['train_error_tsv'], paths['train_error_predict'], generated_rules, pool, chunk_size)
            for rule_results, scoring in zip(all_results, scorings):
                rule_results.set_data(scoring, CorpusOptions.TRAIN, True)

        elif not use_baseline_prediction:
            scorings = parallel_scoring(paths['train_intend_tsv'], paths['train_error_predict'], generated_rules, pool, chunk_size)
            for rule_results, scoring in zip(all_results, scorings):
                rule_results.set_data(scoring, CorpusOptions.TRAIN, True)

        sorted_rules = scores.sort_rules_by_train_results(all_results, generated_rules, score, selected_results)

        all_results = []

        print('sorted rules: %d' % len(sorted_rules))
        if len(sorted_rules) == 0:
            break

        selected_rule_index = 0

        # cross validation to select best rule, which is meaningful in cv set
        
        if do_cv:
            scorings = parallel_scoring(paths['cv_intend_tsv'], paths['cv_intend_predict'], sorted_rules, pool, chunk_size)

            for scoring in scorings:
                rule_results = RuleResults()
                rule_results.set_data(scoring, CorpusOptions.CV, False)
                all_results.append(rule_results)
    
            if use_seperate_error_corpus:
                scorings = parallel_scoring(paths['cv_error_tsv'], paths['cv_error_predict'], sorted_rules, pool, chunk_size)
                for rule_results, scoring in zip(all_results, scorings):
                    rule_results.set_data(scoring, CorpusOptions.CV, True)
    
            elif not use_baseline_prediction:
                scorings = parallel_scoring(paths['cv_intend_tsv'], paths['cv_error_predict'], sorted_rules, pool, chunk_size)
                for rule_results, scoring in zip(all_results, scorings):
                    rule_results.set_data(scoring, CorpusOptions.CV, True)
        
            selected_rule_index = scores.select_rule_by_cv_results(all_results, score, selected_results)

            print('selected rule: %d' % selected_rule_index)
            if selected_rule_index == -1:
                break

            all_results = []

        selected_rule = sorted_rules[selected_rule_index]
        rule_set.append(selected_rule)
        
        selected_results = RuleResults()

        _apply_incremental_prediction(paths['train_intend_tsv'], paths['train_intend_predict'], paths['train_intend_predict_tmp'], paths['train_error_tsv'], paths['train_error_predict'], paths['train_error_predict_tmp'], confusion_set, selected_rule, CorpusOptions.TRAIN, use_seperate_error_corpus, use_baseline_prediction, selected_results)
        if do_cv:
            _apply_incremental_prediction(paths['cv_intend_tsv'], paths['cv_intend_predict'], paths['cv_intend_predict_tmp'], paths['cv_error_tsv'], paths['cv_error_predict'], paths['cv_error_predict_tmp'], confusion_set, selected_rule, CorpusOptions.CV, use_seperate_error_corpus, use_baseline_prediction, selected_results)
        _apply_incremental_prediction(paths['test_intend_tsv'], paths['test_intend_predict'], paths['test_intend_predict_tmp'], paths['test_error_tsv'], paths['test_error_predict'], paths['test_error_predict_tmp'], confusion_set, selected_rule, CorpusOptions.TEST, use_seperate_error_corpus, use_baseline_prediction, selected_results)

        common_data = {'index': len(rule_set) - 1,
                       'rule': selected_rule.__str__(),
                       'origin': selected_rule.origin_token,
                       'replace': selected_rule.replace_token,
                       'skips': selected_rule_index}

        selected_results.set_common(common_data)

        logger.out(selected_results, artificial_error, do_weighting)
        export_rule_set(paths['rules'], rule_set)

    # clean up
    logger.close()
    pool.close()

def _generate_rules_from_prediction_efficient(paths, rule_generation_funcs, confusion_set, use_baseline_prediction, use_seperate_error_corpus):
    result_rules = {}

    intend_tsv_reader = TSVReader(paths['train_intend_tsv'])
    intend_predict_open = open(paths['train_intend_predict'], 'r')

    if not use_seperate_error_corpus and not use_baseline_prediction:
        error_predict_open = open(paths['train_error_predict'], 'r')

    for tsv_data in intend_tsv_reader:

        intend_predict_token = intend_predict_open.readline().strip()

        for i, gen_func in enumerate(rule_generation_funcs):
            new_rules = gen_func(tsv_data.tokens, tsv_data.pos, tsv_data.index, tsv_data.label, intend_predict_token, confusion_set)
            for r in new_rules:
                result_rules[r.hash()] = r

        if not use_seperate_error_corpus and not use_baseline_prediction:
            error_predict_token = error_predict_open.readline().strip()
            if error_predict_token != intend_predict_token:
                new_rules = gen_func(tsv_data.tokens, tsv_data.pos, tsv_data.index, tsv_data.label, error_predict_token, confusion_set)
                for r in new_rules:
                    result_rules[r.hash()] = r

    intend_predict_open.close()

    if not use_seperate_error_corpus and not use_baseline_prediction:
        error_predict_open.close()

    if use_seperate_error_corpus:
        error_tsv_reader = TSVReader(paths['train_error_tsv'])
        error_predict_open = open(paths['train_error_predict'], 'r')

        for tsv_data in error_tsv_reader:

            error_predict_token = error_predict_open.readline().strip()

            for i, gen_func in enumerate(rule_generation_funcs):
                new_rules = gen_func(tsv_data.tokens, tsv_data.pos, tsv_data.index, tsv_data.label, error_predict_token, confusion_set)
                for r in new_rules:
                    result_rules[r.hash()] = r

        error_predict_open.close()

    return [val for key, val in result_rules.items()]

