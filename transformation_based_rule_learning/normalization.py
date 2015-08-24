__author__ = 'Christoph Jansen, HTW Berlin'

import nltk
from io_wrapper import PersistantBalancedCorpusIO
from io_wrapper import TSVReader
from io_wrapper import TSVWriter
from io_wrapper import TSVData
import re
import os
import shutil

def split_tokens_and_pos(tokens_with_pos: [str]):
    tokens = []
    pos = []
    for twp in tokens_with_pos:
        twp_split = twp.split('|')
        tokens.append(twp_split[0])
        pos.append(twp_split[1])
    return (tokens, pos)

def join_tokens_and_pos(tokens: [str], pos: [str]):
    result = []
    for t in zip(tokens, pos):
        result.append('|'.join(t))
    return result

def nltk_corpus_to_file(corpus, output_path: str):
    sentences = corpus.sents()
    open_output = open(output_path, 'w')
    for sentence in sentences:
        joined = ' '.join(sentence)
        print(joined, file=open_output)
    open_output.close()

def pretagged_nltk_corpus_to_file(corpus, output_path: str):
    sentences = corpus.tagged_sents()
    open_output = open(output_path, 'w')
    for sentence in sentences:
        untupled = [nltk.tag.tuple2str(t) for t in sentence]
        joined = ' '.join(untupled)
        print(joined, file=open_output)
    open_output.close()

def combine_parts_of_amazon_corpus(full_corpus, fixed_part_corpus, out_corpus, confusion_set):
    full_reader = TSVReader(full_corpus)
    fixed_part_reader = TSVReader(fixed_part_corpus)
    fixed_part_iter = fixed_part_reader.__iter__()
    writer = TSVWriter(out_corpus)
    counter = 0
    then_correct_counter = 0
    then_wrong_counter = 0
    than_correct_counter = 0
    than_wrong_counter = 0

    for data in full_reader:
        if data.is_correct:
            if data.tokens[data.index] in confusion_set and data.label in confusion_set:
                writer.out(data)
        else:
            try:
                data = fixed_part_reader.__next__()
                counter += 1
            except:
                continue
            if not (data.tokens[data.index] in confusion_set and data.label in confusion_set):
                continue
            if data.label == confusion_set[0]:
                data.tokens[data.index] = confusion_set[1]
            elif data.label == confusion_set[1]:
                data.tokens[data.index] = confusion_set[0]
            else:
                continue
            data.is_correct = False
            writer.out(data)

        if data.tokens[data.index] == 'then':
            if data.is_correct:
                then_correct_counter += 1
            else:
                then_wrong_counter += 1
        else:
            if data.is_correct:
                than_correct_counter += 1
            else:
                than_wrong_counter += 1

    writer.close()
    print(counter)
    print('Then Correct: %d' % then_correct_counter)
    print('Then Wrong: %d' % then_wrong_counter)
    print('Than Correct: %d' % than_correct_counter)
    print('Than Wrong: %d' % than_wrong_counter)

def contains_filter_tokens(tokens: [str], filter_tokens: [str]) -> bool:
    if not filter_tokens:
        return True
    for filter_token in filter_tokens:
        if filter_token in tokens:
            return True
    return False

def pos_tag_corpus(input_path: str, output_path: str):
    io = PersistantBalancedCorpusIO(input_path, output_path)
    for line in io:
        tokens = nltk.word_tokenize(line)
        pos_tags = nltk.pos_tag(tokens)
        pos_tags_str = []
        for pos_tag in pos_tags:
            pos_tags_str.append(nltk.tag.tuple2str(pos_tag))
        result = ' '.join(pos_tags_str)
        io.out(result)

def normalize(token: str) -> str:
    return token.lower()

def normalize_amazon_corpus(input_file, output_file):
    rx = re.compile('III[^I]+III')
    reader = TSVReader(input_file)
    writer = TSVWriter(output_file)
    skips = 0
    for i, data in enumerate(reader):
        norm_tokens = []
        for token in data.tokens:
            m = rx.search(token)
            if m:
                match = m.group(0)
                splits = rx.split(token)
                norm_splits = []
                for split in splits:
                    norm_splits.append(normalize(split))
                norm_token = match.join(norm_splits)
                norm_tokens.append(norm_token)
            else:
                norm_tokens.append(normalize(token))

        data.tokens = norm_tokens
        #word_index = amazon_char_to_word_index(data.tokens, data.index)
        #if word_index == -1 or not data.tokens[word_index] in ['then', 'than']:
        #   skips += 1
        #    continue
        #data.index = word_index
        writer.out(data)
    print('total skips: %d' % skips)
    writer.close()

def amazon_char_to_word_index(tokens, char_index):
    word_index = 0
    char_index_count = 0

    for token in tokens:
        if char_index_count > char_index:
            return -1
        if char_index_count == char_index:
            return word_index
        char_index_count += len(token) + 1
        if token == ',' or '\'' == token[0] or token == 'r' or token == '$' or token == 'n\'t' or token == '(' or token == ')' or token == '"' or token == '`' or token == '..' or token == '...' or token == ';' or token == ':' or token == '%':
            char_index_count -= 1
        word_index += 1
    raise Exception('WTF')

def normalize_acrotagged_corpus(input_file, output_file):
    rx = re.compile('III[^I]+III')
    io = PersistantBalancedCorpusIO(input_file, output_file)
    for line in io:
        tokens, pos = split_tokens_and_pos(line.split())
        norm_tokens = []
        for token in tokens:
            m = rx.search(token)
            if m:
                match = m.group(0)
                splits = rx.split(token)
                norm_splits = []
                for split in splits:
                    norm_splits.append(normalize(split))
                norm_token = match.join(norm_splits)
                norm_tokens.append(norm_token)
            else:
                norm_tokens.append(normalize(token))

        result_twp = join_tokens_and_pos(norm_tokens, pos)
        result = ' '.join(result_twp)
        io.out(result)

def count_tokens(paths, confusion_set:(str, str)):
    reader = TSVReader(paths['filtered_tsv'])
    token_a_count = 0
    token_b_count = 0
    for data in reader:
        if data.label == confusion_set[0]:
            token_a_count += 1
        elif data.label == confusion_set[1]:
            token_b_count += 1

    return (token_a_count, token_b_count)


def filter_corpus_to_tsv(input_path: str, output_path: str, confusion_set: (str, str)):
    open_input = open(input_path, 'r')
    writer = TSVWriter(output_path)

    for line in open_input:
        tokens, pos = split_tokens_and_pos(line.split())
        for i, token in enumerate(tokens):
            if token in confusion_set:
                d = TSVData(True, token, i, tokens, pos)
                writer.out(d)

    open_input.close()
    writer.close()

def directed_error_tsv(paths, confusion_set:(str, str), cv=False):
    tmp_dir = paths['tmp']
    corpora = ['train', 'test']
    if cv:
        corpora.append('cv')

    for cor in corpora:
        intend_tmp = os.path.join(tmp_dir, '%s_intend.tmp' % cor)
        directed_error_corpus(paths['%s_intend_tsv' % cor], intend_tmp, paths['%s_error_tsv' % cor], confusion_set)
        shutil.copyfile(intend_tmp, paths['%s_intend_tsv' % cor])
        os.remove(intend_tmp)

def directed_error_corpus(input_path: str, output_intend_path: str, output_error_path: str, confusion_set:(str, str)):
    reader = TSVReader(input_path)
    intend_writer = TSVWriter(output_intend_path)
    error_writer = TSVWriter(output_error_path)

    for data in reader:
        if data.label == confusion_set[0]:
            intend_writer.out(data)
        elif data.label == confusion_set[1]:
            data.is_correct = False
            data.tokens[data.index] = confusion_set[0]
            error_writer.out(data)

    intend_writer.close()
    error_writer.close()

def split_amazon_error_corpus(input_path: str, output_intend_path: str, output_error_path: str, confusion_set:(str, str)):
    reader = TSVReader(input_path)
    intend_writer = TSVWriter(output_intend_path)
    error_writer = TSVWriter(output_error_path)

    for data in reader:
        if data.label == confusion_set[0] and data.is_correct:
            intend_writer.out(data)
        elif data.label == confusion_set[1] and not data.is_correct:
            error_writer.out(data)

    intend_writer.close()
    error_writer.close()

def token_freq_tsv(path, confusion_set):
    reader = TSVReader(path)
    c0 = 0
    c1 = 0
    for d in reader:
        if d.label == confusion_set[0]:
            c0 += 1
        elif d.label == confusion_set[1]:
            c1 += 1
    return (c0, c1)

def dynamically_split_tsv(paths, confusion_set, test=0, cv=False):
    split_percentage = 20
    num_of_splits = 5

    token_a = confusion_set[0]
    token_b = confusion_set[1]

    frequencies = token_freq_tsv(paths['filtered_tsv'], confusion_set)
    token_a_freq = frequencies[0]
    token_b_freq = frequencies[1]

    split_token_a_freq = (token_a_freq * split_percentage) // 100
    split_token_b_freq = (token_b_freq * split_percentage) // 100

    tmp_dir = paths['tmp']

    splits = []
    for i in range(num_of_splits):
        file = os.path.join(tmp_dir, 'split%d.tsv' % i)
        dict = {'token_a_count': 0,
                'token_b_count': 0,
                'file': file,
                'io': TSVWriter(file)}
        splits.append(dict)

    reader = TSVReader(paths['filtered_tsv'])

    for d in reader:
        for split in splits:
            if d.label == token_a:
                if split['token_a_count'] < split_token_a_freq:
                    split['token_a_count'] = split['token_a_count'] + 1
                    split['io'].out(d)
                    break
            elif d.label == token_b:
                if split['token_b_count'] < split_token_b_freq:
                    split['token_b_count'] = split['token_b_count'] + 1
                    split['io'].out(d)
                    break

    for split in splits:
        split['io'].close()
        split['io'] = TSVReader(split['file'])

    if cv:
        cv_writer = TSVWriter(paths['cv_intend_tsv'])
        for d in splits[cv]['io']:
            cv_writer.out(d)
        cv_writer.close()
    
    test_writer = TSVWriter(paths['test_intend_tsv'])
    for d in splits[test]['io']:
        test_writer.out(d)
    test_writer.close()

    train_writer = TSVWriter(paths['train_intend_tsv'])
    for i, split in enumerate(splits):
        if (cv and i == cv) or i == test:
            continue
        for d in split['io']:
            train_writer.out(d)
    train_writer.close()

    for split in splits:
        os.remove(split['file'])