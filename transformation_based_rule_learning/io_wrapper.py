__author__ = 'Christoph Jansen, HTW Berlin'

import os
import normalization
import pickle

def export_rule_set(path, rule_set):
    with open(path, 'wb') as f:
        pickle.dump(rule_set, f)

def import_rule_set(path):
    with open(path, 'rb') as f:
        rule_set = pickle.load(f)
    return rule_set

class TSVData:
    def __init__(self, is_correct:bool, label:str, index:int, tokens:[str], pos:[str]):
        self.is_correct = is_correct
        self.label = label
        self.index = index
        self.tokens = tokens
        self.pos = pos

    def __str__(self):
        is_c = 'c'
        if not self.is_correct:
            is_c = 'X'
        sentence = ' '.join(normalization.join_tokens_and_pos(self.tokens, self.pos))
        return('%s\t%s\t%d\t%s' % (is_c, self.label, self.index, sentence))

class TSVReader:
    def __init__(self, input_path):
        self.open_input = open(input_path, 'r')
        self.open_input_iter = self.open_input.__iter__()
        self.headline = self.open_input_iter.__next__().strip()

    def __iter__(self):
        return self

    def _line_to_data(self, line):

        t = line.split('\t')
        is_correct = True
        if t[0] == 'X':
            is_correct = False
        label = t[1]
        index = int(t[2])
        sentence = t[3]
        if len(t[3:]) > 1:
            sentence = '\t'.join(t[3:])
        tokens, pos = normalization.split_tokens_and_pos(sentence.split())
        return TSVData(is_correct, label, index, tokens, pos)

    def __next__(self):
        try:
            line = self.open_input_iter.__next__()
        except:
            self.open_input.close()
            raise StopIteration()
        return self._line_to_data(line)

    def close(self):
        self.open_input.close()

class TSVWriter:
    def __init__(self, output_path):
        self.open_output = open(output_path, 'w')
        print('?\tlabel\tindex\tsentence', file=self.open_output)

    def out(self, tsv_data):
        print(tsv_data, file=self.open_output)

    def close(self):
        self.open_output.close()

class MultiInputReader:
    def __init__(self, input_paths):
        self.open_inputs = []
        self.iters = []
        for path in input_paths:
            open_input = open(path, 'r')
            self.open_inputs.append(open_input)
            self.iters.append(open_input.__iter__())

        self._used = False

    def __iter__(self):
        if self._used:
            raise Exception('Cannot reuse iterable. Create new Object first.')
        self._used = True
        return self

    def __next__(self):
        lines = []
        try:
            for it in self.iters:
                line = it.__next__()
                lines.append(line)
        except:
            for open_input in self.open_inputs:
                open_input.close()
            raise StopIteration()
        return tuple(lines)

    def close(self):
        for open_input in self.open_inputs:
            open_input.close()

class PersistantBalancedCorpusIO:
    def __init__(self, input_path: str, output_path: str):
        self._input_path = input_path
        self._output_path = output_path

        self._open_input = open(self._input_path, 'r')
        self._open_input_iter = self._open_input.__iter__()
        self._open_output = open(self._output_path, 'w')

        self._in_count = 0
        self._out_count = 0

        self._used = False

    def __iter__(self):
        if self._used:
            raise Exception('Cannot reuse iterable. Create new Object first.')
        self._used = True
        return self

    def __next__(self):
        if self._in_count > self._out_count:
            m = 'Cannot read new input line before using out function to store intermediate results.'
            raise Exception(m)
        self._in_count += 1
        try:
            return self._open_input_iter.__next__()
        except:
            self._open_input.close()
            self._open_output.close()
            #print('Stop iteration: processed %d lines' % self._out_count)
            raise StopIteration()

    def out(self, line: str):
        if self._out_count >= self._in_count:
            m = 'Cannot store more itermediate results without reading new input line first.'
            raise Exception(m)
        self._out_count += 1
        if not line:
            line = ''
        print(line, file=self._open_output)

    def flush(self):
        self._open_output.flush()
        os.fsync(self._open_output.fileno())

    def close(self):
        if self._in_count > self._out_count:
            m = 'Write last intermediate result before closing files'
            raise Exception(m)
        self._open_input.close()
        self._open_output.close()