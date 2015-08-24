__author__ = 'Christoph Jansen, HTW Berlin'

import copy

class TokenOrPOS:
    def __init__(self, value: str, isToken: bool):
        self.value = value
        self.isToken = isToken

class Collocations:
    def __init__(self, check_sequence: [TokenOrPOS], k: int, origin_token: str, replace_token: str):
        self.check_sequence = check_sequence
        self.k = k
        self.origin_token = origin_token
        self.replace_token = replace_token

    def apply(self, tokens: [str], pos_tags: [str], index: int) -> str:
        len_tokens = len(tokens)
        if tokens[index] == self.origin_token:
            idx = index + self.k
            if idx < 0:
                return tokens[index]
            found_sequence = True
            for j in range(len(self.check_sequence)):
                if idx == index:
                    idx += 1
                if idx >= len_tokens:
                    found_sequence = False
                    break

                check_item = self.check_sequence[j]
                if check_item.isToken:
                    if tokens[idx] != check_item.value:
                        found_sequence = False
                        break
                else:
                    if pos_tags[idx] != check_item.value:
                        found_sequence = False
                        break

                idx += 1

            if found_sequence:
                return self.replace_token
        return tokens[index]

    def _sequences_equal(self, sequence_a, sequence_b):
        if len(sequence_a) != len(sequence_b):
            return False
        for a, b in zip(sequence_a, sequence_b):
            if (a.isToken and not b.isToken) or (not a.isToken and b.isToken):
                return False
            if a.value != b.value:
                return False
        return True

    def equals(self, rule) -> bool:
        if self.k == rule.k and self.origin_token == rule.origin_token and self.replace_token == rule.replace_token and self._sequences_equal(self.check_sequence, rule.check_sequence):
            return True
        return False

    def __str__(self):
        fill_symbol = '<>'
        result = []
        copy = list(self.check_sequence)

        start = min(self.k, 0)
        while True:
            if start == 0:
                result.append(self.origin_token)
            elif not copy:
                if start > 0:
                    break
                else:
                    result.append(fill_symbol)
            elif start < self.k:
                result.append(fill_symbol)
            else:
                result.append(copy[0].value)
                del copy[0]

            start += 1

        result_string = ' '.join(result)
        return result_string

    def hash(self):
        return 'collo%s%s%s%s' %(str(self), str(self.k), self.origin_token, self.replace_token)

class CoOccurrences:
    def __init__(self, check_token: str, window: int,
                 origin_token: str, replace_token: str):
        self.check_token = check_token      # Feature
        self.window = window                # Fenstergroesse k
        self.origin_token = origin_token    # Quellwort
        self.replace_token = replace_token  # Zielwort

    def apply(self, tokens: [str], pos_tags: [str], index: int) -> str:
        len_tokens = len(tokens)

        if tokens[index] == self.origin_token:
            start = max(index - self.window, 0)
            end = min(index + self.window, len_tokens)
            for j in range(start, end):
                if tokens[j] == self.check_token:
                    return self.replace_token
        return tokens[index]

    def equals(self, rule) -> bool:
        if self.window == rule.window and self.check_token == rule.check_token and self.origin_token == rule.origin_token and self.replace_token == rule.replace_token:
            return True
        return False

    def __str__(self):
        return '"%s" in window -%d to %d' % (self.check_token, self.window, self.window)

    def hash(self):
        return 'coocc%s%s%s%s' %(str(self.window), self.check_token, self.origin_token, self.replace_token)

class SimpleBaselinePrediction:
    def __init__(self, origin_token: str, replace_token: str):
        self.origin_token = origin_token
        self.replace_token = replace_token

    def apply(self, tokens: [str], pos_tags: [str], index: int) -> [str]:
        result_tokens = list(tokens)
        if tokens[index] == self.origin_token:
            return self.replace_token
        return tokens[index]

    def __str__(self):
        return '%s_to_%s' % (self.origin_token, self.replace_token)

class IdentityBaselinePrediction:
    def __init__(self):
        self.origin_token = ''
        self.replace_token = ''

    def apply(self, tokens: [str], pos: [str], index: int) -> str:
        return tokens[index]

    def __str__(self):
        return 'identity'

def generate_collocations(tokens: [str], pos:[str], index:int, intend_token: str, predict_token, confusion_set: (str, str)):
    window = 3
    max_sequence_length = 2
    len_tokens = len(tokens)
    result = []

    origin_token = predict_token
    if intend_token != origin_token:
        replace_token = confusion_set[0]
        if origin_token == confusion_set[0]:
            replace_token = confusion_set[1]
        start = max(index - window, 0)
        end = min(index + window, len_tokens)
        for sequence_length in range(1, max_sequence_length + 1):
            sequences = _gen_collocation_sequences(tokens, pos, start, end, index, sequence_length)
            for sequence in sequences:
                rule = Collocations(sequence[0], sequence[1], origin_token, replace_token)
                result.append(rule)
    return result

def _gen_collocation_sequences(tokens: [str], pos: [str], start: int, end: int, i: int, sequence_length: int) -> [([TokenOrPOS], int)]:
    sequences = []
    len_tokens = len(tokens)
    for x in range(start, end):
        can_create_sequence = True
        index = x
        item_sequence = []
        for j in range(1, sequence_length + 1):
            if index == i:
                index += 1
            if index >= len_tokens:
                can_create_sequence = False
                break
            item = (tokens[index], pos[index])
            item_sequence.append(item)

            index += 1
        if can_create_sequence:
            k = x - i
            sequences += _gen_collocation_sequences_at_k(item_sequence, k)
    return sequences


def _gen_collocation_sequences_at_k(item_sequence: [(str, str)], k: int) -> [([TokenOrPOS], int)]:
    item = item_sequence[0]
    del item_sequence[0]
    sequences = [[TokenOrPOS(item[0], True)],[TokenOrPOS(item[1], False)]]
    for item in item_sequence:
        new_sequences = []
        for sequence in sequences:
            new_sequence_token = copy.copy(sequence)
            new_sequence_token.append(TokenOrPOS(item[0], True))
            new_sequence_pos = copy.copy(sequence)
            new_sequence_pos.append(TokenOrPOS(item[1], False))
            new_sequences.append(new_sequence_token)
            new_sequences.append(new_sequence_pos)
        sequences = new_sequences
    result = []
    for sequence in sequences:
        #print(len(sequence))
        result_sequence = (sequence, k)
        result.append(result_sequence)
    return result

def generate_cooccurences(tokens: [str], pos:[str], index:int, intend_token: str, predict_token, confusion_set: (str, str)):
    window = 2
    len_tokens = len(tokens)
    result = []

    origin_token = predict_token
    if intend_token != origin_token:
        start = max(index - window, 0)
        end = min(index + window, len_tokens)
        for j in range(start, end):
            check_token = tokens[j]
            if check_token != confusion_set[0] and check_token != confusion_set[1]:
                replace_token = confusion_set[0]
                if origin_token == replace_token:
                    replace_token = confusion_set[1]
                rule = CoOccurrences(check_token, window, origin_token, replace_token)
                result.append(rule)
    return result

def remove_rule_duplicates(rules) -> None:
    start = 0
    end = len(rules)
    while start < end:
        current_rule = rules[start]
        new_start = start + 1
        while new_start < end:
            if current_rule.equals(rules[new_start]):
                del rules[new_start]
                end -= 1
            else:
                new_start += 1
        start += 1