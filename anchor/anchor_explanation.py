import numpy         as np
import string

from collections     import defaultdict
from IPython.display import HTML


class AnchorExplanation:
    """Object returned by explainers"""
    def __init__(self, type_, exp_map, nlp, stopwords):
        self.type = type_
        self.exp_map = exp_map
        self.nlp = nlp
        self.stopwords = stopwords

    def names(self, partial_index=None):
        """
        Returns a list of the names of the anchor conditions.
        Args:
            partial_index (int): lets you get the anchor until a certain index.
            For example, if the anchor is (A=1,B=2,C=2) and partial_index=1,
            this will return ["A=1", "B=2"]
        """
        names = self.exp_map['names']
        if partial_index is not None:
            names = names[:partial_index + 1]
        return names

    def features(self, partial_index=None):
        """
        Returns a list of the features used in the anchor conditions.
        Args:
            partial_index (int): lets you get the anchor until a certain index.
            For example, if the anchor uses features (1, 2, 3) and
            partial_index=1, this will return [1, 2]
        """
        features = self.exp_map['feature']
        if partial_index is not None:
            features = features[:partial_index + 1]
        return features

    def precision(self, partial_index=None):
        """
        Returns the anchor precision (a float)
        Args:
            partial_index (int): lets you get the anchor precision until a
            certain index. For example, if the anchor has precisions
            [0.1, 0.5, 0.95] and partial_index=1, this will return 0.5
        """
        precision = self.exp_map['precision']
        if len(precision) == 0:
            return self.exp_map['all_precision']
        if partial_index is not None:
            return precision[partial_index]
        else:
            return precision[-1]

    def coverage(self, partial_index=None):
        """
        Returns the anchor coverage (a float)
        Args:
            partial_index (int): lets you get the anchor coverage until a
            certain index. For example, if the anchor has coverages
            [0.1, 0.5, 0.95] and partial_index=1, this will return 0.5
        """
        coverage = self.exp_map['coverage']
        if len(coverage) == 0:
            return 1
        if partial_index is not None:
            return coverage[partial_index]
        else:
            return coverage[-1]

    def examples(self, only_different_prediction=False,
                 only_same_prediction=False, partial_index=None):
        """
        Returns examples covered by the anchor.
        Args:
            only_different_prediction(bool): if true, will only return examples
            where the anchor  makes a different prediction than the original
            model
            only_same_prediction(bool): if true, will only return examples
            where the anchor makes the same prediction than the original
            model
            partial_index (int): lets you get the examples from the partial
            anchor until a certain index.
        """
        if only_different_prediction and only_same_prediction:
            print('Error: you can\'t have only_different_prediction \
and only_same_prediction at the same time')
            return []
        key = 'covered'
        if only_different_prediction:
            key = 'covered_false'
        if only_same_prediction:
            key = 'covered_true'
        size = len(self.exp_map['examples'])
        idx = partial_index if partial_index is not None else size - 1
        if idx < 0 or idx > size:
            return []
        return self.exp_map['examples'][idx][key]

    def _get_counts(self, examples, counter, negative=False):
        for x in examples:
            y = x[0].split(' ')
            y = [a for a in y if a != 'UNK' and a.lower() not in self.stopwords]
            for idx, word_ex in enumerate(y):
                if word_ex != 'UNK' and idx not in self.features() and word_ex.lower() not in self.stopwords:
                    if negative:
                        counter[idx] -= 1
                    else:
                        counter[idx] += 1
        return counter

    def _get_max_num_and_words(self, counter, upper_lim=float('inf')):
        max_num = 0
        max_words = []
        for k, v in counter.items():
            if v > max_num and v < upper_lim:
                max_words = [k]
                max_num = v
            elif v == max_num:
                max_words.append(k)
        return max_num, max_words

    def get_top_counts(self, text, words):
        counter = defaultdict(int)
        counter = self._get_counts(self.examples(only_same_prediction=True), counter)
        counter = self._get_counts(self.examples(only_different_prediction=True), counter, negative=True)
        max_num, max_words = self._get_max_num_and_words(counter)
        second_max_num, second_max_words = self._get_max_num_and_words(counter, upper_lim=max_num)
        return max_num, [words[w] for w in max_words], second_max_num, [words[w] for w in second_max_words]


    def hlstr(self, string, color='white'):
        """
        Return HTML markup highlighting text with the desired color.
        """
        return f"<mark style=background:{color}>{string} </mark>"


    def visualize_results(self, text, color_code='green', depth=2):
        if depth > 3:
            print ('Depth > 3 not supported.\n Depth 1: Anchors only, Depth 2: Common words, Depth 3: Less common words')
            return
        print ('Anchor: %s' % (' AND '.join(self.names())))
        processed = self.nlp(text)
        raw_words = np.array([x.text for x in processed], dtype='|U80')
        words = np.array([x.text for x in processed if x.text.lower() not in self.stopwords], dtype='|U80')
        max_num, max_words, second_max_num, second_max_words = self.get_top_counts(text, words)
        word_to_color = {}
        if depth >= 3:
            for w in second_max_words:
                word_to_color[w] = 'rgba(144,238,144,0.1)' if color_code == 'green' else 'rgba(255,160,122,0.1)'
        if depth >= 2:
            for w in max_words:
                word_to_color[w] = 'rgba(50,205,50,0.5)' if color_code == 'green' else 'rgba(255,69,0,0.5)'
        if depth >= 1:
            for w in self.names():
                word_to_color[w] = 'rgba(0,100,0,1)' if color_code == 'green' else 'rgba(178,34,34,1)'
        colors = []
        for word in raw_words:
            colors.append(word_to_color.get(word.lower(), '#FFFFFF'))
        return HTML("".join(list(map(self.hlstr, raw_words, colors))))
