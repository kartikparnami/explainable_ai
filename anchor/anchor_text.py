import json
import nltk
import numpy as np
import os
import spacy
import string
import sys

from io                  import open
from .anchor_base        import AnchorBaseBeam
from .anchor_explanation import AnchorExplanation


nltk.download('stopwords')


class Neighbors:
    def __init__(self, nlp_obj):
        self.nlp = nlp_obj
        self.to_check = [w for w in self.nlp.vocab if w.prob >= -15 and w.has_vector]
        if not self.to_check:
            raise Exception('No vectors. Are you using en_core_web_sm? It should be en_core_web_lg')
        self.n = {}

    def neighbors(self, word):
        word = word
        orig_word = word
        if word not in self.n:
            if word not in self.nlp.vocab.strings:
                self.n[word] = []
            elif not self.nlp.vocab[word].has_vector:
                self.n[word] = []
            else:
                word = self.nlp.vocab[word]
                queries = [w for w in self.to_check
                            if w.is_lower == word.is_lower]
                if word.prob < -15:
                    queries += [word]
                by_similarity = sorted(
                    queries, key=lambda w: word.similarity(w), reverse=True)
                self.n[orig_word] = [(self.nlp(w.orth_)[0], word.similarity(w))
                                     for w in by_similarity[:500]]
                                    #  if w.lower_ != word.lower_]
        return self.n[orig_word]


class AnchorText(object):
    """bla"""
    def __init__(self, class_names=('Negative', 'Positive'), mask_string='UNK'):
        """
        Args:
            nlp: spacy object
            class_names: list of strings
            use_unk_distribution: if True, the perturbation distribution
                will just replace words randomly with mask_string.
                If False, words will be replaced by similar words using word
                embeddings
            mask_string: String used to mask tokens if use_unk_distribution is True.
        """
        from nltk.corpus import stopwords
        self.nlp = spacy.load('en_core_web_lg')
        self.class_names = class_names
        self.use_unk_distribution = True
        self.tg = None
        self.neighbors = Neighbors(self.nlp)
        self.mask_string = mask_string
        self.stopwords = stopwords.words('english') + list(string.punctuation)

    def get_sample_fn(self, text, classifier_fn, onepass=False, use_proba=False):
        true_label = classifier_fn([text])[0]
        processed = self.nlp(text)
        words = np.array([x.text for x in processed if x.text.lower() not in self.stopwords], dtype='|U80')
        positions = [x.idx for x in processed]
        perturber = None
        def sample_fn(present, num_samples, compute_labels=True):
            data = np.ones((num_samples, len(words)))
            raw = np.zeros((num_samples, len(words)), '|U80')
            raw[:] = words
            for i, t in enumerate(words):
                if i in present:
                    continue
                n_changed = np.random.binomial(num_samples, .5)
                changed = np.random.choice(num_samples, n_changed,
                                           replace=False)
                raw[changed, i] = self.mask_string
                data[changed, i] = 0
            raw_data = [' '.join(x) for x in raw]
            labels = []
            if compute_labels:
                labels = (classifier_fn(raw_data) == true_label).astype(int)
            labels = np.array(labels)
            max_len = max([len(x) for x in raw_data])
            dtype = '|U%d' % (max(80, max_len))
            raw_data = np.array(raw_data, dtype).reshape(-1, 1)
            return raw_data, data, labels
        return words, positions, true_label, sample_fn

    def explain_instance(self, text, classifier_fn, threshold=0.95,
                          delta=0.1, tau=0.15, batch_size=10, onepass=False,
                          use_proba=False, beam_size=4,
                          **kwargs):
        if type(text) == bytes:
            text = text.decode()
        words, positions, true_label, sample_fn = self.get_sample_fn(
            text, classifier_fn, onepass=onepass, use_proba=use_proba)
        exp = AnchorBaseBeam.anchor_beam(
            sample_fn, delta=delta, epsilon=tau, batch_size=batch_size,
            desired_confidence=threshold, stop_on_first=True,
            coverage_samples=1, **kwargs)
        exp['names'] = [words[x] for x in exp['feature']]
        exp['positions'] = [positions[x] for x in exp['feature']]
        exp['instance'] = text
        exp['prediction'] = true_label
        explanation = AnchorExplanation('text', exp, self.nlp, self.stopwords)
        return explanation
