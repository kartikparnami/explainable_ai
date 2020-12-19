import json
import nltk
import numpy as np
import os
import spacy
import string
import sys

from io import open
from .anchor_base import AnchorBaseBeam
from collections import defaultdict
from IPython.display import HTML


nltk.download("stopwords")


class AnchorExplanation:
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
        names = self.exp_map["names"]
        if partial_index is not None:
            names = names[: partial_index + 1]
        return names

    def examples(
        self,
        only_different_prediction=False,
        only_same_prediction=False,
        partial_index=None,
    ):
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
            print(
                "Error: you can't have only_different_prediction \
and only_same_prediction at the same time"
            )
            return []
        key = "covered"
        if only_different_prediction:
            key = "covered_false"
        if only_same_prediction:
            key = "covered_true"
        size = len(self.exp_map["examples"])
        idx = partial_index if partial_index is not None else size - 1
        if idx < 0 or idx > size:
            return []
        return self.exp_map["examples"][idx][key]

    def _get_counts(self, examples, counter, negative=False):
        """
        Gets cumulative count of each word in a positive/negative sense
        across examples.

        Args:
            examples (list): List of sentence examples in which counts have to be done.
            counter (dict): Defaultdict to add counts to.
            negative (bool): Shoule add counts or subtract.
        """
        for x in examples:
            y = x[0].split(" ")
            y = [a for a in y if a != "UNK" and a.lower() not in self.stopwords]
            for idx, word_ex in enumerate(y):
                if (
                    word_ex != "UNK"
                    and idx not in self.features()
                    and word_ex.lower() not in self.stopwords
                ):
                    if negative:
                        counter[idx] -= 1
                    else:
                        counter[idx] += 1
        return counter

    def _get_max_num_and_words(self, counter, upper_lim=float("inf")):
        """
        Gets max count and the words with the max count, the count being
        lesser than upper_lim to facilitate more than one top fetching.

        Args:
            counter (defaultdict): Defaultdict to add counts to.
            upper_lim (float/int): Max count upper limit to fetch top K
        """
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
        """
        Gets top counts for visualization.

        Args:
            text (string): The string for which counts are to be calculated.
            words (list): Words list as the internal method return indices.
        """
        counter = defaultdict(int)
        counter = self._get_counts(self.examples(only_same_prediction=True), counter)
        counter = self._get_counts(
            self.examples(only_different_prediction=True), counter, negative=True
        )
        max_num, max_words = self._get_max_num_and_words(counter)
        second_max_num, second_max_words = self._get_max_num_and_words(
            counter, upper_lim=max_num
        )
        return (
            max_num,
            [words[w] for w in max_words],
            second_max_num,
            [words[w] for w in second_max_words],
        )

    def hlstr(self, string, color="white"):
        """
        Return HTML highlighting text with given color.

        Args:
          string (string): The string to render
          color (string): HTML color for background of the string
        """
        return f"<mark style=background:{color}>{string} </mark>"

    def visualize_results(self, text, color_code="green", depth=2):
        """
        Visualize attributions results of Anchor Text Explanation.

        Args:
            text (string): Text instance to explain
            color_code (string): 'green' for positive visualization,
                                 'red' for negative
            depth (int): Depth in which visualization is needed, top-k attributions
        """
        if depth > 3:
            print(
                "Depth > 3 not supported.\n Depth 1: Anchors only, Depth 2: Common words, Depth 3: Less common words"
            )
            return
        print("Anchor: %s" % (" AND ".join(self.names())))
        processed = self.nlp(text)
        raw_words = np.array([x.text for x in processed], dtype="|U80")
        words = np.array(
            [x.text for x in processed if x.text.lower() not in self.stopwords],
            dtype="|U80",
        )
        max_num, max_words, second_max_num, second_max_words = self.get_top_counts(
            text, words
        )
        word_to_color = {}
        if depth >= 3:
            for w in second_max_words:
                word_to_color[w] = (
                    "rgba(144,238,144,0.1)"
                    if color_code == "green"
                    else "rgba(255,160,122,0.1)"
                )
        if depth >= 2:
            for w in max_words:
                word_to_color[w] = (
                    "rgba(50,205,50,0.5)"
                    if color_code == "green"
                    else "rgba(255,69,0,0.5)"
                )
        if depth >= 1:
            for w in self.names():
                word_to_color[w] = (
                    "rgba(0,100,0,1)" if color_code == "green" else "rgba(178,34,34,1)"
                )
        colors = []
        for word in raw_words:
            colors.append(word_to_color.get(word.lower(), "#FFFFFF"))
        return HTML("".join(list(map(self.hlstr, raw_words, colors))))


class AnchorText(object):
    def __init__(self, class_names=("Negative", "Positive"), mask_string="UNK"):
        """
        AnchorText instance to find a string's anchor for a given model prediction function.

        Args:
            class_names (tuple or list): list of strings
            mask_string (string): String used to mask tokens if use_unk_distribution is True.
        """
        from nltk.corpus import stopwords

        self.nlp = spacy.load("en_core_web_lg")
        self.class_names = class_names
        self.use_unk_distribution = True
        self.tg = None
        self.mask_string = mask_string
        self.stopwords = stopwords.words("english") + list(string.punctuation)

    def get_sample_fn(self, text, classifier_fn):
        """
        Return a sampling function for producing anchors

        Args:
            text (string): The text for which anchor is to be produced.
            classifier_fn (Callable): Prediction function for the model.
        """
        true_label = classifier_fn([text])[0]
        processed = self.nlp(text)
        words = np.array(
            [x.text for x in processed if x.text.lower() not in self.stopwords],
            dtype="|U80",
        )
        positions = [x.idx for x in processed]
        perturber = None

        def sample_fn(present, num_samples, compute_labels=True):
            data = np.ones((num_samples, len(words)))
            raw = np.zeros((num_samples, len(words)), "|U80")
            raw[:] = words
            for i, t in enumerate(words):
                if i in present:
                    continue
                n_changed = np.random.binomial(num_samples, 0.5)
                changed = np.random.choice(num_samples, n_changed, replace=False)
                raw[changed, i] = self.mask_string
                data[changed, i] = 0
            raw_data = [" ".join(x) for x in raw]
            labels = []
            if compute_labels:
                labels = (classifier_fn(raw_data) == true_label).astype(int)
            labels = np.array(labels)
            max_len = max([len(x) for x in raw_data])
            dtype = "|U%d" % (max(80, max_len))
            raw_data = np.array(raw_data, dtype).reshape(-1, 1)
            return raw_data, data, labels

        return words, positions, true_label, sample_fn

    def explain_instance(
        self,
        text,
        classifier_fn,
        threshold=0.95,
        delta=0.1,
        tau=0.15,
        batch_size=10,
        beam_size=4,
        **kwargs,
    ):
        """
        Explain the text instance for a given model and its predictions.

        Args:
            text (string): Text instance to explain.
            classifier_fn (Callable): Prediction function for the model.
            threshold (float): Precision threshold for anchor.
            delta (float): Delta value for computation of bernoulli thresholds.
            tau (float): Allowed epsilon around desired confidence.
            batch_size (int): Batch size for performing beam search.
            beam_size (int): Beam size for performing beam search.
        """
        if type(text) == bytes:
            text = text.decode()
        words, positions, true_label, sample_fn = self.get_sample_fn(
            text, classifier_fn
        )
        exp = AnchorBaseBeam.anchor_beam(
            sample_fn,
            delta=delta,
            epsilon=tau,
            batch_size=batch_size,
            desired_confidence=threshold,
            stop_on_first=True,
            coverage_samples=1,
            **kwargs,
        )
        exp["names"] = [words[x] for x in exp["feature"]]
        exp["positions"] = [positions[x] for x in exp["feature"]]
        exp["instance"] = text
        exp["prediction"] = true_label
        explanation = AnchorExplanation("text", exp, self.nlp, self.stopwords)
        return explanation
