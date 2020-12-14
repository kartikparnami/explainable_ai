import matplotlib    as mpl
import numpy         as np
import spacy

from IPython.display import HTML
from lime.lime_text  import LimeTextExplainer
from PyDictionary    import PyDictionary


class LimeCounterfactualText(object):

    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self.dictionary = PyDictionary()
        self.explainer = LimeTextExplainer()

    def hlstr(self, string, color='white', replacement=None):
        """
        Return HTML markup highlighting text with the desired color.
        """
        if replacement:
            return f"<mark style=background-color:{color}><strike>{string}</strike> {replacement} </mark>"
        else:
            return f"<mark style=background-color:{color}>{string} </mark>"

    def colorize(self, attrs, cmap='PiYG'):
        """
        Compute hex colors based on the attributions for a single instance.
        Uses a diverging colorscale by default and normalizes and scales
        the colormap so that colors are consistent with the attributions.
        """
        cmap_bound = len(attrs)
        norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
        cmap = mpl.cm.get_cmap(cmap)

        # now compute hex values of colors
        colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), range(1, cmap_bound + 1)))
        return colors

    def _explain_instance_with_unk(self, text, predict_lr):
        exp = self.explainer.explain_instance(text, predict_lr, labels=(0, 1))
        explanation_lime = exp.as_list()
        score_new = predict_lr([text])[0]
        class_predicted = 1 if score_new[0] < 0.5 else 0
        processed = self.nlp(text)
        instance = ' '.join(np.array([x.text for x in processed], dtype='|U80'))
        instance_list = instance.split(' ')
        k, blacklisted_features, number_perturbed, replacements = 0, [], 0, {}
        while ((score_new[class_predicted] >= 0.5) and (k != len(explanation_lime)) and (number_perturbed < 20)):
            number_perturbed, features_removed, replacements = 0, [], {}
            k += 1
            perturbed_instance = instance_list[:]
            for i in range(k):
                exp = self.explainer.explain_instance(' '.join(perturbed_instance), predict_lr, labels=(0, 1))
                exp_list = exp.as_list()
                best_score, best_feature = -1, None
                for feature in exp_list:
                    if feature not in blacklisted_features:
                        if class_predicted == 1 and feature[1] > 0 and feature[0] != 'UNK' and abs(feature[1]) > best_score:
                            best_score = abs(feature[1])
                            best_feature = feature
                        elif class_predicted == 0 and feature[1] < 0 and feature[0] != 'UNK' and abs(feature[1]) > best_score:
                            best_score = abs(feature[1])
                            best_feature = feature
                if best_feature[0] not in perturbed_instance:
                    blacklisted_features.append(best_feature)
                else:
                    perturbed_instance = [x if x != best_feature[0] else 'UNK' for x in perturbed_instance]
                    replacements[best_feature[0]] = 'UNK'
                    number_perturbed += 1
                    features_removed.append(best_feature[0])
            score_new = predict_lr([' '.join(perturbed_instance)])[0]
        colors = self.colorize(features_removed)
        all_colors = {word: color for word, color in zip(reversed(features_removed), colors)}
        replacement_list = [replacements.get(word) for word in instance_list]
        color_list = [all_colors.get(word, '#FFFFFF') for word in instance_list]
        return HTML("".join(list(map(self.hlstr, instance_list, color_list, replacement_list))))

    def _explain_instance_with_antonyms(self, text, predict_lr):
        exp = self.explainer.explain_instance(text, predict_lr, labels=(0, 1))
        explanation_lime = exp.as_list()
        score_new = predict_lr([text])[0]
        class_predicted = 1 if score_new[0] < 0.5 else 0
        processed = self.nlp(text)
        instance = ' '.join(np.array([x.text for x in processed], dtype='|U80'))
        instance_list = instance.split(' ')
        k, blacklisted_features, number_perturbed, replacements = 0, [], 0, {}
        while ((score_new[class_predicted] >= 0.5) and (k != len(explanation_lime)) and (number_perturbed < 20)):
            number_perturbed, features_removed, replacements = 0, [], {}
            k += 1
            perturbed_instance = instance_list[:]
            for i in range(k):
                exp = self.explainer.explain_instance(' '.join(perturbed_instance), predict_lr, labels=(0, 1))
                exp_list = exp.as_list()
                best_score, best_feature = -1, None
                for feature in exp_list:
                    if feature not in blacklisted_features:
                        if class_predicted == 1 and feature[1] > 0 and feature[0] != 'UNK' and abs(feature[1]) > best_score:
                            best_score = abs(feature[1])
                            best_feature = feature
                        elif class_predicted == 0 and feature[1] < 0 and feature[0] != 'UNK' and abs(feature[1]) > best_score:
                            best_score = abs(feature[1])
                            best_feature = feature
                if best_feature[0] not in perturbed_instance:
                    blacklisted_features.append(best_feature)
                else:
                    ant = self.dictionary.antonym(best_feature[0])
                    if len(ant):
                        perturbed_instance = [x if x != best_feature[0] else ant[0] for x in perturbed_instance]
                        replacements[best_feature[0]] = ant[0]
                    else:
                        perturbed_instance = [x if x != best_feature[0] else 'UNK' for x in perturbed_instance]
                        replacements[best_feature[0]] = 'UNK'
                    number_perturbed += 1
                    features_removed.append(best_feature[0])
            score_new = predict_lr([' '.join(perturbed_instance)])[0]
        colors = self.colorize(features_removed)
        all_colors = {word: color for word, color in zip(reversed(features_removed), colors)}
        replacement_list = [replacements.get(word) for word in instance_list]
        color_list = [all_colors.get(word, '#FFFFFF') for word in instance_list]
        return HTML("".join(list(map(self.hlstr, instance_list, color_list, replacement_list))))

    def explain_instance(self, text, predict_lr, with_antonyms=False):
        return self._explain_instance_with_unk(text, predict_lr) if not with_antonyms else self._explain_instance_with_antonyms(text, predict_lr)
