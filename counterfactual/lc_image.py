from lime.lime_image import LimeImageExplainer

import random


class LimeCounterfactualImage(object):

    def __init__(self):
        self.lime_explainer = LimeImageExplainer()

    def explain_instance(self, image, predict_fn, step_size=1000):
        explanation = self.lime_explainer.explain_instance(image.astype("uint8"), predict_fn, top_labels=2, hide_color=0, num_samples=10)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        real_pred = explanation.top_labels[0]
        new_img = image.copy()
        all_positives_removed, out_idx, in_idx = set(), 0, 0
        while real_pred == explanation.top_labels[0]:
            positives = list()
            for x, mrow in enumerate(mask):
                for y, mcol in enumerate(mrow):
                    if mcol == 1:
                        positives.append((x, y))
            new_mask = mask.copy()
            new_img = new_img.copy()
            to_change = random.sample(positives, min(step_size, len(positives)))
            for p in to_change:
                new_img[p[0]][p[1]][0], new_img[p[0]][p[1]][1], new_img[p[0]][p[1]][2] = 0, 0, 0
                all_positives_removed.add(p)
            explanation = self.lime_explainer.explain_instance(new_img.astype("uint8"), predict_fn, top_labels=2, hide_color=0, num_samples=10)
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        grayscale_img = image.copy().astype("uint8")
        for out_idx, mrow in enumerate(grayscale_img):
            for in_idx, mcol in enumerate(mrow):
                if (out_idx, in_idx) not in all_positives_removed:
                    grayscale_img[out_idx][in_idx][0] = 0
                    grayscale_img[out_idx][in_idx][1] = 0
                    grayscale_img[out_idx][in_idx][2] = 0
        return grayscale_img
