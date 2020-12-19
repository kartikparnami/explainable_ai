import numpy as np
import sklearn

from .anchor_base import AnchorBaseBeam


class AnchorImage(object):
    def __init__(
        self,
        white=None,
        segmentation_fn=None,
    ):
        """
        AnchorImage instance.

        Args:
            white (int): If custom baseline is required
            segmentation_fn (Callable): Method to segment incoming image
        """
        self.white = white
        if segmentation_fn is None:
            from skimage.segmentation import quickshift

            segmentation_fn = lambda x: quickshift(
                x, kernel_size=4, max_dist=200, ratio=0.2  # noqa
            )
        self.segmentation = segmentation_fn

    def get_sample_fn(self, image, classifier_fn):
        """
        Return a sampling function for producing anchors

        Args:
            image (np.ndarray): Image instance to explain.
            classifier_fn (Callable): A function which gives the models predictions.
        """
        import copy

        segments = self.segmentation(image)
        fudged_image = image.copy()
        for x in np.unique(segments):
            fudged_image[segments == x] = (
                np.mean(image[segments == x][:, 0]),
                np.mean(image[segments == x][:, 1]),
                np.mean(image[segments == x][:, 2]),
            )
        if self.white is not None:
            fudged_image[:] = self.white
        features = list(np.unique(segments))
        n_features = len(features)

        true_label = np.argmax(classifier_fn(np.expand_dims(image, 0))[0])
        print("True pred", true_label)

        def sample_fn(present, num_samples, compute_labels=True):
            """
            Sampling function to produce anchors

            Args:
                present (int): Features already present in the anchor.
                num_samples (int): Number of samples to be generated.
                compute_labels (bool): Compute pred labels of the image and return or not.
            """
            data = np.random.randint(0, 2, num_samples * n_features).reshape(
                (num_samples, n_features)
            )
            data[:, present] = 1
            if not compute_labels:
                return [], data, []
            imgs = []
            for row in data:
                temp = copy.deepcopy(image)
                zeros = np.where(row == 0)[0]
                mask = np.zeros(segments.shape).astype(bool)
                for z in zeros:
                    mask[segments == z] = True
                temp[mask] = fudged_image[mask]
                imgs.append(temp)
            preds = classifier_fn(np.array(imgs))
            preds_max = np.argmax(preds, axis=1)
            labels = (preds_max == true_label).astype(int)
            # raw_data = imgs
            raw_data = data
            return raw_data, data, labels

        sample = sample_fn
        return segments, sample

    def img_scale(self, image):
        """
        Scale the image to 0-255 range

        Args:
            image (np.ndarray): Image instance to explain
        """
        scale = (0, 255)
        img_max, img_min = image.max(), image.min()
        img_std = (image - img_min) / (img_max - img_min)
        img_scaled = img_std * (scale[1] - scale[0]) + scale[0]
        return img_scaled

    def explain_instance(
        self,
        image,
        classifier_fn,
        threshold=0.95,
        delta=0.1,
        tau=0.15,
        batch_size=100,
        **kwargs
    ):
        """
        Explain the image instance using anchors.

        Args:
            image (np.adarray): Image instance to explain.
            classifier_fn (Callable): Prediction function for the model.
            threshold (float): Precision threshold for anchor.
            delta (float): Delta value for computation of bernoulli thresholds.
            tau (float): Allowed epsilon around desired confidence.
            batch_size (int): Batch size for performing beam search.
        """
        segments, sample = self.get_sample_fn(image, classifier_fn)
        exp = AnchorBaseBeam.anchor_beam(
            sample,
            delta=delta,
            epsilon=tau,
            batch_size=batch_size,
            desired_confidence=threshold,
            **kwargs
        )
        results = self.get_exp_from_beam_results(image, exp)
        mask = np.zeros(segments.shape)
        for f in results:
            mask[segments == f[0]] = 1
        scaled_image = self.img_scale(image)
        for x, row in enumerate(mask):
            for y, col in enumerate(row):
                if mask[x][y] == 0:
                    scaled_image[x][y][0] = 0
                    scaled_image[x][y][1] = 0
                    scaled_image[x][y][2] = 0
        return scaled_image, segments, results

    def get_exp_from_beam_results(self, image, exp):
        """
        Get explanation data from beam search results.

        Args:
            image (np.adarray): Image instance to explain.
            exp (dict): Explanation instance
        """
        ret = []

        features = exp["feature"]
        means = exp["mean"]
        if "negatives" not in exp:
            negatives_ = [np.array([]) for x in features]
        else:
            negatives_ = exp["negatives"]
        for f, mean, negatives in zip(features, means, negatives_):
            train_support = 0
            name = ""
            if negatives.shape[0] > 0:
                unique_negatives = np.vstack({tuple(row) for row in negatives})
                distances = sklearn.metrics.pairwise_distances(
                    np.ones((1, negatives.shape[1])), unique_negatives
                )
                negative_arrays = unique_negatives[np.argsort(distances)[0][:4]]
                negatives = []
                for n in negative_arrays:
                    negatives.append(n)
            else:
                negatives = []
            ret.append((f, name, mean, negatives, train_support))
        return ret
