import copy
import logging
import string

import numpy as np
import matplotlib as mpl
import tensorflow as tf

from IPython.display import HTML
from tensorflow.keras.models import Model
from typing import Callable, TYPE_CHECKING, Union, List, Tuple


if TYPE_CHECKING:  # pragma: no cover
    import keras  # noqa

logger = logging.getLogger(__name__)


def hlstr(string, color="white"):
    """
    Return HTML highlighting text with given color.

    Args:
      string (string): The string to render
      color (string): HTML color for background of the string
    """
    return f"<mark style=background-color:{color}>{string} </mark>"


def colorize(attrs, cmap="PiYG"):
    """
    Compute hex colors based on the attributions for a single instance.

    Args:
      attrs (np.ndarray): Attributions for an instance
      cmap (string): Type of color map to use
    """
    cmap_bound = np.abs(attrs).max()
    norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)

    colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))
    return colors


def select_target(ps, ts):
    """
    Select the target

    Args:
        ps (tf.Tensor): Predictions from the model
        ts (tf.Tensor): Target tensor
    """
    ps = tf.linalg.diag_part(tf.gather(ps, ts, axis=1))
    return ps


def gradients_input(model, x, target):
    """
    Calculates the gradients of the target class output with respect
    to each input feature.

    Args:
        model (Model): Tensorflow model.
        x (tf.Tensor): Input data
        target (tf.Tensor or None): Target for which the gradients are calculated
    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x)
        if len(model.output_shape) > 1 and model.output_shape[1] > 1:
            preds = select_target(preds, target)

    grads = tape.gradient(preds, x)

    return grads


def _gradients_layer(
    model,
    layer,
    orig_call,
    x,
    target,
):
    """
    Calculates the gradients of the target class output (or the output if the output dimension is equal to 1)
    with respect to each element of `layer`.

    Args:
        model (keras.models.Model): Tensorflow model.
        layer (keras.layers.Layer): Layer wrt which the gradients are calculated.
        orig_call (Callable): `call` method of the layer.
        x (tf.Tensor): Input data point.
        target (None or tf.Tensor): Target for which the gradients are calculated.
    """

    def watch_layer(layer, tape):
        """
        Make an intermediate hidden `layer` watchable by the `tape`.

        Args:
            layer (keras.layers.Layer): Intermediate layer.
            tape (tf.GradientTape): Tape for computing gradients.
        """

        def dec(func):
            def wrap(*args, **kwargs):
                layer.result = func(*args, **kwargs)
                tape.watch(layer.result)
                return layer.result

            return wrap

        layer.call = dec(layer.call)
        return layer

    with tf.GradientTape() as tape:
        watch_layer(layer, tape)
        preds = model(x)
        if len(model.output_shape) > 1 and model.output_shape[1] > 1:
            preds = select_target(preds, target)

    grads = tape.gradient(preds, layer.result)
    delattr(layer, "result")
    layer.call = orig_call
    return grads


def _sum_integral_terms(step_sizes, grads):
    """
    Sums the terms in the integral path.

    Args:
    step_sizes (list): Weights in the path integral sum.
    grads (tf.Tensor or np.ndarray): Gradients to sum for
                                     each feature.
    """
    input_str = string.ascii_lowercase[1 : len(grads.shape)]
    if isinstance(grads, tf.Tensor):
        step_sizes = tf.convert_to_tensor(step_sizes)
    einstr = "a,a{}->{}".format(input_str, input_str)
    sums = (
        tf.einsum(einstr, step_sizes, grads)
        if isinstance(grads, tf.Tensor)
        else np.einsum(einstr, step_sizes, grads)
    )
    return sums.numpy() if isinstance(grads, tf.Tensor) else sums


def _format_input_baseline(X, baselines):
    """
    Formats baselines to return a numpy array.

    Args:
        X (np.adarray):  Input data points.
        baselines (None or np.ndarray): Baselines.
    """
    if baselines is None:
        bls = np.zeros(X.shape).astype(X.dtype)
    else:
        bls = baselines.astype(X.dtype)

    return bls


def _format_target(target, nb_samples):
    """
    Formats target to return a list.

    Args:
        target (None or np.ndarray): Original target.
        nb_samples (int): Number of samples in the batch.
    """
    return [t.astype(int) for t in target] if target is not None else None


class IntegratedGradientsText(object):
    def __init__(
        self,
        model,
        layer,
        n_steps,
        internal_batch_size,
    ):
        """
        An implementation of Integrated gradients method for Tensorflow/Keras text models.
        For details of the method see the original paper:
        https://arxiv.org/abs/1703.01365 .

        Args:
            model (keras.Model): Tensorflow model.
            layer (keras.layers.Layer): Layer with respect to which the gradients are calculated.
                                        If not provided, the gradients are calculated with respect to the input.
            n_steps (int): Number of step in the path integral approximation from the
                           baseline to the input instance.
            internal_batch_size (None or int): Batch size for the internal batching.
        """
        self.model = model
        self.input_dtype = self.model.input.dtype
        self.layer = layer
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size

    def step_sizes(self, n):
        """
        Compute step sizes.

        Args:
            n (int): The number of integration steps
        """
        return list(0.5 * np.polynomial.legendre.leggauss(n)[1])

    def alphas(self, n):
        """
        Compute alpha coefficients.

        Args:
            n (int): The number of integration steps
        """
        return list(0.5 * (1 + np.polynomial.legendre.leggauss(n)[0]))

    def explain_instance(
        self,
        X,
        baselines,
        target,
    ):
        """
        Compute attributions for each input feature.

        Args:
            X (np.ndarray): Instance for which integrated gradients attribution are computed.
            baselines (None or np.ndarray): Baselines for each instance.
            target (None or np.ndarray):  Defines which element of the model output is considered to compute the gradients.
        """

        if (
            len(self.model.output_shape) == 1 or self.model.output_shape[1] == 1
        ) and target is None:
            logger.warning(
                "It looks like you are passing a model with a scalar output and target is set to `None`."
                "If your model is a regression model this will produce correct attributions. If your model "
                "is a classification model, targets for each datapoint must be defined. "
                "Not defining the target may lead to incorrect values for the attributions."
                "Targets can be either the true classes or the classes predicted by the model."
            )

        nb_samples = len(X)

        # format and check inputs and targets
        baselines = _format_input_baseline(X, baselines)
        target = _format_target(target, nb_samples)

        # defining integral method
        step_sizes_func, alphas_func = self.step_sizes, self.alphas
        step_sizes, alphas = step_sizes_func(self.n_steps), alphas_func(self.n_steps)

        # construct paths and prepare batches
        paths = np.concatenate(
            [baselines + alphas[i] * (X - baselines) for i in range(self.n_steps)],
            axis=0,
        )
        if target is not None:
            target_paths = np.concatenate([target for _ in range(self.n_steps)], axis=0)
            paths_ds = tf.data.Dataset.from_tensor_slices((paths, target_paths)).batch(
                self.internal_batch_size
            )
        else:
            paths_ds = tf.data.Dataset.from_tensor_slices(paths).batch(
                self.internal_batch_size
            )
        paths_ds.prefetch(tf.data.experimental.AUTOTUNE)

        # fix orginal call method for layer
        if self.layer is not None:
            orig_call = self.layer.call
        else:
            orig_call = None

        # calculate gradients for batches
        batches = []
        for path in paths_ds:

            if target is not None:
                paths_b, target_b = path
            else:
                paths_b, target_b = path, None

            if self.layer is not None:
                grads_b = _gradients_layer(
                    self.model,
                    self.layer,
                    orig_call,
                    tf.dtypes.cast(paths_b, self.input_dtype),
                    target_b,
                )
            else:
                grads_b = gradients_input(
                    self.model, tf.dtypes.cast(paths_b, self.input_dtype), target_b
                )

            batches.append(grads_b)

        # tf concatatation
        grads = tf.concat(batches, 0)
        shape = grads.shape[1:]
        if isinstance(shape, tf.TensorShape):
            shape = tuple(shape.as_list())

        # invert sign of gradients for target 0 examples if classifier returns only positive class probability
        if (
            len(self.model.output_shape) == 1 or self.model.output_shape[1] == 1
        ) and target is not None:
            sign = 2 * target_paths - 1
            grads = np.array([s * g for s, g in zip(sign, grads)])

        grads = tf.reshape(grads, (self.n_steps, nb_samples) + shape)

        # sum integral terms and scale attributions
        sum_int = _sum_integral_terms(step_sizes, grads.numpy())
        if self.layer is not None:
            layer_output = self.layer.output
            model_layer = Model(self.model.input, outputs=layer_output)
            norm = (model_layer(X) - model_layer(baselines)).numpy()
        else:
            norm = X - baselines
        attributions = norm * sum_int

        attributions = attributions.sum(axis=2)

        return attributions

    def visualize(self, attrs_i, words):
        """
        Visualize the attributions as computed by integrated gradients.

        Args:
            attrs_i (np.ndarray): Attributions of each feature as computed by integrated gradients.
            words (list): List of words in the text corresponding to each attribution.
        """
        colors = colorize(attrs_i)
        return HTML("".join(list(map(hlstr, words, colors))))
