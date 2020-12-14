import random

import tensorflow as tf


class IntegratedGradientsImage(object):

    def __init__(self,
                 num_bright_image_variants=15,
                 num_interpolations=100,
                 min_brightness_increase=0.4,
                 max_brightness_increase=0.7):
        self.num_bright_image_variants = num_bright_image_variants
        self.num_interpolations = num_interpolations
        self.min_brightness_increase = min_brightness_increase
        self.max_brightness_increase = max_brightness_increase

    def get_integrated_gradients(self,
                                 interpolations,
                                 model,
                                 class_index,
                                 num_interpolations,
                                 baseline,
                                 img):
        """
        Compute Integrated Gradients for a model based
        on the interpolations

        Args:
          interpolations (numpy.ndarray): Interpolations between image
                                          and baseline image
          model (tensorflow.python.keras.engine.training.Model): Classification model
          class_index (numpy.int64): Class index of the target class
          num_interpolations (int): Number of interpolations
          baseline (numpy.ndarray): Baseline image
          img (numpy.ndarray): Target image
        Return:
          scaled_integrated_gradients (numpy.ndarray): Integrated Gradients of the
                                                       model with respect to image
        """
        with tf.GradientTape() as tape:
            tape.watch(interpolations)
            predictions = model(interpolations)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, interpolations)

        # Use trapezoidal rule to approximate the integral.
        # See the paper for details
        grads = (grads[1:] + grads[:-1]) / 2.0

        # Reshape the tensor so as to reduce mean
        # across the interpolation axis later.
        grads_per_image = tf.expand_dims(grads, axis=0)
        integrated_gradients = tf.reduce_mean(grads_per_image, axis=1)

        # Multiplying the difference in the image and the baseline with
        # the computed integrated gradients.
        # Source: Equation 3 in Axiomatic Attribution for Deep Networks
        scaled_integrated_gradients = (
          (img - baseline) * (integrated_gradients)
        )

        return scaled_integrated_gradients

    def get_interpolations(self, img, num_interpolations):
        """
        Get interpolated path between the image and a zero baseline image
        for a given number of steps in the path

        Args:
          img (numpy.ndarray): Target image
          num_interpolations (int): Number of interpolations
        Return:
          interpolations (numpy.ndarray): Interpolations between image & baseline
          baseline (numpy.ndarray): Baseline image
        """
        baseline = tf.zeros(img.shape[1:])

        # We construct the interpolations at regularly spaced
        # intervals between the baseline and the input image.
        interpolations = []
        for idx in range(0, num_interpolations + 1):
            delta = (img - baseline) * (float(idx)/num_interpolations)
            interpolations.append(baseline + delta)

        interpolations = tf.convert_to_tensor(interpolations)
        return interpolations, baseline

    def explain_instance(self, image, preprocessor_with_brightness_fn, model, num_interpolations=None):
        """
        Predict the class for an image with the classifier
        and show integrated gradients for that image.
        
        Args:
          file_path (str): Path to image
          model (tensorflow.python.keras.engine.training.Model): Classification model
          num_interpolations (int): Number of interpolations
          show_variant_images (bool): Show brightness variant images or not
        """
        if num_interpolations is None:
            num_interpolations = self.num_interpolations
        processed_img = preprocessor_with_brightness_fn(image, 0)
        prediction = model.predict(processed_img)[0]
        class_index = 0 if prediction[0] > prediction[1] else 1
        orig_img = image.copy()

        all_intgrads = []
        for i in range(self.num_bright_image_variants):
            img = preprocessor_with_brightness_fn(
                image,
                random.uniform(self.min_brightness_increase, self.max_brightness_increase)
            )
            interpolations, baseline = self.get_interpolations(img[0], num_interpolations)
            integrated_gradients = self.get_integrated_gradients(interpolations,
                                                                 model,
                                                                 class_index,
                                                                 num_interpolations,
                                                                 baseline,
                                                                 img[0])
            all_intgrads.append(integrated_gradients)

        # Averaging the gradients across the image variants
        all_intgrads_tensor = tf.convert_to_tensor(all_intgrads)
        avg_intgrads_tensor = tf.reduce_mean(all_intgrads_tensor, axis=0)
        integrated_gradients_abs = tf.abs(avg_intgrads_tensor)

        # Converting the integrated gradients to a grayscale tensor for visualization
        grayscale_tensor = tf.reduce_sum(integrated_gradients_abs, axis=-1)
        normalized_tensor = tf.cast(
            255 * tf.image.per_image_standardization(grayscale_tensor), tf.uint8
        )

        return normalized_tensor[0].numpy()
