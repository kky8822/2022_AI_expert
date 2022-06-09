import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

tf.disable_v2_behavior()


class DeterministicDecoder(object):
    """The Decoder."""

    def __init__(self, output_sizes):
        """CNP decoder.

        Args:
          output_sizes: An iterable containing the output sizes of the decoder MLP
              as defined in `basic.Linear`.
        """
        self._output_sizes = output_sizes

    def __call__(self, representation, target_x, num_total_points):
        """Decodes the individual targets.

        Args:
          representation: The encoded representation of the context
          target_x: The x locations for the target query
          num_total_points: The number of target points.

        Returns:
          dist: A multivariate Gaussian over the target points.
          mu: The mean of the multivariate Gaussian.
          sigma: The standard deviation of the multivariate Gaussian.
        """

        # Concatenate the representation and the target_x

        representation = tf.tile(
            tf.expand_dims(representation, axis=1), [1, num_total_points, 1]
        )
        print("representation in dec: ", representation.shape.as_list())
        input = tf.concat([representation, target_x], axis=-1)
        print("target_x in Dec: ", target_x.shape.as_list())
        print("Decoder input shape: ", input.shape.as_list())

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, _, filter_size = input.shape.as_list()
        hidden = tf.reshape(input, (batch_size * num_total_points, -1))
        hidden.set_shape((None, filter_size))

        # Pass through MLP
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            for i, size in enumerate(self._output_sizes[:-1]):
                hidden = tf.nn.relu(
                    tf.layers.dense(hidden, size, name="Decoder_layer_{}".format(i))
                )

            # Last layer without a ReLu
            hidden = tf.layers.dense(
                hidden, self._output_sizes[-1], name="Decoder_layer_{}".format(i + 1)
            )

        # Bring back into original shape

        hidden = tf.reshape(hidden, (batch_size, num_total_points, -1))

        # Get the mean an the variance
        mu, log_sigma = tf.split(hidden, 2, axis=-1)
        print("mu in dec: ", mu.shape.as_list())
        print("log_sigma in dec: ", log_sigma.shape.as_list())

        # Bound the variance
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

        # Get the distribution
        dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

        return dist, mu, sigma
