import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class DeterministicEncoder(object):
    """The Encoder."""

    def __init__(self, output_sizes):
        """CNP encoder.

        Args:
          output_sizes: An iterable containing the output sizes of the encoding MLP.
        """
        self._output_sizes = output_sizes

    def __call__(self, context_x, context_y, num_context_points):
        """Encodes the inputs into one representation.

        Args:
          context_x: Tensor of size bs x observations x m_ch. For this 1D regression
              task this corresponds to the x-values.
          context_y: Tensor of size bs x observations x d_ch. For this 1D regression
              task this corresponds to the y-values.
          num_context_points: A tensor containing a single scalar that indicates the
              number of context_points provided in this iteration.

        Returns:
          representation: The encoded representation averaged over all context
              points.
        """

        # Concatenate x and y along the filter axes
        encoder_input = tf.concat([context_x, context_y], axis=-1)
        print("context_x_shape: ", context_x.shape.as_list())
        print("context_y_shape: ", context_y.shape.as_list())
        print("encoder_input_shape: ", encoder_input.shape.as_list())

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, _, filter_size = encoder_input.shape.as_list()
        hidden = tf.reshape(encoder_input, (batch_size * num_context_points, -1))
        hidden.set_shape((None, filter_size))

        # Pass through MLP
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            for i, size in enumerate(self._output_sizes[:-1]):
                hidden = tf.nn.relu(
                    tf.layers.dense(hidden, size, name="Encoder_layer_{}".format(i))
                )

            # Last layer without a ReLu
            hidden = tf.layers.dense(
                hidden, self._output_sizes[-1], name="Encoder_layer_{}".format(i + 1)
            )

        # Bring back into original shape
        hidden = tf.reshape(hidden, (batch_size, num_context_points, size))
        print("hidden_shape: ", hidden.shape.as_list())

        # Aggregator: take the mean over all points
        representation = tf.reduce_mean(hidden, axis=1)
        print("representation_shape: ", representation.shape.as_list())

        return representation
