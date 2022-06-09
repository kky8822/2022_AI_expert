import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from utils_for_dataset import read_h5py


from generator import GPCurvesReader
from model import DeterministicModel
from plot import plot_functions


TRAINING_ITERATIONS = int(2e5)
MAX_CONTEXT_POINTS = 10
PLOT_AFTER = int(2e4)
tf.reset_default_graph()

# Train dataset
dataset_train = GPCurvesReader(batch_size=64, max_num_context=MAX_CONTEXT_POINTS)
data_train = dataset_train.generate_curves()

# Test dataset
dataset_test = GPCurvesReader(
    batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True
)
data_test = dataset_test.generate_curves()


with read_h5py("./burgers.h5") as f:
    snapshots = f["v"][...]
    equation_kwargs = {k: v.item() for k, v in f.attrs.items()}
    print("Inputs have shape: ", snapshots.shape)
    print("equation_kwargs: ", equation_kwargs)


# # Sizes of the layers of the MLPs for the encoder and decoder
# # The final output layer of the decoder outputs two values, one for the mean and
# # one for the variance of the prediction at the target location
# encoder_output_sizes = [128, 128, 128, 128]
# decoder_output_sizes = [128, 128, 2]

# # Define the model
# model = DeterministicModel(encoder_output_sizes, decoder_output_sizes)

# # Define the loss
# print("Define loss from train_set")
# log_prob, _, _ = model(
#     data_train.query,
#     data_train.num_total_points,
#     data_train.num_context_points,
#     data_train.target_y,
# )
# loss = -tf.reduce_mean(log_prob)

# # Get the predicted mean and variance at the target points for the testing set
# print("Define mu, sigma from test_set")
# _, mu, sigma = model(
#     data_test.query, data_test.num_total_points, data_test.num_context_points
# )

# # Set up the optimizer and train step
# optimizer = tf.train.AdamOptimizer(1e-4)
# train_step = optimizer.minimize(loss)
# init = tf.initialize_all_variables()


# with tf.Session() as sess:
#     sess.run(init)

#     for it in range(TRAINING_ITERATIONS):
#         sess.run([train_step])

#         # Plot the predictions in `PLOT_AFTER` intervals
#         if it % PLOT_AFTER == 0:
#             loss_value, pred_y, var, target_y, whole_query = sess.run(
#                 [loss, mu, sigma, data_test.target_y, data_test.query]
#             )

#             (context_x, context_y), target_x = whole_query
#             print("Iteration: {}, loss: {}".format(it, loss_value))

#             # Plot the prediction and the context
#             print("result target_x: ", target_x.shape)
#             print("result target_y: ", target_y.shape)
#             print("result context_x: ", context_x.shape)
#             print("result context_y: ", context_y.shape)
#             print("result pred_y: ", pred_y.shape)
#             print("result var: ", var.shape)
#             plot_functions(target_x, target_y, context_x, context_y, pred_y, var, it)
