from keras import backend as K
from keras.optimizers import Adadelta

def k_model_loss(y, model, mean=True):
    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """
    if mean:
        # Return mean of the loss
        return K.mean(K.categorical_crossentropy(y, model))
    else:
        # Return a vector with the loss per sample
        return K.categorical_crossentropy(y, model)

def model_train(model, x, y, predictions, X_train, Y_train, save=False,
                   predictions_adv=None):
    """
    Train a TF graph
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_train: numpy array with training inputs
    :param Y_train: numpy array with training outputs
    :param save: Boolean controling the save operation
    :param predictions_adv: if set with the adversarial example tensor,
                            will run adversarial training
    :return: True if model trained
    """
    print "Starting model training using TensorFlow."

    # Define loss
    loss = k_model_loss(y, predictions)
    if predictions_adv is not None:
        loss = (loss + k_model_loss(y, predictions_adv)) / 2

    optimizer = AdadeltaOptimizer(lr=FLAGS.learning_rate, rho=0.95, epsilon=1e-08)
    model.compile(optimizer = optimizer,loss = loss)
    # train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)
    print "Defined optimizer."

    # with sess.as_default():

    for epoch in xrange(FLAGS.nb_epochs):
        print("Epoch " + str(epoch))

        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_train)) / FLAGS.batch_size))
        assert nb_batches * FLAGS.batch_size >= len(X_train)

        prev = time.time()
        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))
                cur = time.time()
                print("\tTook " + str(cur - prev) + " seconds")
                prev = cur

            # Compute batch start and end indices
            start, end = batch_indices(batch, len(X_train), FLAGS.batch_size)

            # Perform one training step
            model.train_on_batch(X_train[start:end],
                                      Y_train[start:end])
        assert end >= len(X_train) # Check that all examples were used


    if save:
        # save_path = os.path.join(FLAGS.train_dir, FLAGS.filename)
        # saver = tf.train.Saver()
        # saver.save(sess, save_path)
        print "Completed model training and model saved at:" + str(save_path)
    else:
        print "Completed model training."

    return True
def model_eval(x, y, model, X_test, Y_test):
    """
    Compute the accuracy of a TF model on some data
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param model: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :return: a float with the accuracy value
    """
    # Define sympbolic for accuracy
    acc_value = keras.metrics.categorical_accuracy(y, model)

    # Init result var
    accuracy = 0.0

    # Compute number of batches
    nb_batches = int(math.ceil(float(len(X_test)) / FLAGS.batch_size))
    assert nb_batches * FLAGS.batch_size >= len(X_test)

    for batch in range(nb_batches):
        if batch % 100 == 0 and batch > 0:
            print("Batch " + str(batch))

        # Must not use the `batch_indices` function here, because it
        # repeats some examples.
        # It's acceptable to repeat during training, but not eval.
        start = batch * FLAGS.batch_size
        end = min(len(X_test), start + FLAGS.batch_size)
        cur_batch_size = end - start + 1

        # The last batch may be smaller than all others, so we need to
        # account for variable batch size here
        accuracy += cur_batch_size * acc_value.eval(feed_dict={x: X_test[start:end],
                                        y: Y_test[start:end],
                                        keras.backend.learning_phase(): 0})
    assert end >= len(X_test)

    # Divide by number of examples to get final value
    accuracy /= len(X_test)

    return accuracy

def batch_eval(tf_inputs, tf_outputs, numpy_inputs):
    """
    A helper function that computes a tensor on numpy inputs by batches.
    """
    n = len(numpy_inputs)
    assert n > 0
    assert n == len(tf_inputs)
    m = numpy_inputs[0].shape[0]
    for i in xrange(1, n):
        assert numpy_inputs[i].shape[0] == m
    out = []
    for _ in tf_outputs:
        out.append([])

    for start in xrange(0, m, FLAGS.batch_size):
        batch = start // FLAGS.batch_size
        if batch % 100 == 0 and batch > 0:
            print("Batch " + str(batch))

        # Compute batch start and end indices
        start = batch * FLAGS.batch_size
        end = start + FLAGS.batch_size
        numpy_input_batches = [numpy_input[start:end] for numpy_input in numpy_inputs]
        cur_batch_size = numpy_input_batches[0].shape[0]
        assert cur_batch_size <= FLAGS.batch_size
        for e in numpy_input_batches:
            assert e.shape[0] == cur_batch_size

        feed_dict = dict(zip(tf_inputs, numpy_input_batches))
        feed_dict[keras.backend.learning_phase()] = 0
        numpy_output_batches = tf_outputs.eval(feed_dict=feed_dict)
        for e in numpy_output_batches:
            assert e.shape[0] == cur_batch_size, e.shape
        for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
            out_elem.append(numpy_output_batch)

    out = map(lambda x: np.concatenate(x, axis=0), out)
    for e in out:
        assert e.shape[0] == m, e.shape
    return out