import keras
from keras import backend as K
import sys
import pickle as pkl
sys.path.insert(0,"..")

from google.apputils import app
import gflags

FLAGS = gflags.FLAGS

gflags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
gflags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
gflags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
gflags.DEFINE_integer('batch_size', 128, 'Size of training batches')
gflags.DEFINE_integer('nb_classes', 10, 'Number of classification classes')
gflags.DEFINE_integer('img_rows', 28, 'Input row dimension')
gflags.DEFINE_integer('img_cols', 28, 'Input column dimension')
gflags.DEFINE_integer('nb_filters', 64, 'Number of convolutional filter to use')
gflags.DEFINE_integer('nb_pool', 2, 'Size of pooling area for max pooling')
gflags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')

from cleverhans.utils_mnist_k import data_mnist, model_mnist
# from cleverhans.utils_tf import tf_model_train, tf_model_eval, batch_eval
from cleverhans.utils import model_train, model_eval, batch_eval
from cleverhans.attacks_k import fgsm


def main(argv=None):
    """
    MNIST cleverhans tutorial
    :return:
    """
    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'th':
        keras.backend.set_image_dim_ordering('th')
        print "INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'tf', temporarily setting to 'th'"
    #TODO:Verify all works then remove
    # Create TF session and set as Keras backend session
    #sess = K.get_session()
    #print "Created TensorFlow session and set Keras backend."

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()
    print "Loaded MNIST test data."

    # Define input TF placeholder
    x = K.placeholder(shape=(None, 1, 28, 28), dtype='float32')
    y = K.placeholder(shape=(None, FLAGS.nb_classes), dtype='float32')

    # Define TF model graph
    model = model_mnist()
    predictions = model(x)
    print "Defined TensorFlow model graph."

    # Train an MNIST model
    model_train(model, x, y, predictions, X_train, Y_train)

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    adv_x = fgsm(x, predictions, eps=0.3)
    X_test_adv, = batch_eval( [x], [adv_x], [X_test])
    X_train_adv, = batch_eval( [x], [adv_x], [X_train])
    pkl.dump(X_train_adv,open('train_adv.pkl','wb'))
    pkl.dump(X_test_adv,open('test_adv.pkl','wb')) 
    

if __name__ == '__main__':
    app.run()