import keras
from keras import backend as K
import sys
import pickle as pkl
sys.path.insert(0,"..")

from google.apputils import app
import gflags
import h5py as h5

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
gflags.DEFINE_string('model','',"with dropout '' or no dropout 'nd'")

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
    model.load_weights('model.h5')
    predictions = model(x)
    print "Defined TensorFlow model graph."

    # Train an MNIST model
    if FLAGS.model=='nd':
        layers=['activation_1','activation_2','maxpooling2d_1','activation_3','activation_4']
    else:
        layers=['activation_1','activation_2','maxpooling2d_1','dropout_1','activation_3','dropout_2','activation_4']
    
    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    with h5.File('train_adv.h5','r') as f:
        X_train_adv=f['data']
    with h5.File('test_adv.h5','r') as f:
        X_test_adv=f['data']
    # X_train_adv=pkl.load(open('train_adv.pkl','rb'))
    # X_test_adv=pkl.load(open('test_adv.pkl','rb'))
    
    X_test_adv = batch_eval( [model.layers[0].input], map(lambda x:model.get_layer(x).output,layers), [X_test_adv])
    X_train_adv = batch_eval( [model.layers[0].input], map(lambda x:model.get_layer(x).output,layers), [X_train_adv])

    X_test = batch_eval( [model.layers[0].input], map(lambda x:model.get_layer(x).output,layers), [X_test])
    X_train = batch_eval( [model.layers[0].input], map(lambda x:model.get_layer(x).output,layers), [X_train])
    for i,act in enumerate(X_train_adv):
        with h5.File('train_adv_act_'+str(layers[i])+'.h5','w') as f:
            f.create_dataset('data',data=act)
        # pkl.dump(act,open('train_adv_act_'+str(layers[i])+'.pkl','wb'))
    for i,act in enumerate(X_train):
        with h5.File('train_act_'+str(layers[i])+'.h5','w') as f:
            f.create_dataset('data',data=act)
        # pkl.dump(act,open('train_act_'+str(layers[i])+'.pkl','wb'))
    for i,act in enumerate(X_test_adv):
        with h5.File('test_adv_act_'+str(layers[i])+'.h5','w') as f:
            f.create_dataset('data',data=act)
        # pkl.dump(act,open('test_adv_act_'+str(layers[i])+'.pkl','wb'))
    for i,act in enumerate(X_test):
        with h5.File('test_act_'+str(layers[i])+'.h5','w') as f:
            f.create_dataset('data',data=act)
        # pkl.dump(act,open('test_act_'+str(layers[i])+'.pkl','wb'))
    

if __name__ == '__main__':
    app.run()
