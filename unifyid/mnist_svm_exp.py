import sys
import pickle as pkl
sys.path.insert(0,"..")

from google.apputils import app
import gflags
import numpy as np
import keras
from keras import backend as K
from keras.optimizers import Adadelta
from keras.objectives import hinge
from keras.metrics import binary_accuracy
from mnist_act_discriminators import LinearSVC


FLAGS = gflags.FLAGS

gflags.DEFINE_integer('batch', 128, 'batch_size')
gflags.DEFINE_integer('epochs', 5, 'num_epochs')
gflags.DEFINE_float('learning_rate',.01,'learning_rate')


def main(argv=None):
    """
    MNIST cleverhans tutorial
    :return:
    """
    
    # Train an MNIST model
    layers=['activation_1','activation_2','maxpooling2d_1','dropout_1','activation_3','dropout_2','activation_4']
    optimizer = Adadelta(lr=FLAGS.learning_rate, rho=0.95, epsilon=1e-08)

    def reduce_shape(shape):
        return shape[0],reduce(lambda x,y:x*y,shape[1:])

    for i,layer in enumerate(layers):

        train_adv=pkl.load(open('train_adv_act_'+str(i)+'.pkl','rb'))
        train=pkl.load(open('train_act_'+str(i)+'.pkl','rb'))
        y=np.concatenate([np.zeros(train_adv.shape[0]),np.ones(train.shape[0])],axis=0)
        x=np.concatenate([train_adv,train],axis=0)
        x=x.reshape(reduce_shape(x.shape))
        clf = LinearSVC(x.shape[1:])
        clf.compile(loss='hinge',optimizer=optimizer,metrics=['binary_accuracy'])
        clf.fit(x,y,batch_size=FLAGS.batch,nb_epochs=FLAGS.epochs)
        print 'train_accuracy: '+str(clf.evaluate(x,y,batch_size=FLAGS.batch))+' for '+layer
        test_adv=pkl.load(open('test_adv_act_'+str(i)+'.pkl','rb'))
        test=pkl.load(open('test_act_'+str(i)+'.pkl','rb'))
        y=np.concatenate([np.zeros(test_adv.shape[0]),np.ones(test.shape[0])],axis=0)
        x=np.concatenate([test_adv,test],axis=0)
        x=x.reshape(reduce_shape(x.shape))
        print 'test_accuracy: '+str(clf.evaluate(x,y,batch_size=FLAGS.batch))+' for '+layer
        clf.save_weights('mnist_svc_'+layer+'.h5')
    

if __name__ == '__main__':
    app.run()
