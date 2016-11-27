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
from mnist_act_discriminators import simple_softmax

from generate_results import *

FLAGS = gflags.FLAGS

gflags.DEFINE_integer('batch', 128, 'batch_size')
gflags.DEFINE_integer('epochs', 15, 'num_epochs')
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
        y_1 = np.zeros([train_adv.shape[0],2])
        y_1[:,0]=1
        y_2 = np.zeros([train.shape[0],2])
        y_2[:,1]=1
        y=np.concatenate([y_1,y_2],axis=0)
        x=np.concatenate([train_adv,train],axis=0)
        x=x.reshape(reduce_shape(x.shape))
        print x.shape[1]
        clf = simple_softmax(x.shape[1])
        clf.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        clf.fit(x,y,batch_size=FLAGS.batch,nb_epoch=FLAGS.epochs)
        print 'train_accuracy: '+str(clf.evaluate(x,y,batch_size=FLAGS.batch))+' for '+layer
        predicts=clf.predict(x)
        generate_roc(y,predicts, 'simple_cross_entropy_'+str(layer)+'.png')
        test_adv=pkl.load(open('test_adv_act_'+str(i)+'.pkl','rb'))
        test=pkl.load(open('test_act_'+str(i)+'.pkl','rb'))
        y_1 = np.zeros([test_adv.shape[0],2])
        y_1[:,0]=1
        y_2 = np.zeros([test.shape[0],2])
        y_2[:,1]=1
        x=np.concatenate([test_adv,test],axis=0)
        x=x.reshape(reduce_shape(x.shape))
        y=np.concatenate([y_1,y_2],axis=0)

        print 'test_accuracy: '+str(clf.evaluate(x,y,batch_size=FLAGS.batch))+' for '+layer
        clf.save_weights('mnist_crossentropy_'+layer+'.h5')
    

if __name__ == '__main__':
    app.run()
