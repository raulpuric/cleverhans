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
        y_1 = np.zeros([train_adv.shape[0],2])
        y_1[:,0]=1
        y_2 = np.zeros([train.shape[0],2])
        y_2[:,1]=1
        y=np.concatenate([y_1,y_2],axis=0)
        x=np.concatenate([train_adv,train],axis=0)
        x=x.reshape(reduce_shape(x.shape))
        print x.shape[1]
        clf = LinearSVC(x.shape[1])
        clf.compile(loss='hinge',optimizer=optimizer,metrics=['categorical_accuracy'])
        clf.fit(x,y,batch_size=FLAGS.batch,nb_epoch=FLAGS.epochs)
        print 'train_accuracy: '+str(clf.evaluate(x,y,batch_size=FLAGS.batch))+' for '+layer
        test_adv=pkl.load(open('test_adv_act_'+str(i)+'.pkl','rb'))
        test=pkl.load(open('test_act_'+str(i)+'.pkl','rb'))
        y = np.zeros([test_adv.shape[0],2])
        y[:,0]=1
        x=test_adv.reshape(reduce_shape(test_adv.shape))
         print 'test_accuracy adv: '+str(clf.evaluate(x,y,batch_size=FLAGS.batch))+' for '+layer
        y = np.zeros([test.shape[0],2])
        y[:,1]=1
        x=test.reshape(reduce_shape(test.shape))        
        print 'test_accuracy: '+str(clf.evaluate(x,y,batch_size=FLAGS.batch))+' for '+layer
        clf.save_weights('mnist_svc_'+layer+'.h5')
    

if __name__ == '__main__':
    app.run()
