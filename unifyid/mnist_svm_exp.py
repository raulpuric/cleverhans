import sys
import pickle as pkl
sys.path.insert(0,"..")

from google.apputils import app
import gflags
from sklearn.svm import SVC
import numpy as np

FLAGS = gflags.FLAGS

gflags.DEFINE_string('kernel', 'rbf', 'kernel for svc to use.')
gflags.DEFINE_string('degree', '3', 'degree for svc with polynomial kernel')


def main(argv=None):
    """
    MNIST cleverhans tutorial
    :return:
    """
    
    # Train an MNIST model
    layers=['activation_1','activation_2','maxpooling2d_1','dropout_1','activation_3','dropout_2','activation_4']
    clf = SVC(kernel=FLAGS.kernel,degree=FLAGS.degree)
    for i,layer in enumerate(layers):
        train_adv=pkl.load(open('train_adv_act_'+str(i)+'.pkl','rb'))
        train=pkl.load(open('train_act_'+str(i)+'.pkl','rb'))
        y=np.concatenate([np.zeros(train_adv.shape[0]),np.ones(train.shape[0])],axis=0)
        x=np.concatenate([train_adv,train],axis=0)
        clf.fit(x,y)
        print 'train_accuracy: '+str(clf.score(x,y))+' for '+layer
        test_adv=pkl.load(open('test_adv_act_'+str(i)+'.pkl','rb'))
        test=pkl.load(open('test_act_'+str(i)+'.pkl','rb'))
        y=np.concatenate([np.zeros(test_adv.shape[0]),np.ones(test.shape[0])],axis=0)
        x=np.concatenate([test_adv,test],axis=0)
        print 'test_accuracy: '+str(clf.score(x,y))+' for '+layer
    

if __name__ == '__main__':
    app.run()
