import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
def generate_roc(y_test, y_score,name=None):
    y_test=y_test[:,0]
    y_score=y_score[:,0]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    if name is not None:
        plt.savefig(name)
    plt.show()
    print('AUC: %f' % roc_auc)