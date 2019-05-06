import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics,preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix  
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    #import matplotlib.pyplot as plt
    #import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('GnBu')

    plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
#----------------------------------------------------------------------------------------    
# Read in data, separate, and scale

Tdata = pd.read_csv("training_dataBAL.csv")
Vdata = pd.read_csv("validation_data.csv")
testData = pd.read_csv("test_data.csv")

xTrain = Tdata.iloc[:,0:25]
yTrain = Tdata.iloc[:,26]
xValid = Vdata.iloc[:,0:25]
yValid = Vdata.iloc[:,26]
xTest = testData.iloc[:,0:25]
yTest = testData.iloc[:,26]

# Get column names first
names = xTrain.columns
# Create the Scaler object
scaler = preprocessing.MinMaxScaler()

# Fit your data on the scaler object
scaled_xTrain = scaler.fit_transform(xTrain)
scaled_xTrain = pd.DataFrame(scaled_xTrain, columns=names)

scaled_xValid = scaler.fit_transform(xValid)
scaled_xValid = pd.DataFrame(scaled_xValid, columns=names)

scaled_xTest = scaler.fit_transform(xTest)
scaled_xTest = pd.DataFrame(scaled_xTest, columns=names)
#----------------------------------------------------------------------------------------
# Use this to choose best number of nearest neighbors
#https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
error = []

for i in range(1, 40): 
    model_i = KNeighborsClassifier(n_neighbors=i)
    modelFit_i = model_i.fit(scaled_xTrain,yTrain)
    yPred_i = model_i.predict(scaled_xValid)
    error.append(np.mean(yPred_i != yValid))
        
plt.figure(figsize=(9, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate vs. K Value, Scaled Features', fontsize=18)  
plt.xlabel('Number of Neighbors', fontsize=18)  
plt.ylabel('Mean Error', fontsize=18)
plt.show()

#----------------------------------------------------------------------------------------
#The Classifier
model = KNeighborsClassifier(n_neighbors=8)
modelFit = model.fit(scaled_xTrain,yTrain)
yPredT = model.predict(scaled_xTrain)
yPredV = model.predict(scaled_xValid)
yPredTest = model.predict(scaled_xTest)

print("Training Accuracy:",metrics.accuracy_score(yTrain, yPredT))
print("Validation Accuracy:",metrics.accuracy_score(yValid, yPredV))
print("Test Accuracy:",metrics.accuracy_score(yTest, yPredTest))
conMatT = confusion_matrix(yTrain, yPredT)
plot_confusion_matrix(conMatT, 
                      normalize    = False,
                      target_names = ['Non-Prog', 'Prog'],
                      title        = "Training Set")

conMatV = confusion_matrix(yValid, yPredV)
plot_confusion_matrix(conMatV, 
                      normalize    = False,
                      target_names = ['Non-Prog', 'Prog'],
                      title        = "Validation Set")

conMatTest = confusion_matrix(yTest, yPredTest)
plot_confusion_matrix(conMatTest, 
                      normalize    = False,
                      target_names = ['Non-Prog', 'Prog'],
                      title        = "Test Set")