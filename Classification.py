""" Elena Pan
    ITP-449
    Project Question 3
    Mushroom Edibility Using Trees
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def main():
    # load the data into a dataframe 
    file = 'mushrooms.csv'
    df = pd.read_csv(file)
    
    # partition the dataset 
    y = df['class']
    X = df.drop(columns = 'class')
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.7, stratify=y)

    model_tree = DecisionTreeClassifier(criterion='entropy', max_depth = 6, random_state=42)
    model_tree.fit(X_train, y_train)

    # print and visualilze the confusion matrix for training and testing
    y_pred_train =model_tree.predict(X_train)
    y_pred_test =model_tree.predict(X_test)

    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)
    print('confusion matrix for training: ', '\n', cm_train)
    print('confusion matrix for testing: ', '\n', cm_test)

    cm_disp_train = ConfusionMatrixDisplay(cm_train, display_labels=model_tree.classes_)
    cm_disp_test = ConfusionMatrixDisplay(cm_test, display_labels=model_tree.classes_)
    
    fig, ax = plt.subplots(1, 2, figsize = (15, 8))
    cm_disp_train.plot(ax = ax[0])
    ax[0].set(title = 'Training Confusion Matrix')
    cm_disp_test.plot(ax = ax[1])
    ax[1].set(title = 'Testing Confusion Matrix')

    plt.suptitle('Mushroom Edibility Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion matrix.png')

    # training partition accuracy
    print('accuracy on the training partition: ', model_tree.score(X_train, y_train))

    # test partition accuracy
    print('accuracy on the test partition: ', model_tree.score(X_test, y_test))

    # build a classification tree 
    # plt.figure(figsize = (50,20))
    plt.figure()
    plot_tree(model_tree, fontsize = 5, feature_names=X.columns, class_names=y.unique(), filled=True)
    plt.suptitle('Mushroom Edibility Tree', fontsize = 6)
    plt.tight_layout()
    plt.savefig('tree.png')

    # top three most important features for determining toxicity
    feat_import = pd.Series(model_tree.feature_importances_, index = X.columns)
    print(feat_import.nlargest(3))
    
    # classify the following mushroom:
    data = {'cap-shape': ['x'], 'cap-surface': ['s'], 'cap-color': ['n'], 'bruises': ['t'], 'odor': ['y'],
        'gill-attachment': ['f'], 'gill-spacing': ['c'], 'gill-size': ['n'], 'gill-color': ['k'], 
        'stalk-shape': ['e'], 'stalk-root': ['e'], 'stalk-surface-above-ring': ['s'], 'stalk-surface-below-ring': ['s'],
        'stalk-color-above-ring': ['w'], 'stalk-color-below-ring': ['w'], 'veil-type': ['p'], 'veil-color': ['w'],
        'ring-number': ['o'], 'ring-type': ['p'], 'spore-print-color': ['r'], 'population': ['s'], 'habitat': ['u']}
    
    x_new = df.drop(columns = 'class')
    df_new = pd.concat([pd.DataFrame(data), x_new], ignore_index = True)
    
    df_new = pd.get_dummies(df_new)
    y_pred_data = model_tree.predict(df_new.head(1))
    print('The result for this mushroom is: ', y_pred_data[0])
    






if __name__ == '__main__':
    main()
