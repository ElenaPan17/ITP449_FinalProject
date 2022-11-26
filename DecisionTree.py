""" Elena Pan
    ITP-449
    Project Question 2
    Personal Loan Prediction Using Trees
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def main():
    # load the data into a dataframe 
    file = 'UniversalBank.csv'
    df = pd.read_csv(file)

    # target variable is personal loan
    y = df['Personal Loan']

    # remove attributes Row and Zip Code 
    df_new = df.drop(columns = ['Row', 'ZIP Code'])
    
    # partition the dataset
    X = df_new.drop(columns = 'Personal Loan')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.7, stratify=y)

    # count the number of cases in the training who accepted offers of a personal loan
    print('In the training partition, there are ', sum(y_train), ' people who accepted offers of a personal loan.')
    
    # plot the classification tree
    model_tree = DecisionTreeClassifier(criterion='entropy', max_depth = 5, random_state=42)
    model_tree.fit(X_train, y_train)

    plt.figure(figsize = (70,30))
    plot_tree(model_tree, fontsize = 45, feature_names=X.columns, filled = True)
    plt.suptitle('Personal Loan Tree', fontsize = 60)
    plt.tight_layout()
    plt.savefig('tree.png')

    # plot the confusion matrix to calculate the miss prediction
    y_p = model_tree.predict(X_train)

    cm = confusion_matrix(y_train, y_p)
    cm_disp = ConfusionMatrixDisplay(cm, display_labels=y.unique())
    cm_disp.plot()

    plt.title('Loan Decision Tree')
    plt.tight_layout()
    plt.savefig('confusion matrix.png')

    # According to the confusion matrix 
    # On the training partition 
    # there are 30 acceptors that the model classified as non-acceptors
    # there are 21 non-acceptors that the model classfied as acceptors

    # accuracy on the training partition 
    print('accuracy on the training partition: ', model_tree.score(X_train, y_train))

    # accuracy on the test partition 
    print('accuracy on the test partition: ', model_tree.score(X_test, y_test))


if __name__ == '__main__':
    main()
