""" Elena Pan
    ITP-449
    Project Question 1
    Wine Quality Classification using KNN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def main():
    # load the data from the file 
    file = 'winequality.csv'
    df = pd.read_csv(file)

    # standardize all variables other than Quality
    y = df['Quality']
    x = df.drop(columns = 'Quality')

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

    # partition the dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.6, random_state=42, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, train_size=0.5, random_state=42, stratify=y_temp)

    # iterate on K ranging from 1 to 30
    ks = range(1, 31)
    accuracy_train = []
    accuracy_valid = []
    for k in ks:
        # build a KNN classification model
        model_knn = KNeighborsClassifier(n_neighbors=k)
        model_knn.fit(X_train, y_train)
        # compute the accuracy for both traning and validation for those ks
        accuracy_train.append(model_knn.score(X_train, y_train))
        accuracy_valid.append(model_knn.score(X_valid, y_valid))

    # Plot the accuracy for both the Training and Validation datasets.
    plt.plot(ks, accuracy_train, label='Training accuracy')
    plt.plot(ks, accuracy_valid, label='Validation accuracy')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('KNN: Accuracy for various ks')
    plt.legend()
    plt.tight_layout()
    plt.savefig('KNN.png')

    # k = 5 produces the best accuracy in the Training and Validation datasets

    # generate predictions for the test partition with k = 5
    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(X_train, y_train)
    y_pred = model_knn.predict(X_test)

    # plot the confusion matrix of the actual vs predicted wine quality
    cm = confusion_matrix(y_test, y_pred)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_knn.classes_)
    fig, ax1 = plt.subplots()
    cm_disp.plot(ax=ax1)
    plt.suptitle('Confusion Matrix of Wine Quality')
    plt.savefig('confusion matrix.png')

    # print the accuracy of model
    accuracy = accuracy_score(y_test, y_pred)
    print('The accuracy for the testing dataset is: ', accuracy)

    # print the test dataframe with the added column 'Quality'
    # and 'Predicted Quality'
    X_test['Quality'] = df['Quality']
    X_test['Predicted Quality'] = pd.Series(y_pred).values
    print(X_test)


if __name__ == '__main__':
    main()
