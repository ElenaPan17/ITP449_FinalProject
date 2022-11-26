""" Elena Pan
    ITP-449
    Project Question 4
    Vehicle MPGs Using Linear Regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main():
    # summarize the dataset
    file = 'auto-mpg.csv'
    df = pd.read_csv(file)
    # print(df.describe())
    
    # the mean of mpg is 23.514573
    # the median value of mpg is 23.0
    # the mean is higher than the median, so it is a right-skewed distribution

    # plot the mean and median distribution --> prove right skewed
    plt.hist(df['mpg'])
    plt.title('Histogram Distribution of MPG')
    plt.xlabel('MPG')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('Histogram.png')

    # plot the scatterplot matrix of all the relevant numeric attributes
    # mpg, cylinders, displacement, horespower, weight, acceleration, model_year
    # print(df.info())
    df_scatter = df.drop(columns = ['car_name', 'No'])
    columns = df_scatter.columns
    fig, ax = plt.subplots(len(columns), len(columns))
    for i in range(len(columns)):
        for j in range(len(columns)):
            if i != j:
                ax[i, j].scatter(df_scatter.iloc[:, i], df_scatter.iloc[:, j], s = 2)
                ax[i, j].set_xticklabels('')
                ax[i, j].set_yticklabels('')
                if j == 0:
                    ax[i, j].set_ylabel(columns[i], fontsize = 8, rotation = 0)
                if i == len(columns)- 1:
                    ax[i, j].set_xlabel(columns[j], fontsize = 8)
            else:
                ax[i, j].hist(df_scatter.iloc[:, i])
                ax[i, j].set_xticklabels('')
                ax[i, j].set_yticklabels('')
                if j == 0:
                    ax[i, j].set_ylabel(columns[i], fontsize = 8, rotation = 0)
                if i == len(columns)- 1:
                    ax[i, j].set_xlabel(columns[j], fontsize = 8)
    
    plt.suptitle('Scatterplot Matrix of All Relevant Numeric Attributes')
    plt.tight_layout()
    plt.savefig('Scatterplot_matrix.png')

    # correlation matrix 
    corr_matrix = df_scatter.corr().abs()
    fig, ax1 = plt.subplots(1, 1)
    corr_matrix_plot = ax1.matshow(corr_matrix)
    plt.colorbar(corr_matrix_plot, ax = ax1)
    for (i, j), z in np.ndenumerate(corr_matrix):
        ax1.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    ax1.set_xticks(range(len(corr_matrix.columns)))
    ax1.set_yticks(range(len(corr_matrix.columns)))
    ax1.set_xticklabels(corr_matrix.columns, rotation=45)
    ax1.set_yticklabels(corr_matrix.columns)
    ax1.set(title='Correlation Matrix')
    plt.tight_layout()
    plt.savefig('corr_matrix.png')

    # based on scatterplot and correlation matrix
    # displacement and cylinders seem to be most strongly lineraly correlated (0.95)
    # model_year and acceleration seem to be most weakly lineraly correlated (0.29)

    # scatter plot of mpg vs displacement
    plt.figure()
    plt.scatter(df['displacement'], df['mpg'])
    plt.title('Scatterplot of mpg v.s. displacement')
    plt.xlabel('Displacement')
    plt.ylabel('MPG')
    plt.tight_layout()
    plt.savefig('Scatterplot.png')

    # build a linear regression model 
    x = df['displacement']
    y = df['mpg']
    
    X = x.values.reshape(-1, 1)
    model_linreg = LinearRegression(fit_intercept=True) 
    model_linreg.fit(X, y)
    # intercept
    print('The value of the intercept is: ', model_linreg.intercept_) 
    # coefficient
    print('The value of the coefficient is: ', model_linreg.coef_[0]) 
    # regression equation
    regression = 'mpg = %.2f * displacement + %.2f' %(model_linreg.coef_, model_linreg.intercept_)
    print('The regression equation is: ', regression)
    
    # According to the regression equation
    # as the displacement increases, predicted value for mpg decreases
    # because the coefficient is negative

    # predict mpg when displacement = 220
    y_predict = model_linreg.predict([[220]])
    print('The predict mpg is ', y_predict[0], ' when displacement is 220')

    # scatterplot of actual mpg vs displacement + linear regression line 
    y_pred = model_linreg.predict(X)

    plt.scatter(X, y, label='Scatterplot of Actual MPG vs Displacement', color = 'blue')
    plt.plot(X, y_pred, color='r', label=regression)
    plt.legend()
    plt.savefig('linreg.png')

    # plot the residuals
    residuals = y - y_pred
    SS_Residual = sum((y-y_pred)**2)       
    SS_Total = sum((y-np.mean(y))**2)     
    r_squared = 1 - (float(SS_Residual))/SS_Total

    fig, ax = plt.subplots(1, 2)
    r_label = 'Rsquared = %.3f' %r_squared
    ax[0].scatter(y_pred, residuals, alpha=0.8, label = r_label)
    ax[0].plot([y_pred.min(), y_pred.max()], [0, 0], color='red')
    ax[0].set(xlabel='Prediction', ylabel='Residuals', title='Residuals Plot')
    ax[0].legend()

    ax[1].hist(residuals, orientation='horizontal', bins=25)
    ax[1].set(xlabel='Distribution', title='Histogram of Residuals')
    ax[1].yaxis.tick_right()
    
    plt.suptitle('Residuals Plot')
    plt.tight_layout()
    plt.savefig('residuals.png')



if __name__ == '__main__':
    main()
