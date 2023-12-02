import matplotlib.pyplot as plt
import numpy as np
import util

from lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem: Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    size = len(tau_values)
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    # *** START CODE HERE ***
    MSEs = np.zeros(size)
    for i in range(size):
        clf = LocallyWeightedLinearRegression(tau_values[i])
    # Fit a LWR model
        clf.fit(x_train, y_train)
    # Get predictions on the validation set
        y_pred = clf.predict(x_eval)
        clf.plot_data(x_eval[:,1], y_pred, y_eval)
    # Search tau_values for the best tau (lowest MSE on the validation set)
        MSE = clf.compute_mse(y_pred, y_eval) 
        MSEs[i] = MSE
    
    # print(MSEs)
    index = np.argmin(MSEs)
    print(f'The best value for tau obtained in the validation set is {tau_values[index]} with a MSE of {MSEs[index]}')

    # Fit a LWR model with the best tau value
    final_clf = LocallyWeightedLinearRegression(tau_values[index])
    final_clf.fit(x_train, y_train)
    # Run on the test set to get the MSE value
    y_pred = final_clf.predict(x_test)
    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred)
    # Plot data
    x_train = x_train[:,1]
    x_test = x_test[:,1]
    clf.plot_data(x_test, y_pred, y_test)
    MSE = clf.compute_mse(y_pred, y_test)
    print(f'the final MSE computed on the test set is: {MSE}')
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')
