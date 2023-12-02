import matplotlib.pyplot as plt
import numpy as np
import util



class LocallyWeightedLinearRegression():
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None
        self.theta = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.
        """
        
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.
        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, _= x.shape
        y_pred = np.zeros(m)
        for i in range(m):
            # Calculate weight based on tau parameter and the difference between x and the i-th element of x
            pesi_array = np.exp(-np.sum((self.x - x[i]) ** 2, axis=1) / (2 * (self.tau ** 2)))

            # Generate a diagonal matrix using the weight array
            diag_weight_matrix = np.diag(pesi_array)

            # Estimate theta using the normal equation and the calculated weights
            theta_fit = np.linalg.inv(self.x.T.dot(diag_weight_matrix).dot(self.x)).dot(self.x.T).dot(diag_weight_matrix).dot(self.y)

            y_pred[i] = x[i] @ theta_fit
        return y_pred
    
    def compute_mse(self, y_pred, y):
        MSE = np.mean((y_pred - y)**2)
        return MSE
    
    def plot_data(self, x_, y_pred, y_):
        # Sort the values for plotting
        sorted_indices = np.argsort(x_)
        x_ = x_[sorted_indices]
        y_pred = y_pred[sorted_indices]
        y_ = y_[sorted_indices]
        
        plt.figure()
        plt.scatter(x_, y_, marker = 'x', color ='blue', s=20)
        plt.scatter(x_, y_pred, marker = 'o', color ='red', s=20)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Locally Weighted Linear Regression')
        plt.legend(['Data', 'Predictions'])
        plt.show()

        

def main(tau, train_path, eval_path):
    """Problem: Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
     # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    clf = LocallyWeightedLinearRegression(tau)
    # Fit a LWR model
    clf.fit(x_train, y_train)
    # Get predictions on the validation set
    y_pred = clf.predict(x_eval)

    MSE = clf.compute_mse(y_pred, y_eval)
    print(f'MSE for tau = {tau} is: {MSE}')

    x_train = x_train[:,1]
    x_eval = x_eval[:, 1]
    # Plot validation predictions on top of training set
    clf.plot_data(x_eval, y_pred, y_eval)
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau=5e-1,
         train_path='./train.csv',
         eval_path='./valid.csv')
