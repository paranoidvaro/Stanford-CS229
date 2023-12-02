import numpy as np
import matplotlib.pyplot as plt
import util


class LogisticRegression:
    ###Logistic regression with Newton's Method as the solver.

    def __init__(self, step_size=1, max_iter=1000000, eps=1e-5, theta_0=None, verbose=False):
        """
        Initializes the logistic regression model.

        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    @staticmethod
    def sigmoid(z):
        # Computes the sigmoid function.
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y):
        # Initializing theta with random values 
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])
        # Newton's Optimization 
        for i in range(self.max_iter):
            # Z = X θ
            Z = np.dot(x, self.theta)
            # h_theta = g(Z) = 1 / 1 + e^(-Z)
            h_theta = self.sigmoid(Z)
            # ∇J(θ) = 1/m XT (h_theta - y)
            gradient = np.dot(x.T, (h_theta - y)) / y.size
            # H = 1/m XT diag(h (1 - h)) X
            hessian = np.dot(x.T, np.dot(np.diag(h_theta * (1 - h_theta)), x)) / y.size
            # Δθ = H^(−1) ⋅∇J(θ)
            delta = np.dot(np.linalg.inv(hessian), gradient)

            if np.linalg.norm(delta) < self.eps:
                break
            # Updating parameters
            self.theta -= delta
            # J(θ) = 1/2m (Xθ - y)T (Xθ - y) 
            loss = np.dot((-y).T, np.log(h_theta + self.eps)) - np.dot((1-y).T, np.log((1 - h_theta) + self.eps))/ (y.size)
            if self.verbose == 1:
                print(f'iter {i} loss: {loss}')



    def predict(self, x):
        # Predicts probability of positive class for a given input data.
        probabilities = 1 / (1 + np.exp(-x.dot(self.theta)))
        return probabilities



def compute_metrics(y_eval, y_pred):
    # Computes accuracy on the validation set.
    accuracy = (y_eval == y_pred).sum() / y_eval.shape[0]
    TP = ((y_pred == 1) & (y_eval == 1)).sum()
    FP = ((y_pred == 1) & (y_eval == 0)).sum()
    # Computes precision on the validation set.
    precision = TP / (TP+FP)
    print(f'the accuracy using Logistic Regression was: {round(accuracy, 3)}')
    print(f'the precision using Logistic Regression was: {round(precision, 3)}')



def main(train_path, valid_path, save_path):
    """Main function to execute the logistic regression model."""
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    # Create an object of the class defined above.
    clf = LogisticRegression()
    # Fit the object with training data.
    clf.fit(x_train, y_train)
    # Predict probabilities on the validation set.
    probabilities = clf.predict(x_eval)
    # Transform probabilities in prediction with the cutoff rule of 0.5
    predictions = np.where(probabilities >= 0.5, 1 ,0)
    # Compute metrics needed to compare Logistic and GDA
    compute_metrics(y_eval, predictions)
    np.savetxt(save_path, probabilities)
    util.plot(x_eval, y_eval, clf.theta, save_path.split('.')[0] +'.png')

    


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')

