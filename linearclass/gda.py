import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import util

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
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

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        
        # Obtain the shape of the regressors array
        rows, column = x.shape
        # Obtain the complementary vector to y
        id_n = np.array([1 if y_ == 0 else 0 for y_ in y])
        # Estimate phi using the closed formula
        phi = (1 / rows) * np.sum(y)
        # Estimate mu_0 using the closed formula, the vector id_n is constructed as the complementary vector to y
        mu_0 = np.sum(id_n * x.T, axis = 1)/ np.sum(id_n)
        
        # Since y = id_p, we don't need to construct a vector id_p
        mu_1 = np.sum(y * x.T, axis = 1)/ np.sum(y)
        # In order to compute sigma we need a vector that contains mu_0 whenever y_i = 0 and mu_1 viceversa
        mu_y = np.array([mu_0 if elem == 0 else mu_1 for elem in y])
        
        # Estimate sigma with the closed formula and the vector mu_y created above
        sigma = 1 / rows * np.dot(np.transpose(x - mu_y), x - mu_y)
        # We now obtain the inverse
        sigma_inversa = np.linalg.inv(sigma)
        # Create the expression t_zero needed for the logistic form, obtained from the math
        t_zero = np.dot(sigma_inversa, (mu_1 - mu_0))
        # Create the expression t_1 needed for the logistic form, obtained from the math
        t_uno = - 1/2 * (np.transpose(mu_1) @ sigma_inversa @ mu_1 - np.transpose(mu_0) @ sigma_inversa @ mu_0) - np.log((1-phi)/phi)
        # Update parameter theta
        self.theta = np.concatenate(([t_uno], t_zero))
        

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Here we use the logistic form of GDA to make predictions with the obtained thetas
        return 1 / (1 + np.exp(-(np.dot(self.theta[1:], x.T) + self.theta[0])))
        # *** END CODE HERE

    


def compute_metrics(y_eval, y_pred):
    # Compute accuracy 
    accuracy = (y_eval == y_pred).sum() / y_eval.shape[0]
    TP = ((y_pred == 1) & (y_eval == 1)).sum()
    FP = ((y_pred == 1) & (y_eval == 0)).sum()
    # Compute precision
    precision = TP / (TP+FP)
    print(f'the accuracy using GDA was: {round(accuracy, 3)}')
    print(f'the precision using GDA was: {round(precision,3)}')

##### HERE I TRIED SEVERAL TRANSFORMS TO FIND ONE THAT LEADS GDA TO OUTPERFORM LOGISTIC IN THIS DATASET

def sigmoid_map(arr, scale=1, shift=0):
    return 1 / (1 + np.exp(-scale * (arr - shift)))


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    # x_train = sigmoid_map(x_train, 0.1, 0)
   
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)

    # x_eval = sigmoid_map(x_eval, 0.1, 0)
    


    # x_eval = esscher_transform(x_train, beta)
    # print(f'x_train: {x_train.shape}')
    # print(f'y_train: {y_train.shape}')
    # print(f'x_eval: {x_eval.shape}')
    # print(f'y_eval: {y_eval.shape}')
    # *** START CODE HERE ***
    # Train a GDA classifier
    clf = GDA()
    # Fit the object
    clf.fit(x_train, y_train)
    # Compute probabilities
    probabilities = clf.predict(x_eval)
    # Obtain predictions with the cutoff rule
    predictions = np.where(probabilities >= 0.5, 1, 0)
    
    # Compute metrics needed for comparison
    compute_metrics(y_eval, predictions)
    # Plot decision boundary on validation set
    util.plot(x_eval, y_eval, clf.theta, save_path.split('.')[0] + '.png')
    # Use np.savetxt to save outputs from validation set to save_path
    np.savetxt(save_path, predictions)
    # *** END CODE HERE ***
    

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
