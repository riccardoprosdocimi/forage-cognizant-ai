import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


# Define global constants
K = 10  # K is used to define the number of folds that will be used for cross-validation
SPLIT = 0.8  # Split defines the % of data that will be used as training data, the rest is used for testing


# Load data
def load_data(path: str = None) -> pd.DataFrame:
    """
    Takes a path string to a CSV file and loads it into a Pandas DataFrame.

    :param path: relative path of the CSV file
    :return: a Pandas DataFrame
    """

    df = pd.read_csv(path)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df


# Create label and features
def create_label_and_features(
        data: pd.DataFrame = None,
        label: str = "estimated_stock_pct"
) -> (pd.DataFrame, pd.Series):
    """
    Takes in a Pandas DataFrame and splits the columns into a target column and a set of predictor variables,
    i.e. X & y. These two splits of the data will be used to train a supervised machine learning model.

    :param data: DataFrame containing data for the model
    :param label: target variable that you want to predict (optional)
    :return: a tuple containing the features and the label
    """

    # Check to see if the target variable is present in the data
    if label not in data.columns:
        raise Exception(f"Label: {label} is not present in the data")

    X = data.drop(columns=[label])
    y = data[label]
    return X, y


# Train model
def train_model_with_cross_validation(
    X: pd.DataFrame = None,
    y: pd.Series = None
) -> None:
    """
    This function takes the features and label and trains a Random Forest Regressor model across K folds. Using
    cross-validation, performance metrics will be output for each fold during training.

    :param X: features
    :param y: label
    """

    # Create a list that will store the accuracies of each fold
    accuracies = []

    # Enter a loop to run K folds of cross-validation
    for fold in range(0, K):

        # Instantiate algorithm and scaler
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Create training and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)

        # Scale X data
        # We scale the data because it helps the algorithm to converge and helps the algorithm to not be greedy with
        # large values
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on test data
        y_pred = trained_model.predict(X_test)

        # Compute accuracy using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracies.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    # Finish by computing the average MAE across all folds
    print(f"Average MAE: {(sum(accuracies) / len(accuracies)):.2f}")


# Execute training pipeline
def run(path: str) -> None:
    """
    Executes the training pipeline of loading the prepared dataset from a CSV file and training the machine learning
    model.

    :param path: relative path of the CSV file
    """

    # Load data
    df = load_data(path)

    # Create label and features
    X, y = create_label_and_features(df)

    # Train model
    train_model_with_cross_validation(X, y)
