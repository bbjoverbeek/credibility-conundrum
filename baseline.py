import pandas as pd
from scipy.stats import zscore
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from util import gather_dataset


def create_splits(
    df: pd.DataFrame, train: int = 70, dev: int = 10, test: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into a train, dev, and test split using the specified percentages."""

    if (train + dev + test) != 100:
        raise ValueError('The train, dev, and test splits must sum to 100.')

    train_split, temp_split = train_test_split(
        df, test_size=(dev + test) / 100, random_state=42
    )

    dev_split, test_split = train_test_split(
        temp_split, test_size=test / (dev + test), random_state=42
    )

    return train_split, dev_split, test_split


def main() -> None:
    """Run a baseline model predicting the reliability of a claimant."""

    # prep dataset
    df = gather_dataset('./data/claimant_data/')
    df['z_score'] = zscore(df['score'])

    train_split, _, test_split = create_splits(df)

    # fit the baseline model
    dummy_regressor = DummyRegressor(strategy='mean')
    dummy_regressor.fit(train_split['source'], train_split['z_score'])

    # predict on the test set
    predictions = dummy_regressor.predict(test_split['source'])

    # get MSE
    mse = mean_squared_error(test_split['z_score'], predictions)

    print(f'Dummy regressor MSE: {mse:.3f}')


if __name__ == '__main__':
    main()
