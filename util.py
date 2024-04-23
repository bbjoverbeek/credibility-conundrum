import os
import pandas as pd
from numpy import ndarray
from tqdm import tqdm


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add the numerical value of the scores. 0 will be added when unsure."""

    df["score"] = df["reliability"].map(
        {
            "1 - The source is totally unreliable": 1,
            "2 - The source is somewhat unreliable": 2,
            "3 - The source is potentially reliable": 3,
            "4 - The source is somewhat reliable": 4,
            "5 - The source is fully reliable": 5,
            "I don't know.": 0,
        }
    )

    return df


def gather_dataset(data_path: str) -> pd.DataFrame:
    """Combines the files into one large dataset"""
    df = pd.DataFrame()
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            batch = pd.read_csv(os.path.join(data_path, file))
            batch['batch_no'] = int(file.split(' ')[-1].rstrip('.csv').lstrip('b'))
            df = pd.concat([batch, df])

    df.rename(
        columns={
            "What is your perceived reliability of this source of information?": "reliability"
        },
        inplace=True,
    )

    df = add_scores(df)

    return df


def create_matrix(
    workers: ndarray, tasks: ndarray, judgements: pd.DataFrame
) -> pd.DataFrame:
    """Create a matrix where the rows are the users and the columns are the judgements."""

    matrix = pd.DataFrame(index=list(workers), columns=list(tasks))

    for judgement in tqdm(judgements.itertuples(), total=len(judgements), leave=False):
        if judgement.score is not None:
            matrix.loc[judgement.worker_id, judgement.task_id] = judgement.score

    # fill in the empty values with -1
    matrix = matrix.fillna(-1)

    return matrix
