import os
import gzip
import pickle
import pandas as pd
from numpy import ndarray
from tqdm import tqdm
from scipy.stats import zscore

pd.options.mode.chained_assignment = None  # default='warn'
tqdm.pandas()


def z_score_per_task(df: pd.DataFrame) -> pd.DataFrame:
    """Z-scores the values workers have given using each task as a sample for z-scoring"""
    # create new df with colname to store z_score in
    z_scores_df = pd.DataFrame(columns=['task_response_id', 'z_score_task'])

    # compute z-score per task
    for task_id in df['task_id'].unique():
        task_batch = df[df['task_id'] == task_id]
        task_batch['z_score_task'] = zscore(task_batch['score'])

        # append values to z_score df
        z_scores_df = pd.concat(
            [z_scores_df, task_batch[['task_response_id', 'z_score_task']]]
        )

    # merge values back into the df
    df = df.merge(z_scores_df, on='task_response_id')

    # If all scores are equal, z_score returns Nan. Set those value to 0
    df = df.fillna({'z_score_task': 0})

    return df


def z_score_per_worker(df: pd.DataFrame) -> pd.DataFrame:
    """Z-scores the values workers have given using all the workers scores for z-scoring"""
    # create new df with colname to store z_score in
    z_scores_df = pd.DataFrame(columns=['worker_id', 'z_score_worker'])

    # compute z-score per worker
    for worker_id in df['worker_id'].unique():
        worker_batch = df[df['worker_id'] == worker_id]
        worker_batch['z_score_worker'] = zscore(worker_batch['score'])

        # append values to z_score df
        z_scores_df = pd.concat(
            [z_scores_df, worker_batch[['worker_id', 'z_score_worker']]]
        )

    # merge values back into the df
    df = df.merge(z_scores_df, on='worker_id')

    # If all scores are equal, z_score returns Nan. Set those value to 0
    df = df.fillna({'z_score_worker': 0})

    return df


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


def gather_flattened_dataset(
    path: str = './data/claimant_data/flattened',
) -> pd.DataFrame:
    """Gathers all the flattened dataset batches into a single pandas DataFrame"""

    # load the dataset grouped per task (flattened dataset) and select relevant columns
    df = pd.DataFrame()
    for file in os.listdir(path):
        if file.endswith(".csv"):
            batch = pd.read_csv(os.path.join(path, file))
            batch = batch[
                ['task_id', 'file_id', 'sentence', 'tokens_id', 'publisher', 'source']
            ]
            batch['batch_no'] = int(file.split(' ')[-1].rstrip('.csv').lstrip('b'))
            df = pd.concat([batch, df])

    return df


def get_context(
    file_id: str,
    sentence_id: int,
    pickle_folder: str = './data/vaccination_corpus_pickle/',
) -> str:
    """Uses the file_id to load the pickle file and extract the full context sentence"""

    file_path = f'{pickle_folder}{file_id.rstrip(".annot")}.pickle.gz'

    with gzip.open(file_path, "rb") as inp:
        doc = pickle.load(inp)
        for sentence in doc.sentences:
            if int(sentence.sent_id) == sentence_id:
                return sentence.text

    raise ValueError('Could not find the context sentence in the provided file')


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
