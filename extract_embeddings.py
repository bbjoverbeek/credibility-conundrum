"""
Uses an encoder model to get word embedding representations of claimants
"""

import os
from transformers import (
    pipeline,
    Pipeline,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from datasets import Dataset
from datasets.formatting.formatting import LazyRow
from numpy import mean as np_mean
from util import gather_flattened_dataset, get_context

from tqdm import tqdm

tqdm.pandas()


def extract_embeddings(row: LazyRow, pipe: Pipeline, colname='embeddings') -> LazyRow:
    """Extracts word embeddings for a claimant"""

    embeddings = pipe(row['source'])
    cls_ = embeddings[0][0]  # get first embedding of the span # type: ignore
    row[colname] = cls_

    return row


def extract_embeddings_context(
    row: LazyRow,
    pipe: Pipeline,
    tokenizer=PreTrainedTokenizer,
    colname='context_embeddings',
) -> LazyRow:
    """Extracts word embeddings for a claimant, given a context"""

    context_embeddings = pipe(row['context'], truncation=True)[0]  # type:ignore
    tokens_embedding_indexes = tokenizer.encode_plus(row['context']).word_ids()  # type: ignore

    claimant_token_indexes = set(
        int(idx) for idx in row['tokens_id'].split(' ')  # type:ignore
    )

    # extract all embeddings of the claimant tokens
    claimant_embeddings = []
    for token_index, embedding in zip(tokens_embedding_indexes, context_embeddings):
        if token_index in claimant_token_indexes:
            claimant_embeddings.append(embedding)

    # take average of all embeddings
    row[colname] = list(np_mean(claimant_embeddings, axis=0))

    return row


def main() -> None:

    df = gather_flattened_dataset(path='./data/claimant_data/flattened')
    df['context'] = df.progress_apply(
        lambda row: get_context(row['file_id'], row['sentence']), axis=1
    )  # type: ignore

    ds = Dataset.from_pandas(df)

    os.environ['HF_HUB_CACHE'] = '/scratch/s4415213/'
    model_name = 'microsoft/deberta-v3-large'

    # initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline(
        'feature-extraction', model=model_name, use_fast=True, device='cuda'
    )

    # extract embeddings without context
    ds = ds.map(lambda row: extract_embeddings(row, pipe))

    # extrat embeddings with context
    ds = ds.map(
        lambda row: extract_embeddings_context(row, pipe, tokenizer)  # type:ignore
    )

    # convert to df and save pickle file
    df = ds.to_pandas()
    df.to_pickle(f'data/claimant_embeddings_{model_name.split("/")[-1]}.pickle')  # type: ignore


if __name__ == '__main__':
    main()
