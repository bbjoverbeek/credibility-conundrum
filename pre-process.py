"""
Pre-process the claimant reliability dataset to ensure valid data
"""

import csv
import os
from glob import glob
import pandas as pd
from tqdm import tqdm

from util import gather_flattened_dataset, get_context


def validate_context(df: pd.DataFrame) -> set[str]:
    """Checks whether the span from the context is equal to the source column.
    Returns list of tasks where this is not the case
    """

    invalid_task_ids = []

    for index, row in tqdm(df.iterrows(), desc='Validating context', total=len(df)):
        span_from_context = ''
        for token_idx, token in enumerate(row['context'].split(' '), 1):
            if token_idx in [int(id_) for id_ in row['tokens_id'].split(' ')]:
                span_from_context += token.replace('"', '\\"') + ' '

        if row['source'] != span_from_context.rstrip(' '):
            print(
                f'Context not correctly fetched for example {index}:\n{row["source"]} \n {span_from_context.rstrip(" ")}\n'
            )
            invalid_task_ids.append(row['task_id'])

    return set(invalid_task_ids)


def remove_invalid_examples(
    source_folder: str, target_folder: str, invalid_task_ids: set[str]
) -> None:
    """Removes the invalid tasks from all csv files in the provided path recursively,
    and writes the processed files to the target folder
    """
    # get all source filepaths
    source_filepaths = glob(f'{source_folder}**/*.csv', recursive=True)

    for source_filepath in source_filepaths:
        target_filepath = target_folder + source_filepath.removeprefix(source_folder)
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)

        # extract the fieldnames first
        with open(source_filepath, 'r', encoding='utf-8-sig') as inp:
            fieldnames = inp.readlines()[0].removesuffix('\n').split(',')

        # parse the files and do not write invalid rows
        with (
            open(source_filepath, 'r', encoding='utf-8-sig') as inp,
            open(target_filepath, 'w') as outp,
        ):
            csv_writer = csv.DictWriter(outp, fieldnames=fieldnames)
            # csv_writer.writeheader()

            for row_dict in csv.DictReader(inp, fieldnames=fieldnames):
                # print(row_dict)
                # check if example is invalid
                if row_dict['task_id'] in invalid_task_ids:
                    print(
                        f'removed task {row_dict["task_id"]} from file {source_filepath}'
                    )
                else:
                    csv_writer.writerow(row_dict)


def main() -> None:
    """Find and remove invalid examples in claimant dataset"""
    df = gather_flattened_dataset(path='./data/claimant_data/flattened')

    df['context'] = df.progress_apply(
        lambda row: get_context(row['file_id'], row['sentence']), axis=1
    )  # type: ignore

    invalid_task_ids = set()
    invalid_task_ids = validate_context(df)

    print(f'\nExamples to be removed: {invalid_task_ids}')

    remove_invalid_examples(
        source_folder='data/claimant_data/',
        target_folder='data/claimant_data_processed/',
        invalid_task_ids=invalid_task_ids,
    )

    print(f'\nRemoved {len(invalid_task_ids)} invalid examples')


if __name__ == '__main__':
    main()
