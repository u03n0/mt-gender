from datasets import Dataset
from pathlib import Path
from typing import List



def create_ds(src_trg: str)-> Dataset:
    """ Creates a Dataset from the appropriate
    txt file in /data from a source-target 
    language-code string.
    """
    return Dataset.from_list(
        [{'source':src, 'target':trg.strip()} for _, _, src, trg in read_file_and_split(src_trg)]
        )


def read_file_and_split(src_trg: str)-> List:
    """ Creates an array of lines(str) having been split 
    on '\t' from a txt file.
    """
    return [line.split("\t") for line in open(get_txt_file(src_trg)).readlines()]


def get_txt_file(src_trg: str)-> str:
    """ Retrieve a filename that has
    """
    p = Path("data")
    for file in p.glob("*.txt"):
        if src_trg in file.name:
            return file