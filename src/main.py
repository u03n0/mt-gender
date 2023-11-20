import argparse
import torch
from transformers import (AutoTokenizer, MarianMTModel, 
                          MBartForConditionalGeneration, MBart50TokenizerFast)
from datasets import concatenate_datasets
from preprocessing import create_ds
from utils import lang_converter

from marian import Marian
from mbart import Mbart


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evaluate", choices=['deu-por', 'eng-fra', 'eng-spa'], type=str, required=True, action="store")
    args = parser.parse_args()

    src_trg = args.evaluate
    ds = create_ds(src_trg)
    src, trg = lang_converter(src_trg)

    # Bilingual model
    marian = Marian(src, trg, ds, device)
    # Multilingual model
    mbart = Mbart(src, trg, ds, device)

    models = [marian, mbart]


    for model in models:
        res = ds.map(model.translate, batched=True)
        print(f"running on {device}")
        concatenated_dataset = concatenate_datasets([ds, res], axis=1)
        print(concatenate_datasets)