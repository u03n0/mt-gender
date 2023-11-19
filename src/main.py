import argparse

from transformers import AutoTokenizer, MarianMTModel

from preprocessing import create_ds
from utils import lang_converter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evaluate", choices=['deu-por', 'eng-fra', 'eng-spa'], type=str, required=True, action="store")
    args = parser.parse_args()

    src_trg = args.evaluate
    ds = create_ds(src_trg)
    src, trg = lang_converter(src_trg)

    # Bilingual model
    model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
    model_marian = MarianMTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # for each model, get the custom bleu score
    # for each model, use built in bleu score


    # compare


    # compare multilingual class vs bilingual

    # observations

    # save visuals