from typing import Tuple


def lang_converter(src_trg: str)-> Tuple[str, str]:
    """ Converts a string of source-target language codes
    to ISO language codes.
    """
    d = {
        'deu': 'de',
        'por': 'pt',
        'eng': 'en',
        'fra': 'fr',
        'spa': 'es'
    }
    if src in d and trg in d:
        src, trg = src_trg.split('-')
        return d[src], d[trg]
