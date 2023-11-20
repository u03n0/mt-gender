from typing import Tuple


def lang_converter(src_trg: str)-> Tuple[str, str]:
    """ Converts a string of language codes
    to ISO language codes.
    """
    d = {
        'deu': 'de',
        'por': 'pt',
        'eng': 'en',
        'fra': 'fr',
        'spa': 'es'
    }
    
    src, trg = src_trg.split('-')
    if src in d and trg in d:
        return d[src], d[trg]

def create_visuals():
    pass


def save_results():
    pass