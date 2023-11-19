from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def get_ngrams(sentence, n):
  """
  """
  words = [word.lower() for word in sentence.split()]
  return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def precision(corpus, refs, n):
  total_count = 0
  total_ngrams = 0
  for doc in corpus:
    ngrams = get_ngrams(doc, n)
    total_ngrams += len(ngrams)
    ngram_counts = Counter(ngrams)
    for ngram in set(ngrams):
        max_count = 0
        for ref in refs:
            max_count = max(max_count, Counter(get_ngrams(ref, n)).get(ngram, 0))
        total_count += min(max_count, ngram_counts[ngram])
  return total_count / total_ngrams


def brevity_penalty(corpus, refs):
  lower_n_split = lambda x: x.lower().split()
  corpus_len = len(corpus)
  if corpus_len == 0:
      return 0
  cleaned_refs = (lower_n_split(ref) for ref in refs)
  ref_lens = (len(ref) for ref in cleaned_refs)
  closest_ref_len = min(ref_lens, key=lambda ref_len: abs(corpus_len - ref_len))
  return 1 if corpus_len > closest_ref_len else np.exp(1 - closest_ref_len / corpus_len)


def bleu(corpus, refs, n_start, n_end):
    weights = [0.25] * 4
    assert n_end >= n_start > 0
    brev_pen = brevity_penalty(corpus, refs)
    precisions = [precision(corpus, refs, n) for n in range(n_start, n_end + 1)]
    return brev_pen * np.exp(sum(w * np.log(p) for w, p in zip(weights, precisions)))