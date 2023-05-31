from xml.etree import ElementTree
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np


# =======   TASK A   =======

@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str):
    """
    filename: file (XML markup)
    Example:
        <?xml version="1.0" encoding="utf-8" ?>
        <sentences>
            <s id="project_syndicate_bacchetta1-s1">
                <english>Are the Dollar 's Days Numbered ?</english>
                <czech>Jsou dny dolaru sečteny ?</czech>
                <sure>1-1 3-3 5-2 6-4 7-5</sure>
                <possible>2-2 4-3</possible>
            </s>
            <s id="project_syndicate_bacchetta1-s2">
                <english>Philippe Bacchetta and Eric van Wincoop</english>
                <czech>Philippe Bacchetta and Eric van Wincoop</czech>
                <sure>1-1 2-2 3-3 4-4 5-5 6-6</sure>
                <possible></possible>
            </s>
            <s id="project_syndicate_bacchetta1-s3">
                <english>A year ago , the dollar bestrode the world like a colossus .</english>
                <czech>Ještě před rokem dolar dominoval světu jako imperátor .</czech>
                <sure>10-7 12-8 13-9 2-3 3-2 6-4 7-5 9-6</sure>
                <possible>1-3 11-8 3-1 5-4 8-6</possible>
            </s>
        </sentences>
    Returns
    -------
        sentences: list of (sent1, sent2)
            sent is list of tokens
        alignments: list of (align1, align2)
            align is list of (num1, num2)
                num1 is number of word in sent1
    """
    def subroutine(alignment):
        """parse '1-3 11-8 3-1' to [(1,3), (11,8), (3,1)] """
        if alignment is None:
            return []
        res = []
        for align in alignment.split():
            eng, ch = align.split('-')
            res.append((int(eng), int(ch)))
        return res

    # content of file -> string
    with open(filename, 'r') as fin:
        content = fin.read()

    # to store result
    sentences, alignments = [], []

    # root is <sentences>...</sentences>
    root = ElementTree.fromstring(content.replace('&', '&amp;'))

    # obj is <s id="project_syndicate_bacchetta1-s3">...</s>
    for obj in root:
        # obj[0] is <english>...</english>
        english = obj[0].text.split()

        # obj[1] is <czech>...</czech>
        czech = obj[1].text.split()

        # obj[2] is <sure>...</sure>
        sure = subroutine(obj[2].text)

        # obj[3] is <possible>...</possible>
        possible = subroutine(obj[3].text)

        sentences.append(SentencePair(english, czech))
        alignments.append(LabeledAlignment(sure, possible))

    return sentences, alignments

# =======   TASK B   =======


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None):
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff -- natural number -- most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    Tip: 
        Use cutting by freq_cutoff independently in src and target. Moreover in both cases of freq_cutoff (None or not None) - you may get a different size of the dictionary

    """
    # thank god collections exist
    source_cnt = Counter()
    target_cnt = Counter()

    for pair in sentence_pairs:
        source_cnt.update(pair.source)
        target_cnt.update(pair.target)

    lst = source_cnt.most_common(freq_cutoff)
    source_dict = {token: i for i, (token, cnt) in enumerate(lst)}

    lst = target_cnt.most_common(freq_cutoff)
    target_dict = {token: i for i, (token, cnt) in enumerate(lst)}

    return source_dict, target_dict


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.

    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    def subroutine(sentence, vocabulary):
        """[word1, word2, ...] -> [voc[word1], voc[word2], ...]"""
        res = []
        for t in sentence:
            if t not in vocabulary:
                return None
            res.append(vocabulary[t])

        return np.array(res).astype(int)

    pair_inds = []
    for pair in sentence_pairs:
        pair_inds.append((
            subroutine(pair.source, source_dict),
            subroutine(pair.target, target_dict)
        ))

    res = []
    for source, target in pair_inds:
        if (source is not None) and (target is not None):
            res.append(TokenizedSentencePair(source, target))

    return res
