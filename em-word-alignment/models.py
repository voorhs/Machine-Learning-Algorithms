from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple

import numpy as np

from preprocessing import TokenizedSentencePair


class BaseAligner(ABC):
    """
    Describes a public interface for word alignment models.
    """

    @abstractmethod
    def fit(self, parallel_corpus: List[TokenizedSentencePair]):
        """
        Estimate alignment model parameters from a collection of parallel sentences.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
        """
        pass

    @abstractmethod
    def align(self, sentences: List[TokenizedSentencePair]) -> List[List[Tuple[int, int]]]:
        """
        Given a list of tokenized sentences, predict alignments of source and target words.

        Args:
            sentences: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            alignments: list of alignments for each sentence pair, i.e. lists of tuples (source_pos, target_pos).
            Alignment positions in sentences start from 1.
        """
        pass


class DiceAligner(BaseAligner):
    def __init__(self, num_source_words: int, num_target_words: int, threshold=0.5):
        self.cooc = np.zeros(
            (num_source_words, num_target_words), dtype=np.uint32)
        self.dice_scores = None
        self.threshold = threshold

    def fit(self, parallel_corpus):
        for sentence in parallel_corpus:
            # use np.unique, because for a pair of words we add 1 only once for each sentence
            for source_token in np.unique(sentence.source_tokens):
                for target_token in np.unique(sentence.target_tokens):
                    self.cooc[source_token, target_token] += 1
        self.dice_scores = (2 * self.cooc.astype(np.float32) /
                            (self.cooc.sum(0, keepdims=True) + self.cooc.sum(1, keepdims=True)))

    def align(self, sentences):
        result = []
        for sentence in sentences:
            alignment = []
            for (i, source_token), (j, target_token) in product(
                    enumerate(sentence.source_tokens, 1),
                    enumerate(sentence.target_tokens, 1)):
                if self.dice_scores[source_token, target_token] > self.threshold:
                    alignment.append((i, j))
            result.append(alignment)
        return result


class WordAligner(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters

    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        """
        Given a parallel corpus and current model parameters, get a posterior distribution over alignments for each
        sentence pair.
        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
        Returns:
            posteriors: list of np.arrays with shape (src_len, target_len).
                posteriors[k][i][j] gives a posterior probability of
                target token i to be aligned to source token j in a sentence k.
        """
        res = []

        for pair in parallel_corpus:
            # get matrix of probs `theta(t_i,j)`
            probs = self.translation_probs[np.ix_(pair.source_tokens, pair.target_tokens)]

            # normalize each column (according to formula of q_i(j))
            q = probs / probs.sum(axis=0, keepdims=True)

            # kth sentence is done
            res.append(q)

        return res

    def _compute_elbo(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]) -> float:
        """
        Compute evidence (incomplete likelihood) lower bound for a model given data and the posterior distribution
        over latent variables.
        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (result of WordAligner._e_step).
        Returns:
            elbo: the value of evidence lower bound

        Tips: 
            1) Compute mathematical expectation with a constant
            2) It is preferred to write this computation with 1 cycle only

        """
        res = .0
        for pair, post in zip(parallel_corpus, posteriors):
            # get matrix of probs `theta(t_i, s_j)`
            # np.ix_() applies broadcasting, so `probs` has shape (len(target), len(source))
            probs = self.translation_probs[np.ix_(pair.source_tokens, pair.target_tokens)]

            # lower bound according to formula
            res += np.sum(post * (np.log(np.maximum(probs, 1e-10)) - np.log(np.maximum(post * post.shape[0], 1e-10))))
        return res

    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.
        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).
        Returns:
            elbo:  the value of evidence lower bound after applying parameter updates
        """
        self.translation_probs *= 0
        for pair, post in zip(parallel_corpus, posteriors):
            np.add.at(self.translation_probs, np.ix_(pair.source_tokens, pair.target_tokens), post)

        self.translation_probs /= self.translation_probs.sum(axis=1, keepdims=True)

        return self._compute_elbo(parallel_corpus, posteriors)

    def fit(self, parallel_corpus):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.
        Sorry for not sticking to my own interface ;)
        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for i in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)
        return history

    def align(self, sentences):
        res = []
        for post in self._e_step(sentences):
            # post has shape (len(source), len(target))
            pairs = post.argmax(axis=0)
            tmp = []
            for t, s in enumerate(pairs):
                tmp.append((s+1, t+1))
            res.append(tmp)
        return res
