import numpy as np
from relation_extraction.utils import load_vocab
import constants
import os


# NLPLAB_W2V = 'data/w2v_model/wikipedia-pubmed-and-PMC-w2v.bin'
NLPLAB_W2V = 'data/w2v_model/BioWordVec_PubMed_MIMICIII_d200.vec.bin'
# NLPLAB_W2V = 'data/w2v_model/w2v_retrain.bin'


def export_trimmed_nlplab_vectors(vocab, trimmed_filename, dim=200, bin=NLPLAB_W2V):
    """
    Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
        :param bin:
    """
    # embeddings contains embedding for the pad_tok as well
    embeddings = np.zeros([len(vocab) + 1, dim])
    with open(bin, 'rb') as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        print('nlplab vocab size', vocab_size)
        binary_len = np.dtype('float32').itemsize * layer1_size

        count = 0
        m_size = len(vocab)
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word)
                    break
                if ch != b'\n':
                    word.append(ch)
            word = word.decode("utf-8")

            if word in vocab:
                count += 1
                embedding = np.frombuffer(f.read(binary_len), dtype='float32')
                word_idx = vocab[word]
                embeddings[word_idx] = embedding
            else:
                f.read(binary_len)

    print('Missing rate {}'.format(1.0 * (m_size - count)/m_size))
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


vocab_words = load_vocab(constants.ALL_WORDS)
export_trimmed_nlplab_vectors(vocab_words, 'data/w2v_model/biowordvec_nlplab.npz')



