#! /usr/bin/env python

import numpy as np
import pickle
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('/data/wiki.ko.vec')
vocab = model.vocab
embeddings = np.array([model.word_vec(k) for k in vocab.keys()])

with open('/data/fasttext_vocab_ko.dat', 'wb') as fw:
    pickle.dump(vocab, fw, protocol=pickle.HIGHEST_PROTOCOL)

np.save('/data/fasttext_embedding_ko.npy', embeddings)