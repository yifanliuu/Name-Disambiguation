import config as cfg
import logging

from gensim.models import word2vec

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(cfg.ALL_TEXT_PATH)
    model = word2vec.Word2Vec(sentences, size=100,negative =5, min_count=2, window=5)
    model.save()