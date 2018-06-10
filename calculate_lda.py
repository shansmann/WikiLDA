import logging
import itertools
import numpy as np
import time
import random

from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from sklearn.externals import joblib
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.root.level = logging.INFO


def tokenize(text):
	return [token for token in simple_preprocess(text)]

def iter_wiki(dump_file):
	"""Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
	ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
	for title, text, pageid in _extract_pages(smart_open(dump_file)):
		text = filter_wiki(text)
		tokens = tokenize(text)
		if len(tokens) < MIN_NUMBER_OF_WORDS_PER_ARTICLE or any(title.startswith(ns + ':') for ns in ignore_namespaces):
			continue  # ignore short articles and various meta-articles
		yield title, tokens

class WikiCorpus(object):
	def __init__(self, dump_file, dictionary, clip_docs=None):
		"""
		Parse the first `clip_docs` Wikipedia documents from file `dump_file`.
		Yield each document in turn, as a list of tokens (unicode strings).

		"""
		self.dump_file = dump_file
		self.dictionary = dictionary
		self.clip_docs = clip_docs

	def __iter__(self):
		self.titles = []
		for title, tokens in itertools.islice(iter_wiki(self.dump_file), self.clip_docs):
			self.titles.append(title)
			yield self.dictionary.doc2bow(tokens)

	def __len__(self):
		return self.clip_docs

from heapq import nlargest
def sample_from_iterable(iterable, samplesize):
	return (x for _, x in nlargest(samplesize, ((random.random(), x) for x in iterable)))

def get_bow(num_docs, dic_size):
	start = time.time()
	logging.info('creating doc stream')
	doc_stream = (tokens for _, tokens in nlargest(num_docs, ((random.random(), x) for x in iter_wiki('./data/abstracts/' + data_path))))
	#data = random.sample(doc_stream, num_docs)
	logging.info('doc stream shuffled and limited. time elapsed: {}m', int((time.time() - start)/60))
	start = time.time()
	vectorizer = CountVectorizer(min_df=5, max_df=0.7,
								 lowercase=True,
								 token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}',
								 max_features=dic_size)

	data_vectorized = vectorizer.fit_transform(doc_stream)
	logging.info('doc stream vectorized. time elapsed: {}m', int((time.time() - start)/60))

	return vectorizer, data_vectorized

def calculate_lda(topics, iters, data_vectorized):
	start = time.time()
	# Build a Latent Dirichlet Allocation Model
	lda_model = LatentDirichletAllocation(n_topics=topics,
										  max_iter=iters,
										  learning_method='online',
										  n_jobs=4,
										  verbose=1)

	lda_model.fit(data_vectorized)
	logging.info('calculated lda model. time elapsed: {}m', int((time.time() - start)/60))
	return lda_model

def predict(vectorizer, model, text):
	x = model.transform(vectorizer.transform([text]))[0]
	return x

def load_model(filename):
	return joblib.load(filename)

def save_model(filename, model):
	_ = joblib.dump(model, './models/' + filename, compress=9)
	return _


if __name__ == '__main__':
	###################### GENERAL SETTINGS ######################
	data_path = 'dewiki-latest-pages-articles.xml.bz2'
	MIN_NUMBER_OF_WORDS_PER_ARTICLE = 200
	max_features = 5000
	n_docs = 1000

	######################## EXPERIMENT 1 ########################
	topics = 100
	passes = 1

	vec, data = get_bow(n_docs, max_features)
	lda_model_1 = calculate_lda(topics, passes, data)
	save_model('lda{}_{}.pkl'.format(topics, passes), lda_model_1)

	######################## EXPERIMENT 2 ########################
	topics = 250
	passes = 1

	lda_model_2 = calculate_lda(topics, passes, data)
	save_model('lda{}_{}.pkl'.format(topics, passes), lda_model_2)


