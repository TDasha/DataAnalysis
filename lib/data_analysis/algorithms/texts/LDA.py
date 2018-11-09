import numpy
from lib.data_analysis.texts.Corpus import Corpus
from lib.data_analysis.texts.Vocabulary import Vocabulary
from lib.data_analysis.texts.StopWords import StopWords


class LDA:

    def __init__(self, corpus: Corpus = None, stop_words: StopWords = None, K: int = 20, alpha: float = 0.5,
                 beta: float = 0.5, iterations: int = 50):
        vocabulary = Vocabulary(stop_words, excluds_stopwords=False)
        docs = [vocabulary.doc_to_ids(doc.get_text()) for doc in corpus.get_documents()]
        self.V = vocabulary.size()  # number of different words in the vocabulary\
        self.K = K
        self.alpha = numpy.ones(K) * alpha  # parameter of topics prior
        self.docs = docs  # a list of documents which include the words
        self.pers = []  # Array for keeping perplexities over iterations
        self.beta = numpy.ones(vocabulary.size()) * beta  # parameter of words prior
        self.z_m_n = {}  # topic assignements for documents
        self.n_m_z = numpy.zeros((len(self.docs), K))  # number of words assigned to topic z in document m
        self.n_z_t = numpy.zeros((K, vocabulary.size())) + beta  # number of times a word v is assigned to a topic z
        self.theta = numpy.zeros((len(self.docs), K))  # topic distribution for each document
        self.phi = numpy.zeros((K, vocabulary.size()))  # topic-words distribution for whole of corpus
        self.n_z = numpy.zeros(K) + vocabulary.size() * beta  # total number of words assigned to a topic z
        self.iterations = iterations

        for m, doc in enumerate(docs):  # Initialization
            for n, w in enumerate(doc):
                z = numpy.random.randint(0, K)  # Randomly assign a topic to a word and increase the counting array
                self.n_m_z[m, z] += 1
                self.n_z_t[z, w] += 1
                self.z_m_n[(m, n)] = z
                self.n_z[z] += 1
