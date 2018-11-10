import numpy
from lib.data_analysis.texts.vocabulary import Vocabulary


class LDA:

    def __init__(self, corpus=None, stop_words=None, K=20, alpha=0.5,
                 beta=0.5, iterations=50):

        self.vocabulary = Vocabulary(stop_words, excluds_stopwords=False)
        docs = [self.vocabulary.doc_to_ids(doc.get_text()) for doc in corpus.get_documents()]
        self.V = self.vocabulary.size()  # number of different words in the vocabulary
        self.K = K
        self.alpha = numpy.ones(K) * alpha  # parameter of topics prior
        self.docs = docs  # a list of documents which include the words
        self.pers = []  # Array for keeping perplexities over iterations
        self.beta = numpy.ones(self.vocabulary.size()) * beta  # parameter of words prior
        self.z_m_n = {}  # topic assignements for documents
        self.n_m_z = numpy.zeros((len(self.docs), K))  # number of words assigned to topic z in document m
        self.n_z_t = numpy.zeros((K, self.vocabulary.size())) + beta  # number of times a word v is assigned to a topic z
        self.theta = numpy.zeros((len(self.docs), K))  # topic distribution for each document
        self.phi = numpy.zeros((K, self.vocabulary.size()))  # topic-words distribution for whole of corpus
        self.n_z = numpy.zeros(K) + self.vocabulary.size() * beta  # total number of words assigned to a topic z
        self.iterations = iterations

        for m, doc in enumerate(docs):  # Initialization
            for n, w in enumerate(doc):
                z = numpy.random.randint(0, K)  # Randomly assign a topic to a word and increase the counting array
                self.n_m_z[m, z] += 1
                self.n_z_t[z, w] += 1
                self.z_m_n[(m, n)] = z
                self.n_z[z] += 1

    def inference(self, iteration):
        for m, doc in enumerate(self.docs):
            self.theta[m] = numpy.random.dirichlet(self.n_m_z[m] + self.alpha, 1)
            # sample Theta for each document using uncollapsed gibbs

            for n, w in enumerate(doc):  # update arrays for each word of a document
                z = self.z_m_n[(m, n)]
                self.n_m_z[m, z] -= 1
                self.n_z_t[z, w] -= 1
                self.n_z[z] -= 1
                self.phi[:, w] = self.n_z_t[:, w] / self.n_z

                p_z = self.theta[m] * self.phi[:, w]
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                # sample Z using multinomial distribution of equation 7 of reference 3
                self.n_m_z[m, new_z] += 1
                self.n_z_t[new_z, w] += 1
                self.n_z[new_z] += 1
                self.z_m_n[(m, n)] = new_z

        per = 0
        b = 0
        c = 0
        self.phi = self.n_z_t / self.n_z[:, numpy.newaxis]

        for m, doc in enumerate(self.docs):  # find perplexity over whole of the words of test set
            b += len(doc)

            for n, w in enumerate(doc):
                l = 0
                for i in range(self.K):
                    l += (self.theta[m, i]) * self.phi[i, w]
                c += numpy.log(l)

        per = numpy.exp(-c / b)
        print('perpelixity:', per)

    def worddist(self):
        return self.phi

    def run(self):
        for i in range(self.iterations):
            print('iteration:', i)
            self.inference(i)

        d = self.worddist()
        for i in range(20):
            ind = numpy.argpartition(d[i], -10)[-10:]
            for j in ind:
                print(self.vocabulary[j], ' ', end=""),
            print()
