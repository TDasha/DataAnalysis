from lib.data_analysis.algorithms.texts.LDA import LDA
from lib.data_analysis.texts.StopWords import StopWords
from lib.data_analysis.texts.Corpus import Corpus
from lib.data_analysis.texts.Vocabulary import Vocabulary
import pandas as pd

sw = StopWords("/home/sokolov/PycharmProjects/t/DataAnalysis/lib/data_analysis/texts/stopwords.dic")
sw.load_stop_words_from_file()

data = pd.read_csv("/home/sokolov/PycharmProjects/t/DataAnalysis/LWs/LW_11/data/lenta_ru.csv")
documents = data["text"].tolist()

corpus = Corpus()
corpus.load_corpus_from_list(documents)

iterations = 50
voca = Vocabulary(excluds_stopwords=False)
docs = [voca.doc_to_ids(doc) for doc in corpus]

lda = LDA(K=20, alpha=0.5, beta=0.5, docs=docs, V=voca.size())
print(lda)