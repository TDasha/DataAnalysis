from lib.data_analysis.algorithms.texts.LDA import LDA
from lib.data_analysis.texts.StopWords import StopWords
from lib.data_analysis.texts.Corpus import Corpus
import pandas as pd

sw = StopWords("/home/sokolov/PycharmProjects/t/DataAnalysis/lib/data_analysis/texts/stopwords.dic")
sw.load_stop_words_from_file()
sw.load_stop_word_from_nltk_lib()

data = pd.read_csv("/home/sokolov/PycharmProjects/t/DataAnalysis/LWs/LW_11/data/lenta_ru.csv")
documents = data["text"].tolist()

corpus = Corpus()
corpus.load_corpus_from_list(documents)

lda = LDA(corpus=corpus, stop_words=sw, K=20, alpha=0.5, beta=0.5, iterations=50)
lda.run()
print("\n", lda.worddist())
