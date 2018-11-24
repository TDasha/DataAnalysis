# coding=utf-8
import artm
import subprocess
import pandas as pd
import re
from nltk.corpus import stopwords
import pymorphy2


def preprocessing_for_artm(number_of_docs_as_collection_length=False, number_of_docs=10):
    data = pd.read_csv("../data/lenta_ru.csv")
    texts = data["text"]
    doc_text = ""
    morph = pymorphy2.MorphAnalyzer()
    if number_of_docs_as_collection_length:
        number_of_docs = len(texts)
    for i in range(number_of_docs):
        text = texts[i]
        doc_text += " |text "
        text = str(text).decode('utf-8')
        text = re.sub("[0-9!@#$%^&*()\[\],\.<>;:\"{}/~`\-+=«»—\|?\^\n\t']+", '', text)
        list_of_words = re.sub(ur"(u?)\w+", ' ', text, ).split(" ")
        filtered_list_of_word = [morph.parse(w.lower())[0].normal_form
                                 for w in list_of_words if w not in stopwords.words("russian")]
        filtered_text = u" ".join(filtered_list_of_word).encode('utf-8').strip()
        doc_text += filtered_text
        doc_text += "\n"
    f = open("../data/lenta.txt", "w")
    f.write(doc_text)
    f.close()


def artm_plsa(batch_vectorizer, topics, topic_names, dictionary):
    model_artm = artm.ARTM(num_topics=topics, topic_names=topic_names, num_processors=2, class_ids={"text": 1},
                           reuse_theta=True, cache_theta=True)
    model_artm.initialize(dictionary=dictionary)
    model_artm.scores.add(artm.PerplexityScore("perplexity", class_ids=["text"], dictionary=dictionary))
    model_artm.num_document_passes = 1
    model_artm.fit_offline(batch_vectorizer=batch_vectorizer,
                           num_collection_passes=45)
    print "Peprlexity for BigARTM PLSA: ", model_artm.score_tracker["perplexity"].value[-1]


def artm_lda(batch_vectorizer, topics, dictionary):
    model_lda = artm.LDA(num_topics=topics, num_processors=2, cache_theta=True)
    model_lda.initialize(dictionary=dictionary)
    model_lda.num_document_passes = 1
    model_lda.fit_offline(batch_vectorizer=batch_vectorizer,
                          num_collection_passes=45)
    print "Perplexity for BigARTM LDA: ", model_lda.perplexity_last_value


def run():
    print 'BigARTM version ', artm.version(), '\n\n\n'
    preprocessing_for_artm(True)
    topics = 10
    batch_vectorizer = artm.BatchVectorizer(data_path="../data/lenta.txt", data_format="vowpal_wabbit",
                                            target_folder="batch_vectorizer_target_folder", batch_size=10)
    topic_names = ["sbj" + str(i) for i in range(topics - 1)] + ["bcg"]
    dictionary = artm.Dictionary("dictionary")
    dictionary.gather(batch_vectorizer.data_path)
    artm_plsa(batch_vectorizer, topics, topic_names, dictionary)
    artm_lda(batch_vectorizer, topics, dictionary)
    subprocess.call(['./clear.sh'])


run()
