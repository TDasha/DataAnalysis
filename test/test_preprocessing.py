from unittest import TestCase
from lib.data_analysis.texts.preprocessing import Preprocessing
from lib.data_analysis.texts.document import Document


class TestPreprocessing(TestCase):

    def setUp(self):
        pass

    def test_convert_word_list_to_text(self):
        word_list = ["гнать", "дышать", "hold", "обидеть"]
        text = Preprocessing.convert_word_list_to_text(word_list)
        self.assertEqual("гнать дышать hold обидеть", text)

    def test_lemmatize(self):
        text = Preprocessing.lemmatize("Меня")
        self.assertEqual("меня", text)

    def test_convert_text_to_list_of_words(self):
        word_list = Preprocessing.convert_text_to_list_of_words("А роза упала на лапу Азора")
        self.assertEqual(word_list, ["А", "роза", "упала", "на", "лапу", "Азора"])

    def test_convert_document_to_list_of_words(self):
        doc = Document("А роза упала на лапу Азора")
        word_list = Preprocessing.convert_document_to_list_of_words(doc)
        self.assertEqual(word_list, ["А", "роза", "упала", "на", "лапу", "Азора"])

    def test_convert_list_of_words_to_normal_forms(self):
        word_list = Preprocessing.convert_list_of_words_to_normal_forms(["люблю", "тебя"])
        self.assertEqual(word_list, ["любить", "ты"])

    def test_convert_word_to_normal_form(self):
        word = Preprocessing.convert_word_to_normal_form("гонит")
        self.assertEqual(word, "гнать")

    def test_remove_stop_words_from_list_of_words(self):
        answer = Preprocessing.remove_stop_words_from_list_of_words(["Я"], ["Я", "люблю", "мир"])
        self.assertEqual(answer, ["люблю", "мир"])
