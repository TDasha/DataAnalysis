from unittest import TestCase
from lib.data_analysis.texts.document import Document


class TestDocument(TestCase):

    def setUp(self):
        self.doc = Document("А роза упала на лапу Азора")

    def test_get_text(self):
        self.assertEqual(self.doc.get_text(), "А роза упала на лапу Азора")

    def test_get_tag(self):
        self.assertEqual(self.doc.get_tag(), "")

    def test_get_text_as_list_of_words(self):
        self.assertEqual(self.doc.get_text_as_list_of_words(), ["А", "роза", "упала", "на", "лапу", "Азора"])

    def test_set_text(self):
        self.doc.set_text("Text!")
        self.assertEqual(self.doc.get_text(), "Text!")

    def test_set_tag(self):
        self.doc.set_tag("TAG")
        self.assertEqual(self.doc.get_tag(), "TAG")
