class Document:

    def __init__(self, text: str, tag: str = '') -> None:
        self._text = text
        self._tag = tag

    def getText(self):
        return self._text

    def getTag(self):
        return self._tag

    def setText(self, text: str):
        self._text = text

    def setTag(self, tag: str = ''):
        self._tag = tag