from .preprocess import MapItems


class NLTKTokenizer(MapItems):
    """
    A preprocessor using the nltk tokenizer
    :param language: language used by the tokenizer
    :param preserve_line: defines is line breaks should be kept as line breaks
    """

    def __init__(self, language='english', preserve_line=False):
        import nltk

        def tokenizer(x):
            return nltk.word_tokenize(x, language, preserve_line)

        super().__init__(tokenizer)
