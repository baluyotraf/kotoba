from .preprocess import FunctionPreprocessor


class NLTKTokenizer(FunctionPreprocessor):

    def __init__(self, language='english', preserve_line=False):
        import nltk

        def tokenizer(x):
            return nltk.word_tokenize(x, language, preserve_line)

        super().__init__(tokenizer)
