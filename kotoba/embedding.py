from abc import ABCMeta, abstractmethod
from itertools import chain
from .itertools import uniquify, flatten
from .preprocess import MapItems


class TokenEmbedding(metaclass=ABCMeta):
    """
    Base class for implementing a token embedding class
    """

    @abstractmethod
    def token_to_id(self, token):
        """
        Converts a token to an integer id
        :param token: token to convert
        :return: equivalent integer id
        """
        pass

    @abstractmethod
    def id_to_token(self, id_):
        """
        Converts an integer id back to a token
        :param id_: integer id
        :return:  equivalent token
        """
        pass

    @abstractmethod
    def token_coverage(self, tokens):
        """
        Calculate the rate of tokens that can be embedded
        :param tokens: Data tokens
        :return: Percentage of convertible tokens
        """
        pass

    @abstractmethod
    def id_coverage(self, ids):
        """
        Calculates the rate of ids that can be embedded
        :param ids: Data ids
        :return: Percentage of convertible integer ids
        """
        pass

    @abstractmethod
    def export_token_list(self, path):
        """
        Serializes the embedding to a file
        :param path: Path to save the serialized tokens
        """
        pass


class Embedding(TokenEmbedding):
    """
    Transforms tokens into their integer ids

    :param token_list: List of tokens to embed
    :param special_tokens: Special tokens like padding, ending, etc.
    :param unk_idx: Id for the unknown token
    """
    def __init__(self, token_list, special_tokens=None, unk_idx=None):
        special_tokens = special_tokens or []
        index_to_token = self._create_index_token_list(token_list, special_tokens)
        self._index_to_token = tuple(index_to_token)
        self._token_to_index = self._create_token_index_dict(index_to_token)
        self._token_set = frozenset(index_to_token)
        self._unk = (None if unk_idx is None
                     else index_to_token[unk_idx])
        self._unk_id = unk_idx or -1

    # noinspection PyMethodMayBeStatic
    def _create_parent_dir(self, path):
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    # noinspection PyMethodMayBeStatic
    def _create_index_token_list(self, token_list, special_tokens):
        token_list = flatten(token_list)
        all_tokens = chain(special_tokens, token_list)
        all_tokens = uniquify(all_tokens)
        index_to_token = list(all_tokens)
        return index_to_token

    # noinspection PyMethodMayBeStatic
    def _create_token_index_dict(self, index_token_list):
        count = len(index_token_list)
        return dict(zip(index_token_list, range(count)))

    def token_to_id(self, token):
        return self._token_to_index.get(token, self._unk_id)

    def id_to_token(self, id_):
        if id_ < 0:
            return self._unk
        else:
            try:
                return self._index_to_token[id_]
            except IndexError:
                return self._unk

    # noinspection PyMethodMayBeStatic
    def _get_coverage(self, base_set, target_list):
        target_set = set(flatten(target_list))
        common_entries = base_set & target_set
        return len(common_entries) / len(target_set)

    def token_coverage(self, tokens):
        return self._get_coverage(self._token_set, tokens)

    def id_coverage(self, ids):
        ids_set = set(range(len(self._token_set)))
        return self._get_coverage(ids_set, ids)

    def export_token_list(self, path, encoding='utf-8'):
        self._create_parent_dir(path)
        with open(path, 'w', encoding=encoding) as file:
            file.writelines((l + '\n' for l in self._index_to_token))

    @property
    def token_size(self):
        return len(self._index_to_token)

    @classmethod
    def from_export_file(cls, path, special_tokens=None, unk_idx=None, encoding='utf-8'):
        with open(path, 'r', encoding=encoding) as file:
            return cls((l.strip() for l in file), special_tokens, unk_idx)

    @classmethod
    def from_glove_file(cls, path, special_tokens=None, unk_idx=None, encoding='utf-8'):
        """
        :param path: Path of the glove file
        :param special_tokens: special tokens to append to the gloves
        :param unk_idx: Id for the unknown token
        :param encoding: encoding for the glove file
        :return:
        """
        with open(path, 'r', encoding=encoding) as file:
            parsed_file = (l.strip().split(' ')[0] for l in file)
            return cls(parsed_file, special_tokens, unk_idx)


class EmbedTokenToID(MapItems):
    """
    A preprocessor that converts the tokens to integer id
    :param token_embedding: TokenEmbedding object
    """
    def __init__(self, token_embedding):
        super().__init__(token_embedding.token_to_id)


class EmbedIDToToken(MapItems):
    """
    A preprocessor that converts the integer id to tokens
    :param token_embedding: TokenEmbedding object
    """
    def __init__(self, token_embedding):
        super().__init__(token_embedding.id_to_token)
