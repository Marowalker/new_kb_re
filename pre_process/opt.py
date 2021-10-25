from pre_process.segmenters.simple import SimpleSegmenter
from pre_process.segmenters.spacy import SpacySegmenter

from pre_process.tokenizers.simple import SimpleTokenizer
from pre_process.tokenizers.spacy import SpacyTokenizer

from pre_process.options.normalizers import NumericNormalizer
from pre_process.options.snowball_stemmer import SnowballStemmer
from pre_process.options.wordnet_lemmatizer import WordNetLemmatizer


# Options for segmenters
SimpleSegmenter = SimpleSegmenter
SpacySegmenter = SpacySegmenter

# Options for tokenizers
SimpleTokenizer = SimpleTokenizer
SpacyTokenizer = SpacyTokenizer

# Options for optional processes
NumericNormalizer = NumericNormalizer
SnowballStemmer = SnowballStemmer
WordNetLemmatizer = WordNetLemmatizer

# Keys
TOKENIZER_KEY = "tokenizer"
SEGMENTER_KEY = "segmenter"
OPTION_KEY = "options"
