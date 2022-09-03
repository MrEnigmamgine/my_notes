import pandas as pd

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

toktok = nltk.tokenize.ToktokTokenizer()
snowball = nltk.stem.SnowballStemmer('english')
wordnet = nltk.stem.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')

def lemmatize(sentence, lemmatizer:object = wordnet) -> str:
    words = sentence.split(' ')
    out = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(out)


def stem(sentence, stemmer:object = snowball) -> str:
    words = sentence.split(' ')
    out = [stemmer.stem(word) for word in words]
    return ' '.join(out)


def word_tokenize(string:str, tokenizer:object = toktok) -> str:
    tokens =  tokenizer.tokenize(string)
    return ' '.join(tokens)

def remove_stopwords(string:str, extra_words:list[str] = [], exclude_words:list[str] = []) -> str:
    tokens = string.split(' ')
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords += extra_words
    stopwords = [word for word in stopwords if not word in exclude_words]
    out = [word for word in tokens if not word in stopwords]
    return ' '.join(out)

def basic_string_clean(string: str, strip=True, lower=True, normalize=True, drop_special=True, drop_punctuation=False) -> str:
    """Returns the same string with the following alterations by default:
    - convert all chars to lowercase
    - maps charcters to fit within ASCII character set (converts accented chars to unaccented counterparts)
    - drops anything that didn't get mapped
    - removes special characters and punctuation
    TODO: Hyphen strategy argument?
    """
    import re
    import unicodedata
    if strip:
        string = string.strip()
    if lower:
        string = string.lower()
    if normalize:
        # Handle curly quotes
        charmap = { 0x201c : u'"',
                    0x201d : u'"',
                    0x2018 : u"'",
                    0x2019 : u"'" }
        string = string.translate(charmap)
        string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    if drop_special:
        string = re.sub(r"[\n-]", ' ', string) # Hyphen strategy for now
        regex = r"[^\w\d\s\.\?\!\:\,\']|[_]"
        string = re.sub(regex, '', string)
    if drop_punctuation:
        regex = r"[\.\?\!\:\,]"
        string = re.sub(regex, '', string)

    return string

def make_nlp_cols(series: pd.Series) -> pd.Series:
    clean = series.apply(basic_string_clean, drop_punctuation=True)\
        .apply(word_tokenize)\
        .apply(remove_stopwords)
    stemmed = clean.apply(stem)
    lemmatized = clean.apply(lemmatize)
    out = pd.concat([clean, stemmed, lemmatized], axis=1)
    out.columns = ['cleaned','stemmed','lemmatized']
    return out

def make_ngrams(words, n):
    if type(words) == pd.core.series.Series:
        input = ' '.join(words)
    else:
        input = words
    output = input.split(' ')
    return pd.Series(nltk.ngrams(output, n))