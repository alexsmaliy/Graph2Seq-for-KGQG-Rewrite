import regex

EOS_TOKEN = '</s>'

article_regex = regex.compile(r"\b(a|an|the)\b")
multiple_whitespace_regex = regex.compile(r"\s{2,}")
punctuation_regex = regex.compile(r"\p{Punct}")

def normalize_string(s):
    s = s.lower()
    s = article_regex.sub(" ", s)
    s = punctuation_regex.sub(" ", s)
    s = multiple_whitespace_regex.sub(" ", s)
    s = s.strip()
    return s
