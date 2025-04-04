# Install NLTK
!pip install -q nltk

# Import libraries
import nltk
from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer, TreebankWordTokenizer, TweetTokenizer, MWETokenizer
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

# Download data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample sentence
sentence = "Success usually comes to those who are too busy to be looking for it."

# Tokenizers
print("🔹 Whitespace Tokenizer:")
print(WhitespaceTokenizer().tokenize(sentence), '\n')

print("🔹 WordPunct Tokenizer (Punctuation-based):")
print(WordPunctTokenizer().tokenize(sentence), '\n')

print("🔹 Treebank Word Tokenizer:")
tokens = TreebankWordTokenizer().tokenize(sentence)  # ✅ Use this for rest of processing
print(tokens, '\n')

print("🔹 Tweet Tokenizer:")
print(TweetTokenizer().tokenize(sentence), '\n')

print("🔹 MWE Tokenizer (combining 'too busy'):")
mwe = MWETokenizer([('too', 'busy')])
print(mwe.tokenize(sentence.split()), '\n')

# Stemming
porter = PorterStemmer()
snowball = SnowballStemmer("english")

print("🔹 Porter Stemmer:")
print([porter.stem(word) for word in tokens], '\n')

print("🔹 Snowball Stemmer:")
print([snowball.stem(word) for word in tokens], '\n')

# Lemmatization
lemmatizer = WordNetLemmatizer()
print("🔹 Lemmatization (WordNet):")
print([lemmatizer.lemmatize(word) for word in tokens])
