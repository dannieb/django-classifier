from classifier import stopwords
from nltk.probability import FreqDist
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.treebank import TreebankWordTokenizer
import re
class TrainFailureException(Exception):
    def __init__(self, value):
        self.message = value
        
    def __str__(self):
        return repr(self.message)

class ClassifierFailureException(Exception):
    def __init__(self, value):
        self.message = value
        
    def __str__(self):
        return repr(self.message)
    
    
class FeatureExtractor(object):
    
    def __init__(self, maxFeatures=-1): 
        self.__maxFeatures = maxFeatures
    
    '''
    Given a corpus of text, returns the features
    '''
    def getFeatures(self, corpus):
        stemmer = PorterStemmer()
        stems = FreqDist()
        onlyLettersNumbers = re.compile('[^a-zA-Z0-9%!]')
        corpus = onlyLettersNumbers.sub(' ', corpus.lower())
        corpus = TreebankWordTokenizer().tokenize(corpus)
        
        count = 0
        for word in corpus :
            if not stopwords.STOP_WORDS.get(word) and len(word.strip()) > 1 :
                stems.inc(stemmer.stem_word(word))
                count += 1
                if self.__maxFeatures > 0 and count >= self.__maxFeatures :
                    break
                
        features = stems.samples()
        
        return features