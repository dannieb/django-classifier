'''
Created on Feb 10, 2012

@author: Dannie
'''
from classifier.models import ClassifierCategory, Document, \
    CategoryDocumentCountIndex, FeatureCounts
from classifier.trainer import Trainer
from json.decoder import JSONDecoder
import logging
import math

class Classifier(object):
    trainer = Trainer()
    
    def __init__(self):
        self.savedCategories = None
        self._corpus = ""
        self._features = []
        self._classifierIndex = None
        self._mins = {}
    
    def _getCorpusShort(self):
        return self._corpus[:50] if self._corpus else ""
    
    def _getGroupedCategories(self, categories):
        groupedCategories = {}
        for category in categories :
            if not groupedCategories.get(category.categoryName) :
                groupedCategories[category.categoryName] = {}
            groupedCategories[category.categoryName][category.yes] = category 
        return groupedCategories
    
    def getFeatures(self, corpus):
        return self.trainer.getFeatures(corpus)
    
    def setMinThreshold(self, categoryName, yes, value):
        if not self._mins.get(categoryName) :
            self._mins[categoryName] = {}
        self._mins[categoryName][yes] = value
        
    def getMinThreshold(self, categoryName, yes):
        return self._mins.get(categoryName, {}).get(yes, .6)
        
    '''
    Classification of text corpus
    '''
    def classify(self, corpus):
        self._corpus = corpus
        self._features = self.getFeatures(corpus)
        self._classifierIndex = ClassifierIndex(self._features)
    
        # Find the category with the highest probability
        logger = logging.getLogger("Classifier.classify")
        
        categories = ClassifierCategory.getAllCategories()
        
        groupedCategories = self._getGroupedCategories(categories)
        probableTags = []
        try :
            probableTags = self._getProbableTags(groupedCategories)
        except Exception, ex :
            logger.exception("classification failure:  " + str(ex))

        return probableTags
        
        
    def _getProbableTags(self, groupedCategories):
        logger = logging.getLogger("Classifier._getProbableTags")
        probableTags = []
        
        for tagName in groupedCategories :
            yesProb = self._getProb(groupedCategories[tagName], True)
            noProb = self._getProb(groupedCategories[tagName], False)
                
            yesMin = self.getMinThreshold(tagName, True)
            noMin = self.getMinThreshold(tagName, False)
    
            if self._isYesNo(yesProb, noProb, yesMin, noMin) :
                probableTags.append((tagName, True))
            else :
                probableTags.append((tagName, False))
                
            logger.info("Tag:%s = %s v %s" % (tagName, str(yesProb), str(noProb)))
                
        return probableTags
    
    def __getTagsText(self, tags):
        tagText = ""
        for tag in tags :
            tagText += tag.categoryName + ", "
        return tagText
    
'''
Classification using bayes
'''
class BayesianClassifier(Classifier):
    
    def __init__(self, threshold=0.1):
        super(BayesianClassifier, self).__init__()
        self.threshold = threshold
    
    def _isYesNo(self, yesProb, noProb, yesMin=0, noMin=0):
        return math.log(yesProb/noProb) > 0
       
    
    def _getProb(self, categoryYesNo, yes):
        # Ratio of given category in the trainer database.
        numDocumentsYes = self._classifierIndex.getNumDocumentsForCategory(categoryYesNo[yes])
        
        catprob = float(numDocumentsYes)
        # Probably of given doc being in the given category.
        docprob = self.__getDocumentProb(categoryYesNo, yes)
        return docprob * catprob
    
    ''' Probablity that given document is in the given category. '''
    def __getDocumentProb(self, categoryYesNo, yes):
        # Multiply the probabilities of all the features together
        category = categoryYesNo[yes]
        prob = 0
        numDocumentsForCategory = self._classifierIndex.getNumDocumentsForCategory(category)
        
        if numDocumentsForCategory > 0 :
            prob = 1
            for feature in self._features : 
                weight = 1
                ap = 0.5
                
                featureCategoryCount = self._classifierIndex.getFeatureCategoryCount(feature, category)
                basicProb = float(featureCategoryCount) / float(numDocumentsForCategory)
                
                # Count the number of times this feature has appeared in
                # all categories
                totals = self._classifierIndex.getTotalFeatureCount(feature, categoryYesNo.values())
                
                # Calculate the weighted average
                prob *= ((weight * ap) + (totals * basicProb)) / (weight + totals)
    
        return prob
    
class FisherBayesClassifier(Classifier):
    
    def __init__(self, threshold=0.1):
        super(FisherBayesClassifier, self).__init__()
        self.threshold = threshold
    
    def _isYesNo(self, yesProb, noProb, yesMin=0, noMin=0):
        isYes = False
        if yesProb > yesMin :
            isYes = True
        if noProb > noMin and noProb > yesProb:
            isYes = False
        return isYes
    
    
    def _getProb(self, categoryYesNo, yes):
        # Multiply the probabilities of all the features together
        category = categoryYesNo[yes]
        notCategory = categoryYesNo[not yes]
        prob = 0
        numDocumentsForCategory = self._classifierIndex.getNumDocumentsForCategory(category)
        
        if numDocumentsForCategory > 0 :
            prob = 1
            for feature in self._features : 
                weight = 1
                ap = 0.5
                
                featureCategoryCount = self._classifierIndex.getFeatureCategoryCount(feature, category)
                basicProb = float(featureCategoryCount) / float(numDocumentsForCategory)
                
                if basicProb > 0 :
                    featureCategoryCount2 = self._classifierIndex.getFeatureCategoryCount(feature, notCategory)
                    freqSum = basicProb + (float(featureCategoryCount2/float(numDocumentsForCategory)))
                    basicProb = basicProb/freqSum
                    
                # Count the number of times this feature has appeared in
                # all categories
                totals = self._classifierIndex.getTotalFeatureCount(feature, categoryYesNo.values())
                
                # Calculate the weighted average
                prob *= ((weight * ap) + (totals * basicProb)) / (weight + totals)
    
    
        if prob > 0 :
            prob = self.__invchi2(-2*math.log(prob), len(self._features) * 2)
    
        return prob
        
    
    def __invchi2(self, chi, df) :
        m = chi/2.0
        sum = term = math.exp(-m)
        for i in range(1, df//2) :
            term *= m / i
            sum += term
        return min(sum, 1.0)

'''
This index is purely for performance reasons.  When the classifier starts, it loads all the classification
data into memory so that the database doesn't get hit aftewards.
'''
class ClassifierIndex(object):
    __featureTagIndex = None
    __documentCountHash = None
    
    def __init__(self, features):
        self.__jsonDecoder = JSONDecoder()
        self.__featureTagIndex = {}
        self.__documentCountHash = {}
        self.loadFeatureCountsForCategories(features)
        self.loadAllDocumentCounts()
        self.loadNumberOfDocuments()

    def loadNumberOfDocuments(self):
        logger = logging.getLogger("ClassifierIndex.loadNumberOfDocuments")
        try :
            self.__numberOfDocuments = Document.objects.count()
        except Exception, ex :
            logger.exception("Failed to load number of documents: " + str(ex))
            raise ClassifyIndexLoadFailure("Failed to load number of documents: " + str(ex))

    def loadAllDocumentCounts(self):
        logger = logging.getLogger("ClassifierIndex.loadAllDocumentCounts")
        try :
            countIndex = CategoryDocumentCountIndex.getCountIndex()
            self.__documentCountHash = self.__jsonDecoder.decode(countIndex.countData) if countIndex.countData else {}
        except Exception, ex :
            logger.exception("Failed to load all documents counts: " + str(ex))
            raise ClassifyIndexLoadFailure("Failed to load all document counts: " + str(ex))
        
    '''
    Caches the feature counts for all top-level categories
    Constructs the following index to improve to calculations:
    index[featureName][tagId][categoryId]=count
    '''  
    def loadFeatureCountsForCategories(self, features, categoryIds=[0]):
        logger = logging.getLogger("ClassifierIndex.loadAllFeatureCounts")
        
        logger.info("loading all feature counts")
        try :
            features = FeatureCounts.objects.filter(featureName__in=features)
        
            if features :
                for feature in features :
                    index = self.__jsonDecoder.decode(feature.countData) if feature.countData else {}
                    self.__featureTagIndex[feature.featureName] = index

        except Exception, ex :
            raise ClassifyIndexLoadFailure("Failed to load the feature counts!: %s " % str(ex))
        
            
    '''
    Retrieves the total number of documents that have been trained for a given category
    '''
    def getNumDocumentsForCategory(self, category):
        count = self.__documentCountHash.get(str(category.id))
        return count if count else 0
            
    '''
    Returns the total number of documents trained
    '''
    def getNumberOfDocuments(self):
        return self.__numberOfDocuments 
    
    '''
    Retrieves the total feature counts for a particular classifier category
    '''
    def getFeatureCategoryCount(self, featureName, category):
        index = self.__featureTagIndex.get(featureName)
        return index.get(str(category.id), 0) if index else 0
    
    '''
    Retrieves the total feature counts for the classifier category Yes and No
    '''
    def getTotalFeatureCount(self, featureName, categoriesYesNo) :
        index = self.__featureTagIndex.get(featureName)
        return index.get(str(categoriesYesNo[0].id), 0) + index.get(str(categoriesYesNo[1].id), 0) if index else 0
          
          
class ClassifyIndexLoadFailure(Exception):
    def __init__(self, value):
        self.value = value
        
    def __str__(self):
        return repr(self.value)  

