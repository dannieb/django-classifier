'''
Created on Feb 10, 2012

@author: Dannie
'''
from classifier.models import ClassifierCategory, Document, \
    CategoryDocumentCountIndex, FeatureCounts
from classifier.trainer import Trainer
from json.decoder import JSONDecoder
import logging

class Classifier(object):
    trainer = Trainer()
    
    def __init__(self, threshold=.5):
        self.threshold = threshold
        self.savedCategories = None
        self._corpus = ""
        self._features = []
        self._classifierIndex = None
    
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
    
    '''
    Generic classification
    '''
    def classify(self, corpus):
        self._corpus = corpus
        self._features = self.getFeatures(corpus)
        self._classifierIndex = ClassifierIndex(self._features)
    
        # Find the category with the highest probability
        logger = logging.getLogger("Classifier.classify")
        groupedCategories = self._getGroupedCategories(ClassifierCategory.getAllCategories())
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
            
            dilutedYesProb = yesProb * self.threshold
            if dilutedYesProb > noProb :
                probableTags.append(tagName)
                
                yesCategory = groupedCategories[tagName][True]
                numDocsForCategory = self._classifierIndex.getNumDocumentsForCategory(yesCategory)
                logger.info("YES %s: %s NO:%s [Final Yes]:%s - sample:%s" % (tagName, str(yesProb), str(noProb), str(dilutedYesProb), str(numDocsForCategory)))
    
        return probableTags
    
    def __getTagsText(self, tags):
        tagText = ""
        for tag in tags :
            tagText += tag.categoryName + ", "
        return tagText
    
    @staticmethod
    def getClassifier():
        return BayesianClassifier()
'''
Classification using bayes
'''
class BayesianClassifier(Classifier):
    
    def __init__(self, threshold=0.1):
        super(BayesianClassifier, self).__init__()
        self.threshold = threshold
    
    def _getProb(self, categoryYesNo, yes):
        category = categoryYesNo[yes]
        # Ratio of given category in the trainer database.
        catprob = float(self._classifierIndex.getNumDocumentsForCategory(category)) / float(self._classifierIndex.getNumberOfDocuments())
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

