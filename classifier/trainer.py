'''
Created on Feb 10, 2012

@author: Dannie
'''
from classifier import FeatureExtractor
from classifier.models import ClassifierCategory, Document, FeatureCounts, \
    DocumentCategoryCounts, CategoryDocumentCountIndex
from django.db import transaction
from json.decoder import JSONDecoder
from json.encoder import JSONEncoder
import logging

'''
Deal Trainer.
'''
class Trainer(object):
    
    categories = {}
    
    def __init__(self):
        self._jsonDecoder = JSONDecoder()
        self._jsonEncoder = JSONEncoder()
        self.__featureExtractor = FeatureExtractor()
    
    def setFeatureExtractor(self, featureExtractor):
        self.__featuredExtractor = featureExtractor
        
        
    def __isNumeric(self, feature):
        isNumeric = False
        try :
            float(feature)
            isNumeric = True
        except ValueError:
            pass
        return isNumeric

    '''
    Given a list of yes category names, retrieves the yes/no category hash that the trainer needs
    '''
    def __getCategoriesFromNames(self, yesTagNames, noTagNames):
        finalCategories = []
        
        #create the categories if they don't already exist
        for tagName in yesTagNames :
            if tagName :
                categoryYes,_ = ClassifierCategory.objects.get_or_create(categoryName=tagName, yes=True)
                categoryNo,_ = ClassifierCategory.objects.get_or_create(categoryName=tagName, yes=False)
    
                finalCategories.append(categoryYes)
        for tagName in noTagNames :
            if tagName :
                categoryYes,_ = ClassifierCategory.objects.get_or_create(categoryName=tagName, yes=True)
                categoryNo,_ = ClassifierCategory.objects.get_or_create(categoryName=tagName, yes=False)
                
                finalCategories.append(categoryNo)

        return finalCategories 

    '''
    Trains a corpus of data.
    '''
    @transaction.commit_manually   
    def train(self, corpus="", yesTagNames=None, noTagNames=None):
        logger = logging.getLogger("Trainer.train")
        success = False
        
        categories = []
        try :
            document = Document.getDocumentByCorpus(corpus)
            if not document :
                features = self.__featuredExtractor.getFeatures(corpus)
                categories = self.__getCategoriesFromNames(yesTagNames, noTagNames)

                document = Document(corpus=corpus)
                document.save()
                documentCounts = {}
                for category in categories :
                    self.__incrementCategoryCount(documentCounts, category)
                DocumentCategoryCounts(document=document, countData=self._jsonEncoder.encode(documentCounts)).save()
                
                for feature in features :
                    featureCount,_ = FeatureCounts.objects.get_or_create(featureName=feature)
                    counts = self._jsonDecoder.decode(featureCount.countData) if featureCount.countData else {}
                    
                    for category in categories :
                        self.__incrementCategoryCount(counts, category)
                            
                    featureCount.countData = self._jsonEncoder.encode(counts)
                    featureCount.save()
                        
                #We keep an index of category document counts for faster classification later on
                catDocCountIndex = CategoryDocumentCountIndex.getCountIndex()
                index = self._jsonDecoder.decode(catDocCountIndex.countData) if catDocCountIndex.countData else {}
                for category in categories :
                    self.__incrementCategoryCount(index, category)
                catDocCountIndex.countData = self._jsonEncoder.encode(index)
                catDocCountIndex.save()
                    
                success = True
                
                transaction.commit()
            else :
                logger.info("Document already exists: " + str(document.id) + " - " + document.corpusHash)
                success = True
    
        except Exception, ex :
            logger.info("Bad data:%s" % corpus)
            logger.exception("Failed to save the trained data: " + str(ex))
            transaction.rollback()
        
        return success

    
    '''
    Helper function to increment the category count
    '''
    def __incrementCategoryCount(self, dict, category):
        id = str(category.id)
        if not dict.get(id) :
            dict[id] = 1
        else :
            dict[id] += 1
            
    def __decrementCategoryCount(self, dict, category):
        id = str(category.id)
        if dict.get(id) and dict[id] > 0 :
            dict[id] -= 1
    
    
    '''
    Untrains a corpus of data ... assuming it exists in the system.
    ''' 
    @transaction.commit_manually   
    def untrain(self, corpus=""):
        logger = logging.getLogger("Trainer.untrain")
        success = False

        try :
            document = Document.getDocumentByCorpus(corpus)
            
            if document :
                categories = DocumentCategoryCounts.getCategoriesForDocument(document)
                features = self.__featuredExtractor.getFeatures(corpus)
                document.delete()
                
                for feature in features :
                    featureCount,_ = FeatureCounts.objects.get_or_create(featureName=feature)
                    counts = self._jsonDecoder.decode(featureCount.countData) if featureCount.countData else {}
                    
                    for category in categories :
                        self.__decrementCategoryCount(counts, category)
                            
                    featureCount.countData = self._jsonEncoder.encode(counts)
                    featureCount.save()
                        
                #We keep an index of category document counts for faster classification later on
                catDocCountIndex = CategoryDocumentCountIndex.getCountIndex()
                index = self._jsonDecoder.decode(catDocCountIndex.countData) if catDocCountIndex.countData else {}
                for category in categories :
                    self.__decrementCategoryCount(index, category)
                catDocCountIndex.countData = self._jsonEncoder.encode(index)
                catDocCountIndex.save()
                
                success = True
                
                transaction.commit()
                    
            else :
                logger.info("Document doesn't exist")
                success = True
        except Exception, ex :
            logger.exception("Failed to untrain the document: " + str(ex))
            transaction.rollback()
            
        return success
            

    