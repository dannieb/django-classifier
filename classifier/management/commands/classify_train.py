'''
Created on Feb 10, 2012

@author: Dannie
'''
from classifier.classifiers import Classifier
from classifier.trainer import Trainer
from django.core.management.base import BaseCommand
from optparse import make_option
import os
class Command(BaseCommand):
    help = "Classify/Train Utilities"
    __doc__ = help
    
    option_list = BaseCommand.option_list + (
        make_option('--train_corpus', dest='train_corpus',
                    default=False, help='Trains a corpus'),
        make_option('--untrain_corpus', dest='untrain_corpus',
                    default=False, help='Untrains a corpus'),
                                             
                                             
        make_option('--classify', dest='classify',
                    default='', help='Classifies a text corpus'),
                                             
        make_option('--directory', dest='directory',
                    default='', help='Generic directory input'),
        make_option('--yes_train', dest='yes_train',
                    default='', help='Yes train'),
        make_option('--no_train', dest='no_train',
                    default='', help='No train'),
    )
    
    __verbose = False
    
    def __init__(self):
        self.__directory = ""

    def handle(self, *args, **options):
        yesTrain = ""
        noTrain = ""
        directory = None
        if options['directory'] :
            directory = options['directory']
        if options['yes_train'] :
            yesTrain = options['yes_train']
        if options['no_train'] :
            noTrain = options['no_train']
            
        if options['train_corpus'] :
            self.__trainCorpus(options['train_corpus'], yesTrain.split(","), noTrain.split(","))
        elif options['untrain_corpus'] :
            self.__untrainCorpus(options['untrain_corpus'])
        elif options['classify'] :
            self.__classify(options['classify'], directory)
            
            
    '''
    Trains a corpus
    '''
    def __trainCorpus(self, corpus, yesTagNames=[], noTagNames=[]):
        trainer = Trainer()
        success = trainer.train(corpus, yesTagNames=yesTagNames, noTagNames=noTagNames)
        if not success :
            print "Failed to classify document:%s " % corpus[:50]
        else :
            print "Trained document: %s" % corpus[:50]
            
    '''
    Untrains a corpus
    '''
    def __untrainCorpus(self, corpus):
        trainer = Trainer()
        
        if corpus :
            success = trainer.untrain(corpus)
            if not success :
                print "Failed to untrain corpus: %s" % corpus[:50]
            else :
                print "Untrained corpus: %s" % corpus[:50]
            
    
    def __getNextFileInDir(self, directory):
        if directory :
            files = os.listdir(directory) 
            for file in files :
                filePath = os.path.join(directory, file) 
                yield filePath
        
    
    def __classify(self, corpus, directory):
        classifier = Classifier.getClassifier()
     
        tags = classifier.classify(corpus)
        for tag in tags :
            print tag
    
        
        

    
