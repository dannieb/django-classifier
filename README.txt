This is a general purpose text classifier.
The only implementation provided is a Naive Bayes Classifier (http://en.wikipedia.org/wiki/Naive_Bayes_classifier)

The classifier can classify any number of categories.


INSTALLATION:
Just run [setup.py install]


USAGE:
1.) Django 1.2+, NLTK (for feature extraction)
2.) Add the classifier app to your settings
3.) The classifier stores data offline so you need to build the data model. (syncdb or south)
4.) Training the classifier:
	Pass in documents and the name of the categories that the documents represents.

	[usage]
	mange.py classify_train --train_corpus "some corpus" --yes_train='spam' --no_train='ham'

	     
5.) Untraining the classifier:
	Pass in the document to untrain.
	
	[usage]
	manage.py classify_train --untrain_corpus "some corpus" 
	     
6.) Once you have a sufficient number of documents trained you can try classifying any text corpus.
	The classifier will print out all categories that it thinks best represents that corpus.

	[usage]
	manage.py classiy_train --classify "some corpus"
