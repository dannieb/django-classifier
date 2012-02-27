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