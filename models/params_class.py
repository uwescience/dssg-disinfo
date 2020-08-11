class Params(object):
    """
    The RnnParams tracks the parameters for the clean_retinotopy
    function.
    """
    # PARAMETERS THAT WON't CHANGE And can be inherited by all models
    and is immutable
    vocab_size = 10000
    max_length=681
    # define the default parameters live in the init argument list:
    def __init__(self,
                 oov_token='<OOV>',
                 truncating='post',
                 embedding_dim=300,
                 input_length=681,
                 epochs=5,
                 optimizer='adam',
                 ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.oov_token = oov_token
        self.truncating = truncating
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.epochs = epochs
        self.optimizer = optimizer
   

        
def clean_rnn(params):
    """ ... """
    vocab_size = params.vocab_size
    max_length = params.max_length
    oov_token = params.oov_token
    truncating = params.truncating
    embedding_dim = params.embedding_dim
    input_length = params.input_length
    epochs = params.epochs
    optimizer = params.optimizer
    return vocab_size, max_length, oov_token, truncating, embedding_dim, input_length, epochs, optimizer

# instantiates the params object
params = Params()