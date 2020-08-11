class params(object):
    """
    The RnnParams tracks the parameters for the clean_retinotopy
    function.
    """
    # PARAMETERS THAT WON't CHANGE And can be inherited by all models
    vocab_size = 10000
    maxlen=681
    # define the default parameters live in the init argument list:
    def __init__(self,
                 oov_token='<OOV>',
                 truncating='post',
                 embedding_dim=300,
                 epochs=5,
                 optimizer='adam',
                 bidir_num_filters=64,
                 dense_1_filters=10,
                 embedding_path=None
                 ):
        self.oov_token = oov_token
        self.truncating = truncating
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.optimizer = optimizer
        self.bidir_num_filters=bidir_num_filters
        self.dense_1_filters=dense_1_filters
        self.embedding_path=embedding_path
   

        
'''def clean_rnn(params):
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
params = Params()'''