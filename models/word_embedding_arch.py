from model_arch import *

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath) as f:
        for line in f:
            l = len(line.split())
            word = line.split()[0]
            vector = line.split()[-embedding_dim:]
            if word in word_index:
                #n = n + 1
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix



def create_word_embd_model_arch(word_index, embedding_path, embedding_dim, bidir_num_filters=64, dense_1_filters=10, vocab_size=10000, maxlen=681, optimizer='adam'):
    


    embedding_matrix=create_embedding_matrix(filepath=embedding_path,
                                        word_index=word_index, embedding_dim=embedding_dim)
    vocab_size = len(word_index) + 1 # Adding again 1 because of reserved 0 index
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim,
                                   weights=[embedding_matrix],
                                   input_length=maxlen),
    keras.layers.Bidirectional(keras.layers.LSTM(bidir_num_filters)),
    keras.layers.Dense(dense_1_filters, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
        ])
        
    return model

# Register the basic model (writing into our dictionary of models)
register_model_arch("word_embedding", create_word_embd_model_arch,
                    ["embedding_path", "bidir_num_filters", "dense_1_filters", "vocab_size", "embedding_dim", "maxlen",   "optimizer", "word_index"])
        
