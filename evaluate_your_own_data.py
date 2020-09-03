import pandas as pd
from dssg_disinfo.models.prepare_your_data import prepare_your_data
import tensorflow as tf

my_data = pd.read_csv('../../../../../data/dssg-disinfo/articles_v3_50.csv') # Replace this path with path to your data
processed_data=prepare_your_data(my_data)

load_model = tf.keras.models.load_model('output/best_model.h5') # calls the trained model saved in the /output folder

# Evaluate the trained model
loss, acc = load_model.evaluate(processed_data,  my_data['label'], verbose=2)
print('Loaded model, loss: {:5.2f}%'.format(100*loss))
print('Loaded model, accuracy: {:5.2f}%'.format(100*acc))