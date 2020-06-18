from datetime import datetime
import io
from zipfile import ZipFile
import dill
import numpy
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import yaml

TOPIC = 'coronavirus'

def persist_classifier(filename, classifier, transformer, version=None):
    """Converts an sklearn classifier object to ONNX format

    Parameters:
    filename (string): name of output zipfile
    classifier (*Classifier): sklearn classifier object
    transformer (*Vectorizer): sklearn vectorizer object
    version (datetime): optional versioning by date

   """
    vocabulary = {str(k): int(v) for k, v in transformer.vocabulary_.items()}
    vocabulary_size = len(vocabulary)
    onnx = convert_sklearn(
        classifier,
        initial_types = [
            ('text', FloatTensorType([None, vocabulary_size])),
        ],
        target_opset = 11,
    )
    if version is None:
        version = datetime.now().isoformat()

    with ZipFile(filename, 'w') as zfh:
        with zfh.open('vocabulary.yaml', 'w') as fh:
            yaml.dump(vocabulary, fh, encoding="utf-8")
        with zfh.open('idf.npz', 'w') as fh:
            idf = transformer._tfidf._idf_diag
            save_npz(fh, idf)
        with zfh.open('classifier.onnx', 'w') as fh:
            fh.write(onnx.SerializeToString())
        with zfh.open('VERSION', 'w') as fh:
            fh.write(version.encode("utf-8"))


def run():
    """Convert pickled, trained classifier and vectorizer objects 
    into zipfile of ONNX objects for deployment

   """
    with open(TOPIC+'_SGD_count_vect.pkl', 'rb') as fh:
        vectorizer = dill.load(fh)
    with open(TOPIC+'_SGD_clf.pkl', 'rb') as fh:
        classifier = dill.load(fh)
    persist_classifier(TOPIC+'_SGD.classifier', classifier, vectorizer)


if __name__ == "__main__":
    run()
