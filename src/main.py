import os
import logging
from pathlib import Path
from tfidf_ranking import EAL

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)


dir = Path(__file__).parent.parent
# set model file to your local path
model_file = Path.joinpath(dir, 'model', 'tfidf_model.pkl')

while True:
    sentence = input('enter sentence:')
    entity = input('enter entity: ')
    eal = EAL(model_file)
    aspect_predicted = eal.get_prediction(sentence, entity)
    print('Predicted most relevant aspect is: %s' % aspect_predicted)





