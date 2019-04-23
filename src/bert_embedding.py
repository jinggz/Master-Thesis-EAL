# The goal of this code is to connect to bert-as-service api, and get the embedded vector of input list of
# BertServer config:
# max_seq_len=None
# num_worker=2
# max_batch_size=16
# ZEROMQ_SOCK_TMP_DIR : export ZEROMQ_SOCK_TMP_DIR=/tmp/

import logging
from bert_serving.client import BertClient

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)

class BertEmbedd:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def get_connection(self, inport=8000, outport=8010):
        '''
        Sets up a connection with the specified BERT server api.
        :param inport: Port for pushing data from client to server (defaults to 8000)
        :param outport: Port for publishing results from server to client (defaults to 8010)
        :return: Bert client object connected to a BertServer
        '''
        try:
            self.connection = BertClient(port=inport, port_out=outport)
        except Exception as e:
            logger.error(f'Connection to BERT server failed: {str(e)}')

        logger.info("Connection to BERT server was successful.")

    def close_connection(self):
        if self.connection:
           self.connection.close()

    def get_encode(self, data, istokenized=True, isblocked=True):
        '''
        :param inport: the port where bert service located
        :param outport: the port where the results get back
        :param sentences: list of sentences, (preprocessed, tokenized-rejoin)
        :return:
        encoded sentence/token-level embeddings, rows correspond to sentences
        :type:
        numpy.ndarray or list[list[float]]
        '''

        logger.info('sending new request...')
        try:
            # encode tokenized sentences
            result = self.connection.encode(data, blocking=isblocked, is_tokenized=istokenized)
            logger.info('encoding job done')
        except Exception as e:
            logger.error(f'getting encodes from BERT failed: {str(e)}')
            return []
        return result


