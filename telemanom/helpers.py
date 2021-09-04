import logging
import yaml
import json
import sys
import os

logger = logging.getLogger('telemanom')
sys.path.append('../../../../Download/tmep/Thesis/telemanom')


class Config:
    """Loads parameters from config.yaml into global object

    """

    def __init__(self, path_to_config):

        self.execution = None
        self.path_to_config = path_to_config

        if os.path.isfile(path_to_config):
            pass
        else:
            self.path_to_config = '../{}'.format(self.path_to_config)

        with open(self.path_to_config, "r") as f:
            self.dictionary = yaml.load(f.read(), Loader=yaml.FullLoader)

        for k, v in self.dictionary.items():
            setattr(self, k, v)

    def build_group_lookup(self, path_to_groupings):

        channel_group_lookup = {}

        with open(path_to_groupings, "r") as f:
            groupings = json.loads(f.read())

            for subsystem in groupings.keys():
                for subgroup in groupings[subsystem].keys():
                    for chan in groupings[subsystem][subgroup]:
                        channel_group_lookup[chan["key"]] = {}
                        channel_group_lookup[chan["key"]]["subsystem"] = subsystem
                        channel_group_lookup[chan["key"]]["subgroup"] = subgroup

        return channel_group_lookup


def make_dirs(_id):
    '''Create directories for storing data in repo (using datetime ID) if they don't already exist'''

    config = Config("config.yaml")

    #if not config.train or not config.predict:
    if config.execution == "predict" or config.execution == "search_p":
        if not os.path.isdir('data/%s' %config.use_id) or config.use_id == "":
            raise ValueError("Run ID {} is not valid. If loading prior models or predictions, must provide valid ID.".format(_id))

    if config.execution == "find_hp":
        if _id == "":
            raise ValueError(
                "Run ID is empty. Provide valid ID."
                    .format(_id))
        paths = ['hp', 'hp/%s' % _id, 'hp/%s/config' % _id, 'hp/%s/kerastuner' % _id]

    if config.execution != "find_hp":
        paths = ['data', 'data/%s' %_id, 'data/logs', 'data/%s/images' %_id, 'data/%s/models' %_id, 'data/%s/smoothed_errors' %_id, 'data/%s/y_hat' %_id,
             'hp']

    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)

def setup_logging():
    '''Configure logging object to track parameter settings, training, and evaluation.
    
    Args:
        config(obj): Global object specifying system runtime params.

    Returns:
        logger (obj): Logging object
        _id (str): Unique identifier generated from datetime for storing data/models/results
    '''

    logger = logging.getLogger('telemanom')
    logger.setLevel(logging.INFO)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    logger.addHandler(stdout)

    return logger