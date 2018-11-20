'''The base simple agent use to train agents.
This agent is also the benchmark for other agents.
'''
from collections import defaultdict
import queue
import random

import numpy as np

from . import BaseAgent
from .. import constants
from .. import utility


class DeadAgent(BaseAgent):
    """This is a baseline agent. After you can beat it, submit your agent to
    compete.
    """
    def __init__(self, *args, **kwargs):
        super(DeadAgent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):
       return 5