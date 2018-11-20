'''
A TestAgent using pytorch
'''

from . import BaseAgent
from .. import characters
from .. import constants
import os
import numpy as np
from .models.factory import create_policy
# from .envs import make_vec_envs
import torch
import gym
from .util import copy_obs_dict, dict_to_obs, obs_space_info
from . import HelpLearnAgent

def get_unflat_obs_space(channels=15, board_size=11, rescale=True):
    min_board_obs = np.zeros((channels, board_size, board_size))
    max_board_obs = np.ones_like(min_board_obs)
    min_other_obs = np.zeros(3)
    max_other_obs = np.ones_like(min_other_obs)

    if rescale:
        min_board_obs = _rescale(min_board_obs)
        max_board_obs = _rescale(max_board_obs)
        min_other_obs = _rescale(min_other_obs)
        max_other_obs = _rescale(max_other_obs)

    return gym.spaces.Tuple([
        gym.spaces.Box(min_board_obs, max_board_obs),
        gym.spaces.Box(min_other_obs, max_other_obs)])


def _rescale(x):
    return (x - 0.5) * 2.0

def get_observation():
	bs = 11
	channels = 9
	config = True
	obs_unflat = get_unflat_obs_space(channels, bs, config)
	min_flat_obs = np.concatenate([obs_unflat.spaces[0].low.flatten(), obs_unflat.spaces[1].low])
	max_flat_obs = np.concatenate([obs_unflat.spaces[0].high.flatten(), obs_unflat.spaces[1].high])

	return gym.spaces.Box(min_flat_obs, max_flat_obs)

def make_np_float(feature):
    return np.array(feature).astype(np.float32)

def featurize(obs, agent_id, config):
    max_item = constants.Item.Agent3.value

    ob = obs["board"]
    ob_bomb_blast_strength = obs["bomb_blast_strength"].astype(np.float32) / constants.AGENT_VIEW_SIZE
    ob_bomb_life = obs["bomb_life"].astype(np.float32) / constants.DEFAULT_BOMB_LIFE

    # one hot encode the board items
    ob_values = max_item + 1
    ob_hot = np.eye(ob_values)[ob]

    # replace agent item channels with friend, enemy, self channels
    if config['recode_agents']:
        self_value = constants.Item.Agent0.value + agent_id
        enemies = np.logical_and(ob >= constants.Item.Agent0.value, ob != self_value)
        self = (ob == self_value)
        friends = (ob == constants.Item.AgentDummy.value)
        ob_hot[:, :, 9] = friends.astype(np.float32)
        ob_hot[:, :, 10] = self.astype(np.float32)
        ob_hot[:, :, 11] = enemies.astype(np.float32)
        ob_hot = np.delete(ob_hot, np.s_[12::], axis=2)

    if config['compact_powerups']:
        # replace powerups with single channel
        powerup = ob_hot[:, :, 6] * 0.5 + ob_hot[:, :, 7] * 0.66667 + ob_hot[:, :, 8]
        ob_hot[:, :, 6] = powerup
        ob_hot = np.delete(ob_hot, [7, 8], axis=2)

    # replace bomb item channel with bomb life
    ob_hot[:, :, 3] = ob_bomb_life

    if config['compact_structure']:
        ob_hot[:, :, 0] = 0.5 * ob_hot[:, :, 0] + ob_hot[:, :, 5]  # passage + fog
        ob_hot[:, :, 1] = 0.5 * ob_hot[:, :, 2] + ob_hot[:, :, 1]  # rigid + wood walls
        ob_hot = np.delete(ob_hot, [2], axis=2)
        # replace former fog channel with bomb blast strength
        ob_hot[:, :, 5] = ob_bomb_blast_strength
    else:
        # insert bomb blast strength next to bomb life
        ob_hot = np.insert(ob_hot, 4, ob_bomb_blast_strength, axis=2)

    self_ammo = make_np_float([obs["ammo"]])
    self_blast_strength = make_np_float([obs["blast_strength"]])
    self_can_kick = make_np_float([obs["can_kick"]])

    ob_hot = ob_hot.transpose((2, 0, 1))  # PyTorch tensor layout compat

    if config['rescale']:
        ob_hot = _rescale(ob_hot)
        self_ammo = _rescale(self_ammo / 10)
        self_blast_strength = _rescale(self_blast_strength / constants.AGENT_VIEW_SIZE)
        self_can_kick = _rescale(self_can_kick)

    return np.concatenate([
        np.reshape(ob_hot, -1), self_ammo, self_blast_strength, self_can_kick])




class PytorchAgent(BaseAgent):
    """docstring for PytorchAgent"""
    def __init__(self, character=characters.Bomber):
        pass
        super(PytorchAgent, self).__init__(character)

        torch.set_num_threads(1)

        num_env = 1
        # self.device = torch.device("cpu")

        observation_space = get_observation()
        action_space = gym.spaces.Discrete(6)

        actor_critic = create_policy(
        observation_space,
        action_space,
        name='pomm',
        nn_kwargs={
           #'conv': 'conv3',
            'batch_norm': True,
            'recurrent': False,
            'hidden_size': 512,
        },
        train=False)

        load_path = "./pytorch_agent.pt"
        state_dict, ob_rms = torch.load(load_path)

        actor_critic.load_state_dict(state_dict)
        # actor_critic.to(self.device)

        recurrent_hidden_states = torch.zeros(num_env, actor_critic.recurrent_hidden_state_size)
		# print("actor_critic.recurrent_hidden_state_size", actor_critic.recurrent_hidden_state_size)
        masks = torch.zeros(num_env, 1)


        ###帮助躲避炸弹的agent
        action_space = gym.spaces.Discrete(6)
        help = HelpLearnAgent()
        self.num_envs = num_env
        self.actor_critic = actor_critic
        self.recurrent_hidden_states = recurrent_hidden_states
        self.masks = masks
        self.keys, shapes, dtypes = obs_space_info(observation_space)
        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.helpLearnAgent = help
        self.action_space = action_space

	

    def act(self, obs, action_space):

        help_action = self.helpLearnAgent.act(obs, self.action_space)
            # help_action = None
        if help_action is not None:
            return help_action
            


        feature_config = {'recode_agents': True, 'compact_powerups': True, 'compact_structure': True, 'rescale': True}
        #目前假设自己是0号Agent

        agent_state = featurize(obs, 0, feature_config)
        tmpt1 = self._save_obs(0, agent_state)
        tmpt2 = dict_to_obs(copy_obs_dict(self.buf_obs))
        obs = torch.from_numpy(tmpt2).float()
        # print("shape", obs.shape)
        #不知道reward是怎么获得的
        agent_reward = 0

        #done应该没用？
        done = [False]
        _ = [{}]

        with torch.no_grad():
            value, action, _, self.recurrent_hidden_states = self.actor_critic.act(
            obs, self.recurrent_hidden_states, self.masks, deterministic=True)


        self.masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done])
        return int(action)

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

		

