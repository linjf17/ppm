"""The baseline Pommerman environment.

This evironment acts as game manager for Pommerman. Further environments,
such as in v1.py, will inherit from this.
"""
import json
import os

import numpy as np
import time
from gym import spaces
from gym.utils import seeding
import gym

from .. import characters
from .. import constants
from .. import forward_model
from .. import graphics
from .. import utility

count = 0
win = 0
lose = 0
tie = 0

filename = './write_data.txt'
# with open(filename,'w') as f:
#     f.write("result\n")
#     f.close()


class Pomme(gym.Env):
    '''The base pommerman env.'''
    metadata = {
        'render.modes': ['human', 'rgb_array', 'rgb_pixel'],
    }

    def __init__(self,
                 render_fps=None,
                 game_type=None,
                 board_size=None,
                 agent_view_size=None,
                 num_rigid=None,
                 num_wood=None,
                 num_items=None,
                 max_steps=1000,
                 is_partially_observable=False,
                 env=None,
                 **kwargs):
        self._render_fps = render_fps
        self._intended_actions = []
        self._agents = None
        self._game_type = game_type
        self._board_size = board_size
        self._agent_view_size = agent_view_size
        self._num_rigid = num_rigid
        self._num_wood = num_wood
        self._num_items = num_items
        self._max_steps = max_steps
        self._viewer = None
        self._is_partially_observable = is_partially_observable
        self._env = env
        self.alive_enemy_num = 3

        self.training_agent = None
        self.model = forward_model.ForwardModel()

        # This can be changed through set_render_mode
        # or from the cli tool using '--render_mode=MODE_TYPE'
        self._mode = 'human'

        # Observation and Action Spaces. These are both geared towards a single
        # agent even though the environment expects actions and returns
        # observations for all four agents. We do this so that it's clear what
        # the actions and obs are for a single agent. Wrt the observations,
        # they are actually returned as a dict for easier understanding.
        self._set_action_space()
        self._set_observation_space()

    def _set_action_space(self):
        self.action_space = spaces.Discrete(6)

    def set_render_mode(self, mode):
        self._mode = mode

    def _set_observation_space(self):
        """The Observation Space for each agent.

        There are a total of 3*board_size^2+12 observations:
        - all of the board (board_size^2)
        - bomb blast strength (board_size^2).
        - bomb life (board_size^2)
        - agent's position (2)
        - player ammo counts (1)
        - blast strength (1)
        - can_kick (1)
        - teammate (one of {AgentDummy.value, Agent3.value}).
        - enemies (three of {AgentDummy.value, Agent3.value}).
        """
        bss = self._board_size**2
        min_obs = [0] * 3 * bss + [0] * 5 + [constants.Item.AgentDummy.value
                                            ] * 4
        max_obs = [len(constants.Item)] * bss + [self._board_size
                                                ] * bss + [25] * bss
        max_obs += [self._board_size] * 2 + [self._num_items] * 2 + [1]
        max_obs += [constants.Item.Agent3.value] * 4
        self.observation_space = spaces.Box(
            np.array(min_obs), np.array(max_obs))

    def set_agents(self, agents):
        self._agents = agents

    def set_training_agent(self, agent_id):
        self.training_agent = agent_id

    def set_init_game_state(self, game_state_file):
        """Set the initial game state.

        The expected game_state_file JSON format is:
          - agents: list of agents serialized (agent_id, is_alive, position,
            ammo, blast_strength, can_kick)
          - board: board matrix topology (board_size^2)
          - board_size: board size
          - bombs: list of bombs serialized (position, bomber_id, life,
            blast_strength, moving_direction)
          - flames: list of flames serialized (position, life)
          - items: list of item by position
          - step_count: step count

        Args:
          game_state_file: JSON File input.
        """
        self._init_game_state = None
        if game_state_file:
            with open(game_state_file, 'r') as f:
                self._init_game_state = json.loads(f.read())

    def make_board(self):
        self._board = utility.make_board(self._board_size, self._num_rigid,
                                         self._num_wood)

    def make_items(self):
        self._items = utility.make_items(self._board, self._num_items)

    def act(self, obs):
        agents = [agent for agent in self._agents \
                  if agent.agent_id != self.training_agent]
        return self.model.act(agents, obs, self.action_space)

    def get_observations(self):
        self.observations = self.model.get_observations(
            self._board, self._agents, self._bombs,
            self._is_partially_observable, self._agent_view_size,
            self._game_type, self._env)
        return self.observations

    def _get_rewards(self):
        global win
        global tie
        global lose
        global count

        alive = [agent for agent in self._agents if agent.is_alive]
        alive_ids = sorted([agent.agent_id for agent in alive])
        if self.training_agent is not None :
            if self.training_agent not in alive_ids:
                print("lose")
                count+=1
                lose+=1
                if count%100 == 0:
                    print("win rate with tie: ", win/count)
                    print("win rate without tie: ", win/(count-tie))
                    print("win: ", win, " lose: ", lose, " tie: ", tie)
                return [-80]*4
            if self._step_count >= self._max_steps:
                print("tie")
                count+=1
                tie+=1
                if count%100 == 0:
                    print("win rate with tie: ", win/count)
                    print("win rate without tie: ", win/(count-tie))
                    print("win: ", win, " lose: ", lose, " tie: ", tie)
                return [-20]*4

            if len(alive_ids) == 1:
                print("win")
                count+=1
                win+=1
                if count%100 == 0:
                    print("win rate with tie: ", win/count)
                    print("win rate without tie: ", win/(count-tie))
                    print("win: ", win, " lose: ", lose, " tie: ", tie)
                return[150]*4
            if len(alive_ids) <= self.alive_enemy_num:
                self.alive_enemy_num-=1
                return[30]*4
            reward = 0
            return [reward,0,0,0]
        # print("qnmb")
        return self.model.get_rewards(self._agents, self._game_type,
                                      self._step_count, self._max_steps)

    def _get_done(self):
        return self.model.get_done(self._agents, self._step_count,
                                   self._max_steps, self._game_type,
                                   self.training_agent)

    def _get_info(self, done, rewards):
        return self.model.get_info(done, rewards, self._game_type, self._agents)

    def reset(self):
        assert (self._agents is not None)

        if self._init_game_state is not None:
            self.set_json_info()
        else:
            self._step_count = 0
            self.make_board()
            self.make_items()
            self._bombs = []
            self._flames = []
            self._powerups = []
            for agent_id, agent in enumerate(self._agents):
                pos = np.where(self._board == utility.agent_value(agent_id))
                row = pos[0][0]
                col = pos[1][0]
                agent.set_start_position((row, col))
                agent.reset()

        return self.get_observations()

    def seed(self, seed=None):
        gym.spaces.prng.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        self._intended_actions = actions

        if self.training_agent is not None :
            pre_board = self.observations[self.training_agent]['board'].copy()
            pre_bombs = self.observations[self.training_agent]['bomb_blast_strength'].copy()
            pre_bomb_life = self.observations[self.training_agent]['bomb_life'].copy()
            can_kick = self.observations[self.training_agent]['can_kick']
            blast_strength = self.observations[self.training_agent]['blast_strength']
            ammo = self.observations[self.training_agent]['ammo']

        max_blast_strength = self._agent_view_size or 10
        result = self.model.step(
            actions,
            self._board,
            self._agents,
            self._bombs,
            self._items,
            self._flames,
            max_blast_strength=max_blast_strength)
        self._board, self._agents, self._bombs, self._items, self._flames = \
                                                                    result[:5]

        done = self._get_done()
        obs = self.get_observations()
        reward = self._get_rewards()
        info = self._get_info(done, reward)


        if self.training_agent is not None :
            pos = np.where(pre_board == self.training_agent+10)
            alive_now = np.where(self.observations[self.training_agent]['board'] == self.training_agent+10)
            if len(pos[0]) > 0 and len(alive_now[0]) > 0:
                row = pos[0][0]
                col = pos[1][0]
                x = [0, -1, 1, 0, 0, 0]
                y = [0, 0, 0, -1, 1, 0]


                if row + x[actions[self.training_agent]] < 0 or row + x[actions[self.training_agent]] > 10 or col + y[actions[self.training_agent]] < 0 or col + y[actions[self.training_agent]] > 10:
                    reward[self.training_agent] -= 1
                elif pre_board[row + x[actions[self.training_agent]]][col + y[actions[self.training_agent]]] == 4:
                    reward[self.training_agent] -= 10
                elif pre_board[row + x[actions[self.training_agent]]][col + y[actions[self.training_agent]]] == 1:
                    reward[self.training_agent] -= 1
                elif pre_board[row + x[actions[self.training_agent]]][col + y[actions[self.training_agent]]] == 3 and can_kick:
                    reward[self.training_agent] += 2

                if actions[self.training_agent] == 5 and ammo > 0:
                    reward[self.training_agent] += 2

                    bomb_map = obs[self.training_agent]['bomb_blast_strength'].copy()
                    def cross(x, y, radius):
                        for xx in range(x-radius, x+radius+1):
                            if 0 <= xx < 11 and bomb_map[xx][y] == 0:
                                bomb_map[xx][y] = 1
                        for yy in range(y-radius, y+radius+1):
                            if 0 <= yy < 11 and bomb_map[x][yy] == 0:
                                bomb_map[x][yy] = 1

                    for x in range(11):
                        for y in range(11):
                            if (bomb_map[x][y] > 1):
                                cross(x, y, int(bomb_map[x][y]-1))

                    distance = 5

                    tmpt_id = 11#假设离我们最近的是11#注意训练的agent必须是第一个
                    enemy_x = -1
                    enemy_y = -1
                    while 1:
                        for i in range(-distance,distance+1):#help agent里面如果采取进攻模式那么敌人一定离我们四步之内
                            for j in range(-distance,distance+1):
                                if 0 <= row+i <= 10 and 0 <= col+j <= 10:
                                    if self._board[row+i][col+j] == tmpt_id:
                                        enemy_x = row+i
                                        enemy_y = col+j
                                        break
                        tmpt_id += 1
                        if tmpt_id >= 14 :
                            break

                    if enemy_x != -1:
                        enemy_id = tmpt_id
                        n = 2
                        floodfill_area = 1#防止最后除以0
                        safety_area = 0

                        enemy_row = enemy_x
                        enemy_col = enemy_y

                        
                        for i in range(0, n+1):
                            for j in range(-i, 1+i):
                                target_row, target_col = enemy_row - n + i, enemy_col + j
                                target_position = (target_row, target_col)
                                if -1<target_col<11 and -1<target_row<11:
                                    # safe
                                    if self._board[target_row][target_col] in [0, 5, 6, 7, 8, 9]:
                                        floodfill_area += 1
                                        if (bomb_map[target_row][target_col] != 1):
                                            safety_area += 1
                        for i in range(1, n+1):
                            for j in range(-2+i, 3-i):
                                target_row, target_col = enemy_row + i, enemy_col + j
                                target_position = (target_row, target_col)
                                if -1<target_col<11 and -1<target_row<11:
                                    # safe
                                    if self._board[target_row][target_col] in [0, 5, 6, 7, 8, 9]:
                                        floodfill_area += 1
                                        if (bomb_map[target_row][target_col] != 1):
                                            safety_area += 1


                        reward[self.training_agent] += 6*(1 - safety_area / floodfill_area)



                    t_alive = [agent for agent in self._agents if agent.is_alive]
                    t_alive_ids = sorted([agent.agent_id for agent in t_alive])

                    x1 = row
                    y1 = col
                    while 1: 
                        y1 -= 1

                        if y1 > 10 or y1 < 0 or col - y1 >= blast_strength:
                            break
                        if (pre_board[x1][y1]-10) in t_alive_ids:
                            reward[self.training_agent] += 2
                            break
                        elif pre_board[x1][y1] not in [0, 4]:
                            break

                    x2 = row
                    y2 = col
                    while 1:
                        y2 += 1

                        if y2 > 10 or y2 < 0 or y2 - col>= blast_strength:
                            break
                        if (pre_board[x2][y2]-10) in t_alive_ids:
                            reward[self.training_agent] += 2
                            break
                        elif pre_board[x2][y2] not in [0, 4]:
                            break

                    x3 = row
                    y3 = col
                    while 1:
                        x3 -= 1

                        if x3 > 10 or x3 < 0 or row - x3 >= blast_strength:
                            break
                        if (pre_board[x3][y3]-10) in t_alive_ids:
                            reward[self.training_agent] += 2
                            break
                        elif pre_board[x3][y3] not in [0, 4]:
                            break
                    
                    x4 = row
                    y4 = col
                    while 1:

                        x4 += 1

                        if x4 > 10 or x4 < 0 or x4 - row >= blast_strength:
                            break
                        if (pre_board[x4][y4]-10) in t_alive_ids:
                            reward[self.training_agent] += 2
                            break
                        elif pre_board[x4][y4] not in [0, 4]:
                            break
                    



                    


        if done:
            # Callback to let the agents know that the game has ended.
            for agent in self._agents:
                agent.episode_end(reward[agent.agent_id])

        self._step_count += 1
        return obs, reward, done, info

    def render(self,
               mode=None,
               close=False,
               record_pngs_dir=None,
               record_json_dir=None,
               do_sleep=True):
        if close:
            self.close()
            return

        mode = mode or self._mode or 'human'

        if mode == 'rgb_array':
            rgb_array = graphics.PixelViewer.rgb_array(
                self._board, self._board_size, self._agents,
                self._is_partially_observable, self._agent_view_size)
            return rgb_array[0]

        if self._viewer is None:
            if mode == 'rgb_pixel':
                self._viewer = graphics.PixelViewer(
                    board_size=self._board_size,
                    agents=self._agents,
                    agent_view_size=self._agent_view_size,
                    partially_observable=self._is_partially_observable)
            else:
                self._viewer = graphics.PommeViewer(
                    board_size=self._board_size,
                    agents=self._agents,
                    partially_observable=self._is_partially_observable,
                    agent_view_size=self._agent_view_size,
                    game_type=self._game_type)

            self._viewer.set_board(self._board)
            self._viewer.set_agents(self._agents)
            self._viewer.set_step(self._step_count)
            self._viewer.render()

            # Register all agents which need human input with Pyglet.
            # This needs to be done here as the first `imshow` creates the
            # window. Using `push_handlers` allows for easily creating agents
            # that use other Pyglet inputs such as joystick, for example.
            for agent in self._agents:
                if agent.has_user_input():
                    self._viewer.window.push_handlers(agent)
        else:
            self._viewer.set_board(self._board)
            self._viewer.set_agents(self._agents)
            self._viewer.set_step(self._step_count)
            self._viewer.render()

        if record_pngs_dir:
            self._viewer.save(record_pngs_dir)
        if record_json_dir:
            self.save_json(record_json_dir)

        if do_sleep:
            time.sleep(1.0 / self._render_fps)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

        for agent in self._agents:
            agent.shutdown()

    @staticmethod
    def featurize(obs):
        board = obs["board"].reshape(-1).astype(np.float32)
        bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1) \
                                                        .astype(np.float32)
        bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
        position = utility.make_np_float(obs["position"])
        ammo = utility.make_np_float([obs["ammo"]])
        blast_strength = utility.make_np_float([obs["blast_strength"]])
        can_kick = utility.make_np_float([obs["can_kick"]])

        teammate = utility.make_np_float([obs["teammate"].value])
        enemies = utility.make_np_float([e.value for e in obs["enemies"]])
        return np.concatenate(
            (board, bomb_blast_strength, bomb_life, position, ammo,
             blast_strength, can_kick, teammate, enemies))

    def save_json(self, record_json_dir):
        info = self.get_json_info()
        count = "{0:0=3d}".format(self._step_count)
        suffix = count + '.json'
        path = os.path.join(record_json_dir, suffix)
        with open(path, 'w') as f:
            f.write(json.dumps(info, sort_keys=True, indent=4))

    def get_json_info(self):
        """Returns a json snapshot of the current game state."""
        ret = {
            'board_size': self._board_size,
            'step_count': self._step_count,
            'board': self._board,
            'agents': self._agents,
            'bombs': self._bombs,
            'flames': self._flames,
            'items': [[k, i] for k, i in self._items.items()],
            'intended_actions': self._intended_actions
        }
        for key, value in ret.items():
            ret[key] = json.dumps(value, cls=utility.PommermanJSONEncoder)
        return ret

    def set_json_info(self):
        """Sets the game state as the init_game_state."""
        board_size = int(self._init_game_state['board_size'])
        self._board_size = board_size
        self._step_count = int(self._init_game_state['step_count'])

        board_array = json.loads(self._init_game_state['board'])
        self._board = np.ones((board_size, board_size)).astype(np.uint8)
        self._board *= constants.Item.Passage.value
        for x in range(self._board_size):
            for y in range(self._board_size):
                self._board[x, y] = board_array[x][y]

        self._items = {}
        item_array = json.loads(self._init_game_state['items'])
        for i in item_array:
            self._items[tuple(i[0])] = i[1]

        agent_array = json.loads(self._init_game_state['agents'])
        for a in agent_array:
            agent = next(x for x in self._agents \
                         if x.agent_id == a['agent_id'])
            agent.set_start_position((a['position'][0], a['position'][1]))
            agent.reset(
                int(a['ammo']), bool(a['is_alive']), int(a['blast_strength']),
                bool(a['can_kick']))

        self._bombs = []
        bomb_array = json.loads(self._init_game_state['bombs'])
        for b in bomb_array:
            bomber = next(x for x in self._agents \
                          if x.agent_id == b['bomber_id'])
            moving_direction = b['moving_direction']
            if moving_direction is not None:
                moving_direction = constants.Action(moving_direction)
            self._bombs.append(
                characters.Bomb(bomber, tuple(b['position']), int(b['life']),
                                int(b['blast_strength']), moving_direction))

        self._flames = []
        flame_array = json.loads(self._init_game_state['flames'])
        for f in flame_array:
            self._flames.append(
                characters.Flame(tuple(f['position']), f['life']))
