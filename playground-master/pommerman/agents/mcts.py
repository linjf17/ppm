#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import queue
import random
import numpy as np
from . import BaseAgent
import sys
import math
import time
from .. import constants
from .. import utility

class State(object):
    """
    蒙特卡罗树搜索的游戏状态，记录在某一个Node节点下的状态数据，包含当前的游戏得分、当前的游戏round数、从开始到当前的执行记录。
    需要实现判断当前状态是否达到游戏结束状态，支持从Action集合中随机取出操作。
    """

    def __init__(self, obs, mode, id):
        self.current_value = 0.0
        # For the first root node, the index is 0 and the game should start from 1
        self.move = 0  # 用来记录这个state自己是怎么动的（譬如向上还是放炸弹
        self.obs = obs
        self.can_kick = obs['can_kick']
        self.my_position = tuple(obs['position'])
        self.board = np.array(obs['rem_board'])
        self.flame_board = obs['flame_board']
        self.bombs = np.array(obs['rem_bom'])  # bombs是爆炸半径
        self.bombs_life_time = np.array(obs['rem_bom_life'])
        self.mode = mode  # 0是逃跑，1是进攻
        self.target_enemy = None
        self.blast_strength = obs['blast_strength']
        self.my_id = id
        if id in [10, 12]:
        	self.enemies = [11, 13]
        else:
        	self.enemies = [10, 12]


    def set_target_enemy(self, enemy):
        self.target_enemy = enemy

    def get_available_action(self):
        return self.available_action(self.board)

    def position_is_passable(self, board, flame_board, position):
        x, y = position
        if any([len(board) <= x, len(board[0]) <= y, x < 0, y < 0]):
            return False
        availabe_choice = [0, 5, 6, 7, 8, 9]
        if board[x, y] in availabe_choice or flame_board[x][y] == 1:
            return True
        else:
            return False

    def available_action(self, board):

        def is_stuck_direction(next_position, next_board, enemies, blast_strength, my_position):
            bomb_range = blast_strength
            '''Helper function to do determine if the agents next move is possible.'''
            Q = queue.PriorityQueue()
            Q.put((1, next_position))
            seen = set()

            my_x, my_y = my_position

            tmpt_board = next_board.copy()
            tmpt_board[my_x][my_y] = 1

            is_stuck = True
            while not Q.empty():
                dist, position = Q.get()
                seen.add(position)

                position_x, position_y = position
                if my_x != position_x and my_y != position_y:
                    is_stuck = False
                    break

                if dist > bomb_range:
                    is_stuck = False
                    break

                for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row + position_x, col + position_y)
                    if new_position in seen:
                        continue

                    if not self.position_is_passable(tmpt_board,self.flame_board,
                                                     new_position):
                        continue

                    dist = abs(row + position_x - my_x) + abs(col + position_y - my_y)
                    Q.put((dist, new_position))
            return is_stuck

        row, col = self.my_position
        actions = []#因为现在进入evade的条件是周围可能被炸到，因此不动也是选项之一
        if self.mode == 1:
            erow, ecol = self.target_enemy[0]
            canKill = abs(row-erow+col-ecol) < self.blast_strength
            if (row == erow or col == ecol) and canKill:
                actions.append(5)
  
        i = 0
        for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            i += 1
            new_position = (row + drow, col + dcol)

            next_board = self.board.copy()
            blast_strength = 0

            if self.bombs[row, col] != 0:
                blast_strength = self.bombs[row, col] - 1  # 因为爆炸为2的时候实际上只涉及周围一格
                next_board[self.my_position] = constants.Item.Bomb.value

            if self.position_is_passable(next_board,self.flame_board, new_position):
                if not is_stuck_direction(new_position, next_board, self.enemies, blast_strength, self.my_position):
                    actions.append(i)
        if not actions:
            actions.append(random.randint(0, 4))

        return actions

    def convert_bombs(self, bomb_map):
        '''Flatten outs the bomb array'''
        ret = []
        locations = np.where(bomb_map > 0)
        for r, c in zip(locations[0], locations[1]):
            ret.append({
                'position': (r, c),
                'blast_strength': int(bomb_map[(r, c)])
            })
        return ret

    def convert_bombs_life_time(self, bomb_map):
        '''Flatten outs the bomb array'''
        ret = []
        locations = np.where(bomb_map > 0)
        for r, c in zip(locations[0], locations[1]):
            ret.append({
                'position': (r, c),
                'bombs_life_time': int(bomb_map[(r, c)])
            })
        return ret

    def get_current_obs(self):
        return self.obs

    def get_current_board(self):
        return self.board

    def set_current_board(self, board):
        self.board = board

    def get_current_bombs(self):
        return self.bombs

    def set_current_bombs(self, bombs):
        self.bombs = bombs

    def get_current_bombs_life_time(self):
        return self.bombs_life_time

    def set_current_bombs_life_time(self, bombs_life_time):
        self.bombs_life_time = bombs_life_time

    def get_current_position(self):
        return self.my_position

    def set_current_position(self, position):
        self.position = position

    def get_current_value(self):
        return self.current_value

    def set_current_value(self, value):
        self.current_value = value

    def get_current_move(self):
        return self.move

    def get_current_mode(self):
        return self.mode

    def set_current_move(self, move):
        self.move = move

    def is_terminal(self):
        alive = [False for i in range(4)]
        check = np.where(self.board == self.my_id)
        if check[0].size == 0:
            return 1  # 自己已经被干掉了
        dead_enemy = 0
        for j in self.enemies:
	        tmptId = j
	        check = np.where(self.get_current_board() == tmptId)
	        if check[0].size == 0:
	            dead_enemy += 1
        if dead_enemy == 2:
            return 2  # 说明其他人都被干掉了
        return 3  # 游戏继续

    def compute_reward(self):

        if self.is_terminal() == 1:
            return -40
        if self.is_terminal() == 2:
            return 150
        if self.get_current_mode() == 0:

            total = 0
            x, y = self.my_position
            tmptBomb = self.convert_bombs(self.bombs)
            for bomb in tmptBomb:
                position = bomb['position']
                bomb_range = bomb['blast_strength']
                x1, y1 = position
                distance = abs(x1 - x + y1 - y)
                if distance >= bomb_range or self.get_current_bombs_life_time()[x1][y1] > 9:
                    continue

                can_be_kill = True
                if x == x1 or y == y1:
                    if x == x1:
                        if y >= y1:
                            for i in range(y1, y):
                                if self.board[x][i] not in [0, 3, 4]:
                                    can_be_kill = False
                        else:
                            for i in range(y, y1):
                                if self.board[x][i] not in [0, 3, 4]:
                                    can_be_kill = False
                        if can_be_kill:
                            total += 25 * (10 - self.get_current_bombs_life_time()[x1][y1]) / (4*(distance+1))
                    else:
                        if x >= x1:
                            for i in range(x1, x):
                                if self.board[i][y] not in [0, 3, 4]:
                                    can_be_kill = False
                        else:
                            for i in range(x, x1):
                                if self.board[i][y] not in [0, 3, 4]:
                                    can_be_kill = False
                        if can_be_kill:
                            total += 25 * (10 - self.get_current_bombs_life_time()[x1][y1]) / (4*(distance+1))


            # print("original", (100-total), "free", ((100-total)*free/90), "now", (100 - total + (100-total)*free))
            #根据evade的得分的规模来决定加上多少free?
            # return ((100 - total)*(1+free/2))
            return (100-total)
            
        else:
            # total = 0
            # x, y = self.my_position
            # tmptBomb = self.convert_bombs(self.bombs)
            # for bomb in tmptBomb:
            #     position = bomb['position']
            #     bomb_range = bomb['blast_strength']
            #     x1, y1 = position
            #     distance = abs(x1 - x + y1 - y)
            #     if distance >= bomb_range or self.get_current_bombs_life_time()[x1][y1] > 5:
            #         continue

            #     if x == x1 or y == y1:
            #         total += 25 * (6 - self.get_current_bombs_life_time()[x1][y1]) /6#进攻的时候只会管5以下的炸弹
            # attack
            bomb_map = self.bombs.copy()
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


            enemy, n, enemy_id = self.target_enemy
            n = 2
            floodfill_area = 1#防止最后除以0
            safety_area = 0
            enemy_row, enemy_col = enemy
            enemy_down = True
            for i in range(11):
                for j in range(11):
                    if self.board[i][j] == enemy_id:
                        enemy_row = i
                        enemy_col = j
                        enemy_down = False

            if enemy_down:
                return 100#干掉了我们的目标敌人

            
            for i in range(0, 2 * n):
                for j in range(0, 2 * n):
                    target_row, target_col = enemy_row - n + i, enemy_col - n + j
                    target_position = (target_row, target_col)
                    if abs(i - n) + abs(j - n) <= n and -1<target_col<11 and -1<target_row<11:
                        
                        # safe
                        if self.position_is_passable(self.board, self.flame_board, target_position):
                            floodfill_area += 1
                            if (bomb_map[target_row][target_col] != 1):
                                safety_area += 1
            return ((100) * (1 - safety_area / floodfill_area))

    def get_next_state_with_random_choice(self):
        # print(self.obs)
        actions = [0, 0, 0, 0]
        actions[self.my_id-10] = random.choice(self.get_available_action())
        for i in range(10, 14):
            if i != self.my_id:
                actions[i-10] = (random.randint(0, 5))

        my_id = self.my_id
        tmpt_board = self.board.copy()    
        tmpt_bombs = self.bombs.copy()
        tmpt_flame_board = self.flame_board.copy()
        tmpt_bombs_life_time = self.bombs_life_time.copy()


        #判断是否会bounceback
        bounce_back = [False for i in range(4)]
        next_pos = [(-1, -1) for i in range(4)]
        tmpt_id =  10
        for action in actions:
            find_agent = np.where(tmpt_board == tmpt_id)
            if find_agent[0].size > 0:
                agentX = find_agent[0][0]
                agentY = find_agent[1][0]
                next_pos[tmpt_id-10] = (agentX,agentY)
                if action in [0, 1, 2, 3, 4]:
                    x = [0, -1, 1, 0, 0]
                    y = [0, 0, 0, -1, 1]
                    if (-1 < agentX + x[action] < 11) and (-1 < agentY + y[action] < 11):
                        next_pos[tmpt_id-10] = (agentX + x[action],agentY + y[action])
            tmpt_id += 1

        for i in range(3):
            if next_pos[i] == (-1,-1):
                bounce_back[i] = True#如果这个agent死了，就当bounceback
                continue
            for j in range(i + 1, 4):
                if next_pos[i] == next_pos[j]:
                    bounce_back[i] = True
                    bounce_back[j] = True

        # 先走再更新炸弹
        id = 10
        next_state = State(self.get_current_obs(), self.get_current_mode(), self.my_id)
        for action in actions:
            find_agent = np.where(tmpt_board == id)
            if find_agent[0].size > 0:
                agentX = find_agent[0][0]
                agentY = find_agent[1][0]
                if (id == 10):
                    next_state.my_position = (agentX, agentY)
                if action == 5:
                    tmpt_bombs[agentX][agentY] = self.blast_strength  # 注意默认的爆炸半径是2
                    tmpt_bombs_life_time[agentX][agentY] = 10
                else:
                    if not bounce_back[id-10]:
                        x,y = next_pos[id-10]
                        if tmpt_board[x][y] == 0:
                            tmpt_board[x][y] = id
                            tmpt_board[agentX][agentY] = 0
            id += 1

        #（连环）引爆函数
        visited = set()
        def cross_fire(x, y, radius):
            visited.add((x,y))
            tmpt_flame_board[x][y] = 4
            for i in range(1, radius+1):
                xx = x - i
                if 0 <= xx < 11:
                    if (tmpt_board[xx][y] in (1,5)):
                        break
                    elif tmpt_board[xx][y] == 3:
                        if  (xx,y) not in visited:
                            cross_fire(xx, y, int(tmpt_bombs[xx][y]-1))
                        break
                    elif tmpt_board[xx][y] in (0,4):
                        tmpt_flame_board[xx][y] = 4
                    else:
                        tmpt_flame_board[xx][y] = 4
                        break

            for i in range(1, radius+1):
                xx = x + i
                if 0 <= xx < 11:
                    if (tmpt_board[xx][y] in (1,5)):
                        break
                    elif tmpt_board[xx][y] == 3:
                        if  (xx,y) not in visited:
                            cross_fire(xx, y, int(tmpt_bombs[xx][y]-1))
                        break
                    elif tmpt_board[xx][y] in (0,4):
                        tmpt_flame_board[xx][y] = 4
                    else:
                        tmpt_flame_board[xx][y] = 4
                        break

            for i in range(1, radius+1):
                yy = y - i
                if 0 <= yy < 11:
                    if (tmpt_board[x][yy] in (1, 5)):
                        break
                    elif  tmpt_board[x][yy] == 3:
                        if (x, yy) not in visited:
                            cross_fire(x, yy, int(tmpt_bombs[x][yy]-1))
                        break
                    elif tmpt_board[x][yy] in (0,4):
                        tmpt_flame_board[x][yy] = 4
                    else:
                        tmpt_flame_board[x][yy] = 4
                        break

            for i in range(1, radius+1):
                yy = y + i
                if 0 <= yy < 11:
                    if (tmpt_board[x][yy] in (1,5)):
                        break
                    elif  tmpt_board[x][yy] == 3:
                        if (x, yy) not in visited:
                            cross_fire(x, yy, int(tmpt_bombs[x][yy]-1))
                        break
                    elif tmpt_board[x][yy] in (0,4):
                        tmpt_flame_board[x][yy] = 4
                    else:
                        tmpt_flame_board[x][yy] = 4
                        break

        #引爆炸弹
        for i in range(11):
            for j in range(11):
                if tmpt_bombs_life_time[i][j] == 1:
                    cross_fire(i, j, int(tmpt_bombs[i][j]-1))

        #炸弹时间-1
        for i in range(11):
            for j in range(11):
                if tmpt_bombs_life_time[i][j] != 0:
                    tmpt_bombs_life_time[i][j] -= 1

        #根据flame_board更新其他数组
        for i in range(11):
            for j in range(11):
                if tmpt_flame_board[i][j] != 0:
                    tmpt_flame_board[i][j] -= 1
                if tmpt_flame_board[i][j] != 0:
                    tmpt_board[i][j] = 4
                    tmpt_bombs[i][j] = 0
                    tmpt_bombs_life_time[i][j] = 0

        # 炸弹放置更新
        for x1 in range(11):
            for y1 in range(11):
                if tmpt_bombs[x1, y1] != 0 and tmpt_board[x1, y1] == 0:
                    tmpt_board[x1, y1] = 3

        next_state.flame_board = tmpt_flame_board
        next_state.set_current_board(tmpt_board)
        next_state.set_current_bombs(tmpt_bombs)
        next_state.set_current_move(actions[my_id - 10])
        next_state.set_target_enemy(self.target_enemy)
        return next_state



class Node(object):
    """
    蒙特卡罗树搜索的树结构的Node，包含了父节self.enemies点和直接点等信息，还有用于计算UCB的遍历次数和quality值，还有游戏选择这个Node的State。
    """

    def __init__(self):
        self.parent = None
        self.children = []
        self.visit_times = 1
        self.quality_value = 50
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_children(self):
        return self.children

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_quality_value(self):
        return self.quality_value

    def set_quality_value(self, value):
        self.quality_value = value

    def quality_value_add_n(self, n):
        self.quality_value += n

    def is_all_expand(self):
        if len(self.children) == len(self.get_state().get_available_action()):
            return True
        else:
            return False

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)

    def __repr__(self):
        return "Node: {}, Q/N: {}/{}, state: {}".format(
            hash(self), self.quality_value, self.visit_times, self.state)


def selection(node):
    # print("selection")
    if node.is_all_expand():
        sub_node = best_child(node)
    else:
        # Return the new sub node
        sub_node = expand(node)
    return sub_node


def simulation(node):
    """
    模拟三步写在state_stimulate里面
    """
    # print("stimulation")
    start = time.time()
    current_state = node.get_state()

    index = 0
    NUMBER = random.randint(0, 2)
    while index < NUMBER and current_state.is_terminal() == 3:
        # print("index",index)
        index += 1

        current_state = current_state.get_next_state_with_random_choice()
    final_state_reward = current_state.compute_reward()#越长期的模拟越可能不准确，因此得分要加权?
    # print("simulation: ", 1000 * (time.time() - start))
    return final_state_reward


def expand(node):
    """
    输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。注意，需要保证新增的节点与其他节点Action不同。
    """
    # print("expand")
    tried_sub_node_states_move = [
        sub_node.get_state().get_current_move() for sub_node in node.get_children()
    ]

    new_state = node.get_state().get_next_state_with_random_choice()

    # Check until get the new state which has the different action from others
    while new_state.get_current_move() in tried_sub_node_states_move:
        new_state = node.get_state().get_next_state_with_random_choice()

    sub_node = Node()
    sub_node.set_state(new_state)
    node.add_child(sub_node)
    sub_node.set_parent(node)
    # for sub_node in node.get_children():
    #     print("sub_node", sub_node.get_state().get_current_move(), end = " ")
    # print(" ")
    return sub_node


def best_child(node):
    def node_score(node):  # 目前是UCB选取模式
        my_score = node.get_quality_value()
        my_visit_times = node.get_visit_times()
        my_parent_visit_times = node.get_parent().get_visit_times()
        score = my_score / my_visit_times + 60 * math.sqrt((2 * math.log(my_parent_visit_times) / my_visit_times))
        return score

    """
    得分平均值策略
    """
    # TODO: Use the min float value
    best_score = -sys.maxsize
    best_sub_node = None

    # Travel all sub nodes to find the best one
    for sub_node in node.get_children():
        score = node_score(sub_node)
        if score > best_score:
            best_sub_node = sub_node
            best_score = score
    return best_sub_node


def best_child_with_score(node):
    def node_score_with_score(node):  #
        my_score = node.get_quality_value()
        my_visit_times = node.get_visit_times()
        my_parent_visit_times = node.get_parent().get_visit_times()
        score = my_score / my_visit_times
        return score

    """
    得分平均值策略
    """
    # TODO: Use the min float value
    best_score = -sys.maxsize
    best_sub_node = None

    # Travel all sub nodes to find the best one
    for sub_node in node.get_children():
        score = node_score_with_score(sub_node)
        if score > best_score:
            best_sub_node = sub_node
            best_score = score
    return best_sub_node


def backup(node, reward):
    """
    模拟的得分加到节点上
    """
    # 模拟次数+1
    node.visit_times_add_one()
    # 更新总分
    node.quality_value_add_n(reward)
    node.get_parent().visit_times_add_one()


def MCTS_search(obs, mode, id, enemy=None):
    # Create the initialized state and initialized node
    # print(obs)
    init_state = State(obs, mode, id)
    init_state.set_target_enemy(enemy)
    init_node = Node()
    init_node.set_state(init_state)
    current_node = init_node

    # Set the rounds to play

    computation_budget = 50 # 假设每次模拟1000次
    for i in range(computation_budget):
        # 1. Find the best node to expand
        expand_node = selection(current_node)

        # 2. Random run to add node and get reward
        reward = simulation(expand_node)

        # 3. Update all passing nodes with reward
        backup(expand_node, reward)

    tmpt_score = -sys.maxsize
    tmpt_node = None
    for sub_node in current_node.get_children():
        # print(sub_node.get_state().get_current_move(), " : ", end = "")
        score = (sub_node.get_quality_value() / sub_node.get_visit_times())
        if score > tmpt_score:
            tmpt_score = score
            tmpt_node = sub_node
    # time.sleep(2)

    return tmpt_node.get_state().get_current_move()
