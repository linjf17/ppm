from collections import defaultdict
import queue
import random

import numpy as np

from . import BaseAgent
from .. import constants
from .. import utility
from . import mcts
from time import time

FLAME_BOARD = np.zeros((11, 11))
REM_BOARD = np.full((11, 11), 5, dtype="int64")
REM_BOM = np.full((11, 11), 5, dtype="int64")
REM_BOM_LIFE = np.full((11, 11), 5, dtype="int64")

class MyAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(MyAgent, self).__init__(*args, **kwargs)

        # Keep track of recently visited uninteresting positions so that we
        # don't keep visiting the same places.
        self._recently_visited_positions = []
        self._recently_visited_length = 6
        # Keep track of the previous direction to help with the enemy standoffs.
        self._prev_direction = None
        self.my_id_is_set = False
        self.my_id = None

    def act(self, obs, action_space):
        print(obs)
        def convert_bombs(bomb_map):
            '''Flatten outs the bomb array'''
            ret = []
            locations = np.where(bomb_map > 0)
            for r, c in zip(locations[0], locations[1]):
                ret.append({
                    'position': (r, c),
                    'blast_strength': int(bomb_map[(r, c)])
                })
            return ret

        def convert_bombs_life_time(bomb_map):
            '''Flatten outs the bomb array'''
            ret = []
            locations = np.where(bomb_map > 0)
            for r, c in zip(locations[0], locations[1]):
                ret.append({
                    'position': (r, c),
                    'bombs_life_time': int(bomb_map[(r, c)])
                })
            return ret

        my_position = tuple(obs['position'])  # obs??应该是observation
        if self.my_id_is_set == False:
        	self.my_id_is_set = True
        	if my_position == (1, 1):
        		self.my_id = 10
        	elif my_position == (9, 1):
        		self.my_id = 11
        	elif my_position == (9, 9):
        		self.my_id = 12
        	else:
        		self.my_id = 13
        board = np.array(obs['board'])
        bombArray = np.array(obs['bomb_blast_strength'])
        bombTimeArray = np.array(obs['bomb_life'])
        bombs = convert_bombs(np.array(obs['bomb_blast_strength']))  # bombs是爆炸半径
        bombs_life_time = convert_bombs_life_time(np.array(obs['bomb_life']))
        global FLAME_BOARD
        global REM_BOARD
        global REM_BOM
        global REM_BOM_LIFE
        obs['flame_board'] = FLAME_BOARD.copy()
        for i in range(11):
            for j in range(11):
                if board[i][j] != 5:
                    REM_BOARD[i][j] = board[i][j]
                    REM_BOM[i][j] = np.array(obs['bomb_blast_strength'])[i][j]
                    REM_BOM_LIFE[i][j] = np.array(obs['bomb_life'])[i][j]
        obs['rem_board'] = REM_BOARD.copy()
        obs['rem_bom'] = REM_BOM.copy()
        obs['rem_bom_life'] = REM_BOM_LIFE.copy()

        visited = set()
        def cross_fire(x, y, radius):
            visited.add((x,y))
            FLAME_BOARD[x][y] = 4
            for i in range(1, radius+1):
                xx = x - i
                if 0 <= xx < 11:
                    if (board[xx][y] in (1,5)):
                        break
                    elif board[xx][y] == 3:
                        if  (xx,y) not in visited:
                            cross_fire(xx, y, int(bombArray[xx][y]-1))
                        break
                    elif board[xx][y] in (0,4):
                        FLAME_BOARD[xx][y] = 4
                    else:
                        FLAME_BOARD[xx][y] = 4
                        break

            for i in range(1, radius+1):
                xx = x + i
                if 0 <= xx < 11:
                    if (board[xx][y] in (1,5)):
                        break
                    elif board[xx][y] == 3:
                        if  (xx,y) not in visited:
                            cross_fire(xx, y, int(bombArray[xx][y]-1))
                        break
                    elif board[xx][y] in (0,4):
                        FLAME_BOARD[xx][y] = 4
                    else:
                        FLAME_BOARD[xx][y] = 4
                        break

            for i in range(1, radius+1):
                yy = y - i
                if 0 <= yy < 11:
                    if (board[x][yy] in (1, 5)):
                        break
                    elif  board[x][yy] == 3:
                        if (x, yy) not in visited:
                            cross_fire(x, yy, int(bombArray[x][yy]-1))
                        break
                    elif board[x][yy] in (0,4):
                        FLAME_BOARD[x][yy] = 4
                    else:
                        FLAME_BOARD[x][yy] = 4
                        break

            for i in range(1, radius+1):
                yy = y + i
                if 0 <= yy < 11:
                    if (board[x][yy] in (1,5)):
                        break
                    elif  board[x][yy] == 3:
                        if (x, yy) not in visited:
                            cross_fire(x, yy, int(bombArray[x][yy]-1))
                        break
                    elif board[x][yy] in (0,4):
                        FLAME_BOARD[x][yy] = 4
                    else:
                        FLAME_BOARD[x][yy] = 4
                        break

        for i in range(11):
            for j in range(11):
                if board[i][j] == 3 and bombTimeArray[i][j] == 1:
                    cross_fire(i, j, int(bombArray[i][j]-1))
        for i in range(11):
            for j in range(11):
                if FLAME_BOARD[i][j] != 0:
                    FLAME_BOARD[i][j] -= 1
        enemies = [constants.Item(e) for e in obs['enemies']]
        ammo = int(obs['ammo'])
        # print("ammo",ammo)
        blast_strength = int(obs['blast_strength'])
        items, dist, prev = self._djikstra(
            board, my_position, bombs, enemies, depth=10)  # djikstra最短路径

        

        def evade_condition(radius, position):
            def in_range_of_bomb(direction, radius, position):
                x, y = position
                # 原地有炸弹
                if bombArray[x][y] != 0:
                    return True
                # 上面有炸弹
                evadeTime = 4
                if direction == constants.Action.Up:
                    for i in range(1, radius + 1):
                        if x - i >= 0:
                            if board[x - i][y] == 3 and bombArray[x - i][y] > i and bombTimeArray[x-i][y] < evadeTime:
                                return True
                            elif board[x - i][y] != 0:
                                return False
                        else:
                            return False
                # 下面有炸弹
                elif direction == constants.Action.Down:
                    for i in range(1, radius + 1):
                        if x + i < 11:
                            if board[x + i][y] == 3 and bombArray[x + i][y] > i and bombTimeArray[x+i][y] < evadeTime:
                                return True
                            elif board[x + i][y] != 0:
                                return False
                        else:
                            return False
                # 左边有炸弹
                elif direction == constants.Action.Left:
                    for i in range(1, radius + 1):
                        if y - i >= 0:
                            if board[x][y - i] == 3 and bombArray[x][y - i] > i and bombTimeArray[x][y-i] < evadeTime:
                                return True
                            elif board[x][y - i] != 0:
                                return False
                        else:
                            return False
                # 右边有炸弹
                elif direction == constants.Action.Right:
                    for i in range(1, radius + 1):
                        if y + i < 11:
                            if board[x][y + i] == 3 and bombArray[x][y + i] > i and bombTimeArray[x][y+i] < evadeTime:
                                return True
                            elif board[x][y + i] != 0:
                                return False
                        else:
                            return False
                # 没有炸弹
                else:
                    return False
            for direction in [
                constants.Action.Right,
                constants.Action.Left,
                constants.Action.Up,
                constants.Action.Down,
            ]:
                if in_range_of_bomb(direction, radius, position):
                    return True
            return False

        def direction_to_safe_location():
            
                    
            directions = [constants.Action.Stop]
            for direction in [
                constants.Action.Right,
                constants.Action.Left,
                constants.Action.Up,
                constants.Action.Down,
            ]:
                position = utility.get_next_position(my_position, direction)
                if utility.position_on_board(board, position) \
                        and utility.position_is_passable(board, position, enemies):
                    if not evade_condition(5, position):
                        return direction.value
                    else:
                        directions.append(direction)
            return directions[-1].value


        def enemy_within_range(radius):

            for enemy in enemies:
                for position in items.get(enemy, []):
                    d = dist[position]
                    if d <= radius:
                        return True
            return False

        def attack_condition():
            if self._maybe_bomb(
                    ammo, blast_strength, items, dist, my_position, bombs, bombs_life_time, board) \
                    and enemy_within_range(5):
                return True
            return False


        def direction_to_enemy():
            dire = self._near_enemy(my_position, items, dist, prev, enemies, 5)
            if dire is None:
                return 0
            return dire

        if evade_condition(5, my_position):
            # 不会走到火上的方向
            # direction = direction_to_safe_location()
            # if in_range_of_bomb(direction, 5, my_position):
            direction = mcts.MCTS_search(obs, 0, self.my_id)
            return direction

        # Move towards a good item if there is one within two reachable spaces.
        direction = self._near_good_powerup(my_position, items, dist, prev, 2)
        if direction is not None:
            directions = self._filter_unsafe_directions(board, my_position,[direction], bombs)
            if not directions:
                return 0
            #print("explore 0 ", directions[0])
            return directions[0].value

        if attack_condition():
            #print("attack")
            if enemy_within_range(3):
            #     adjacent_enemy = self._is_adjacent_enemy(items, dist, enemies)
            #     direction = mcts.MCTS_search(obs, 1, adjacent_enemy)
            #     #print(direction)
            #     return direction
            # #print(direction_to_enemy())
            # return direction_to_enemy().value
            # 
                adjacent_enemy = self._is_adjacent_enemy(items, dist, enemies)
                direction = mcts.MCTS_search(obs, 1, self.my_id,adjacent_enemy)
                return direction
            else:
                dir_to_enemy = direction_to_enemy()
                safe_dir = self._filter_unsafe_directions(board, my_position,
                                                        [dir_to_enemy], bombs)
                if not safe_dir:
                    pass
                else:
                    return safe_dir[0].value
        


        

        # Maybe lay a bomb if we are within a space of a wooden wall.
        if self._near_wood(my_position, items, dist, prev, 1):
            #print("explore 1")
            if self._maybe_bomb(ammo, blast_strength, items, dist, my_position, bombs, bombs_life_time, board):
                return constants.Action.Bomb.value
            else:
                return constants.Action.Stop.value

        # Move towards a wooden wall if there is one within two reachable spaces and you have a bomb.
        direction = self._near_wood(my_position, items, dist, prev, 2)
        if direction is not None:
            directions = self._filter_unsafe_directions(board, my_position,
                                                        [direction], bombs)
            if directions:
                #print("explore 2")
                return directions[0].value

        # Choose a random but valid direction.
        directions = [
            constants.Action.Stop, constants.Action.Left,
            constants.Action.Right, constants.Action.Up, constants.Action.Down
        ]
        valid_directions = self._filter_invalid_directions(
            board, my_position, directions, enemies)
        directions = self._filter_unsafe_directions(board, my_position,
                                                    valid_directions, bombs)
        directions = self._filter_recently_visited(
            directions, my_position, self._recently_visited_positions)
        directions = self._filter_easy_stuck(directions, my_position, board)
        if len(directions) > 1:
            directions = [k for k in directions if k != constants.Action.Stop]
        if not len(directions):
            directions = [constants.Action.Stop]

        # Add this position to the recently visited uninteresting positions so we don't return immediately.
        self._recently_visited_positions.append(my_position)
        self._recently_visited_positions = self._recently_visited_positions[
                                           -self._recently_visited_length:]
        move = random.choice(directions).value
        return move

    @staticmethod
    def _djikstra(board, my_position, bombs, enemies, depth=None, exclude=None):
        assert (depth is not None)

        if exclude is None:
            exclude = [
                constants.Item.Fog, constants.Item.Rigid, constants.Item.Flames
            ]

        def out_of_range(p_1, p_2):
            '''Determines if two points are out of rang of each other'''
            x_1, y_1 = p_1
            x_2, y_2 = p_2
            return abs(y_2 - y_1) + abs(x_2 - x_1) > depth

        items = defaultdict(list)
        dist = {}
        prev = {}
        Q = queue.Queue()

        my_x, my_y = my_position
        for r in range(max(0, my_x - depth), min(len(board), my_x + depth)):
            for c in range(max(0, my_y - depth), min(len(board), my_y + depth)):
                position = (r, c)
                if any([
                    out_of_range(my_position, position),
                    utility.position_in_items(board, position, exclude),
                ]):
                    continue

                prev[position] = None
                item = constants.Item(board[position])
                items[item].append(position)

                if position == my_position:
                    Q.put(position)
                    dist[position] = 0
                else:
                    dist[position] = np.inf

        for bomb in bombs:
            if bomb['position'] == my_position:
                items[constants.Item.Bomb].append(my_position)

        while not Q.empty():
            position = Q.get()

            if utility.position_is_passable(board, position, enemies):
                x, y = position
                val = dist[(x, y)] + 1
                for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row + x, col + y)
                    if new_position not in dist:
                        continue

                    if val < dist[new_position]:
                        dist[new_position] = val
                        prev[new_position] = position
                        Q.put(new_position)
                    elif (val == dist[new_position] and random.random() < .5):
                        dist[new_position] = val
                        prev[new_position] = position

        return items, dist, prev

    def _directions_in_range_of_bomb(self, board, my_position, bombs, dist, bombs_life_time):
        ret = defaultdict(int)
        count = 0
        for bombslife in bombs_life_time:
            count += 1  # 计算炸弹数
        x, y = my_position
        for bomb in bombs:
            position = bomb['position']
            distance = dist.get(position)
            if distance is None:
                continue

            bomb_range = bomb['blast_strength']
            this_bomb_life_time = 1000  # 说明这是一个不需要担心的炸弹
            for life_time in bombs_life_time:
                if life_time['position'] == position:
                    this_bomb_life_time = life_time['bombs_life_time']
                    break
            if distance > bomb_range or this_bomb_life_time > 10:  # 时间在一定限度内的炸弹才需要考虑
                continue
            return True
        return False

    def _find_safe_directions(self, board, my_position, unsafe_directions,
                              bombs, enemies):

        def is_stuck_direction(next_position, bomb_range, next_board, enemies):
            '''Helper function to do determine if the agents next move is possible.'''
            Q = queue.PriorityQueue()
            Q.put((0, next_position))
            seen = set()

            next_x, next_y = next_position
            is_stuck = True
            while not Q.empty():
                dist, position = Q.get()
                seen.add(position)

                position_x, position_y = position
                if next_x != position_x and next_y != position_y:
                    is_stuck = False
                    break

                if dist > bomb_range:
                    is_stuck = False
                    break

                for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row + position_x, col + position_y)
                    if new_position in seen:
                        continue

                    if not utility.position_on_board(next_board, new_position):
                        continue

                    if not utility.position_is_passable(next_board,
                                                        new_position, enemies):
                        continue

                    dist = abs(row + position_x - next_x) + abs(col + position_y - next_y)
                    Q.put((dist, new_position))
            return is_stuck

        # All directions are unsafe. Return a position that won't leave us locked.
        safe = []

        if len(unsafe_directions) == 4:
            next_board = board.copy()
            next_board[my_position] = constants.Item.Bomb.value

            for direction, bomb_range in unsafe_directions.items():
                next_position = utility.get_next_position(
                    my_position, direction)
                next_x, next_y = next_position
                if not utility.position_on_board(next_board, next_position) or \
                        not utility.position_is_passable(next_board, next_position, enemies):
                    continue

                if not is_stuck_direction(next_position, bomb_range, next_board,
                                          enemies):
                    # We found a direction that works. The .items provided
                    # a small bit of randomness. So let's go with this one.
                    return [direction]
            if not safe:
                safe = [constants.Action.Stop]
            return safe

        x, y = my_position
        disallowed = []  # The directions that will go off the board.

        for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            position = (x + row, y + col)
            direction = utility.get_direction(my_position, position)

            # Don't include any direction that will go off of the board.
            if not utility.position_on_board(board, position):
                disallowed.append(direction)
                continue

            # Don't include any direction that we know is unsafe.
            if direction in unsafe_directions:
                continue

            if utility.position_is_passable(board, position,
                                            enemies) or utility.position_is_fog(
                board, position):
                safe.append(direction)

        if not safe:
            # We don't have any safe directions, so return something that is allowed.
            safe = [k for k in unsafe_directions if k not in disallowed]

        if not safe:
            # We don't have ANY directions. So return the stop choice.
            return [constants.Action.Stop]

        return safe

    @staticmethod
    def _is_adjacent_enemy(items, dist, enemies):
        for enemy in enemies:
            for position in items.get(enemy, []):
                if dist[position] <= 4:
                    return (position, dist[position], enemy.value)
        return False

    @staticmethod
    def _has_bomb(obs):
        return obs['ammo'] >= 1

    # @staticmethod

    @staticmethod
    def _maybe_bomb(ammo, blast_strength, items, dist, my_position, bombs, bombs_life_time, board):
        def passage_will_be_bomb(board, my_position, bombs, dist, bombs_life_time):
            ret = defaultdict(int)
            count = 0
            for bombslife in bombs_life_time:
                count += 1  # 计算炸弹数
            x, y = my_position
            for bomb in bombs:
                position = bomb['position']
                distance = dist.get(position)
                if distance is None:
                    continue

                bomb_range = bomb['blast_strength']
                this_bomb_life_time = 1000  # 说明这是一个不需要担心的炸弹
                for life_time in bombs_life_time:
                    if life_time['position'] == position:
                        this_bomb_life_time = life_time['bombs_life_time']
                        break
                if distance > bomb_range or this_bomb_life_time > 9:  # 在我到达这个位置之前不会爆炸
                    continue
                return True
            return False
        """Returns whether we can safely bomb right now.

        Decides this based on:
        1. Do we have ammo?
        2. If we laid a bomb right now, will we be stuck?
        """
        # Do we have ammo?
        if ammo < 1:
            return False

        # Will we be stuck?
        x, y = my_position
        for position in items.get(constants.Item.Passage):
            if dist[position] == np.inf:
                continue

            # We can reach a passage that's outside of the bomb strength.#并且这个通路不会被一个5步爆炸的炸弹炸死
            if dist[position] > blast_strength and not passage_will_be_bomb(board, position, bombs, dist,
                                                                            bombs_life_time):
                return True

            # We can reach a passage that's outside of the bomb scope.
            position_x, position_y = position
            if position_x != x and position_y != y and not passage_will_be_bomb(board, position, bombs, dist,
                                                                                bombs_life_time):
                return True

        return False

    @staticmethod
    def _nearest_position(dist, objs, items, radius):
        nearest = None
        dist_to = max(dist.values())

        for obj in objs:
            for position in items.get(obj, []):
                d = dist[position]
                if d <= radius and d <= dist_to:
                    nearest = position
                    dist_to = d

        return nearest

    @staticmethod
    def _get_direction_towards_position(my_position, position, prev):
        if not position:
            return None

        next_position = position
        while prev[next_position] != my_position:
            next_position = prev[next_position]

        return utility.get_direction(my_position, next_position)

    @classmethod
    def _near_enemy(cls, my_position, items, dist, prev, enemies, radius):
        nearest_enemy_position = cls._nearest_position(dist, enemies, items,
                                                       radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_enemy_position, prev)

    @classmethod
    def _near_good_powerup(cls, my_position, items, dist, prev, radius):
        objs = [
            constants.Item.ExtraBomb, constants.Item.IncrRange,
            constants.Item.Kick
        ]
        nearest_item_position = cls._nearest_position(dist, objs, items, radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_item_position, prev)

    @classmethod
    def _near_wood(cls, my_position, items, dist, prev, radius):
        objs = [constants.Item.Wood]
        nearest_item_position = cls._nearest_position(dist, objs, items, radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_item_position, prev)

    @staticmethod
    def _filter_invalid_directions(board, my_position, directions, enemies):
        ret = []
        for direction in directions:
            position = utility.get_next_position(my_position, direction)
            if utility.position_on_board(
                    board, position) and utility.position_is_passable(
                board, position, enemies):
                ret.append(direction)
        return ret

    @staticmethod
    def _filter_unsafe_directions(board, my_position, directions, bombs):
        ret = []
        for direction in directions:
            x, y = utility.get_next_position(my_position, direction)
            is_bad = False
            for bomb in bombs:
                bomb_x, bomb_y = bomb['position']
                blast_strength = bomb['blast_strength']
                if (x == bomb_x and abs(bomb_y - y) <= blast_strength) or \
                        (y == bomb_y and abs(bomb_x - x) <= blast_strength):
                    is_bad = True
                    break
            if not is_bad:
                ret.append(direction)
        return ret

    @staticmethod
    def _filter_recently_visited(directions, my_position,
                                 recently_visited_positions):
        ret = []
        for direction in directions:
            if not utility.get_next_position(
                    my_position, direction) in recently_visited_positions:
                ret.append(direction)

        if not ret:
            ret = directions
        return ret

    @staticmethod
    def _filter_easy_stuck(directions, my_position, board):
        ret = []
        r,c = my_position
        if (r == 0 or r == 10) and (c == 0 or c == 10):
            return directions
        for direction in directions:
            x, y = utility.get_next_position(my_position, direction)
            if not (x == 10 or x == 0 or y == 0 or y == 10):
                ret.append(direction)
        return ret


