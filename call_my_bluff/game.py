"""
The game module contains the game state defintion and the game logic as functions.
"""
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

NO_BET_INDEX = -1
MAX_BET_INDEX = 109
NUM_DICE = 5
STAR = 5


class ActionType(Enum):
    """
    This class is responsible for enumerating the different types of actions.
    """

    CALL = 0
    BET = 1
    REROLL_BET = 2
    RESULT = 3


class Bet:
    """
    This class is responsible for representing a bet.

    Args:
        num_dice (int): The number of dice in the bet.
        dice_value (int): The value of the dice in the bet.

        or

        index (int): The bet index.

    Raises:
        ValueError: If the bet index is invalid.

    """

    def __init__(self, **kwargs):
        # Accept either (num_dice, dice_value) or bet_index
        if "index" in kwargs:
            self._index = kwargs["index"]
        elif "num_dice" in kwargs and "dice_value" in kwargs:
            self._index = self._bet_to_bet_index(
                kwargs["num_dice"], kwargs["dice_value"]
            )
        else:
            raise ValueError("Invalid bet arguments")
        if self._index < NO_BET_INDEX or self._index > MAX_BET_INDEX:
            raise ValueError("Invalid bet index")

    @property
    def index(self):
        """The bet index."""
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def num_dice(self):
        """The number of dice in the bet."""
        return self._bet_index_to_bet(self._index)[0]

    @num_dice.setter
    def num_dice(self, value):
        dice_value = self._bet_index_to_bet(self._index)[1]
        self._index = self._bet_to_bet_index(value, dice_value)

    @property
    def dice_value(self):
        """The value of the dice in the bet."""
        return self._bet_index_to_bet(self._index)[1]

    @dice_value.setter
    def dice_value(self, value):
        num_dice = self._bet_index_to_bet(self._index)[0]
        self._index = self._bet_to_bet_index(num_dice, value)

    def _bet_index_to_bet(self, index: int):
        if index == NO_BET_INDEX:
            return (0, 0)

        group = index // 11
        position = index % 11
        if position == 5:
            bet = (group + 1, STAR)
        elif position < 5:
            bet = (2 * group + 1, position)
        else:
            bet = (2 * group + 2, position - 6)

        return bet

    def _bet_to_bet_index(self, num_dice: int, dice_value: int):
        if (num_dice, dice_value) is (0, 0):
            return NO_BET_INDEX

        if dice_value == STAR:
            group = num_dice - 1
            position = 5
        else:
            group = (num_dice - 1) // 2
            position = dice_value
            if num_dice % 2 == 0:
                position += 6

        index = group * 11 + position
        return index


@dataclass
class Action:
    """This class is responsible for holding the action for the current player."""

    type: ActionType
    dice_to_lock: Optional[List[bool]] = None
    bet: Optional[Bet] = None
    result: Optional[Tuple[int]] = None
    player: Optional[int] = None


@dataclass
class State:
    """
    This class is responsible for keeping track of the state of the game.
    """

    num_players: int
    bet: Bet
    player_curr: int
    player_prev: Optional[int]
    turn_order: List[int]
    num_dice: List[int]
    dice: List[List[int]]
    dice_locked: List[List[bool]]
    action_log: List[Action]


@dataclass
class Observation:
    """
    This class holds the observation for the current player.
    """

    player: int
    bet: Bet
    unknown_dice: List[int]
    known_dice: List[List[int]]
    player_locked_dice: List[bool]
    action_log: List[Action]


def initialize_game(num_players: int) -> State:
    """
    Starts a new game.

    Args:
        num_players (int): The number of players in the game.

    Returns:
        State: The state of the game.
    """
    bet = Bet(index=NO_BET_INDEX)
    num_dice = [NUM_DICE for _ in range(num_players)]
    turn_order = list(range(num_players))
    np.random.shuffle(turn_order)
    dice = [
        np.random.randint(low=0, high=6, size=num_dice[player]).tolist()
        for player in range(num_players)
    ]
    dice_locked = [
        np.zeros((num_dice[player]), dtype=bool).tolist()
        for player in range(num_players)
    ]
    action_log = []
    return State(
        num_players=num_players,
        bet=bet,
        turn_order=turn_order,
        player_curr=turn_order[0],
        player_prev=None,
        num_dice=num_dice,
        dice=dice,
        dice_locked=dice_locked,
        action_log=action_log,
    )


def game_over(state: State) -> bool:
    """
    Determines if the game is over.

    Args:
        state (State): The state of the game.

    Returns:
        bool: True if the game is over, False otherwise.
    """
    players_with_dice = 0
    for player in range(state.num_players):
        if state.num_dice[player] > 0:
            players_with_dice += 1
    return players_with_dice == 1


def round_over(state: State) -> bool:
    """
    Determines if the round is over.

    Args:
        state (State): The state of the game.

    Returns:
        bool: True if the round is over, False otherwise.
    """
    if len(state.action_log) == 0:
        return False

    return state.action_log[-1].type == ActionType.RESULT


def player_observation(state: State):
    """
    Returns the observation for the current player.

    Args:
        state (State): The state of the game.

    Returns:
        dict: The observation for the current player.
    """
    unknown_dice = []
    known_dice = []
    for player in range(state.num_players):
        if player == state.player_curr:
            unknown_dice.append(0)
            known_dice.append(state.dice[player])
        else:
            unknown = 0
            known = []
            for dice_index, dice_value in enumerate(state.dice[player]):
                if state.dice_locked[player][dice_index]:
                    known.append(dice_value)
                else:
                    unknown += 1
            unknown_dice.append(unknown)
            known_dice.append(known)

    return Observation(
        player=state.player_curr,
        bet=state.bet,
        unknown_dice=unknown_dice,
        known_dice=known_dice,
        player_locked_dice=state.dice_locked[state.player_curr],
        action_log=state.action_log,
    )


def _validate_action(state: State, action: Action):
    if action.type == ActionType.CALL:
        if state.bet.index == NO_BET_INDEX:
            raise ValueError("Cannot call without a bet.")
    elif action.type == ActionType.BET or action.type == ActionType.REROLL_BET:
        if action.bet is None:
            raise ValueError("Must specify a bet.")
        if action.bet.index <= state.bet.index:
            raise ValueError("Bet index must be higher than previous bet.")
        if action.type == ActionType.REROLL_BET:
            if action.dice_to_lock is None:
                raise ValueError("Must specify dice to lock.")
            if len(action.dice_to_lock) != state.num_dice[state.player_curr]:
                raise ValueError("Must specify dice to lock for all dice.")
            for dice_index, dice_value in enumerate(action.dice_to_lock):
                if dice_value and state.dice_locked[state.player_curr][dice_index]:
                    raise ValueError("Cannot lock dice that are already locked.")
            if sum(action.dice_to_lock) == 0:
                raise ValueError("Must lock at least one die.")
    else:
        raise ValueError("Invalid action type.")


def _lose_dice(state: State, player: int, num_dice: int):
    state.num_dice[player] -= num_dice
    if state.num_dice[player] < 0:
        state.num_dice[player] = 0
    if state.num_dice[player] == 0:
        state.turn_order.remove(player)
    return state


def _call(state: State):
    # Figure out difference between bet and actual number of dice
    num_dice = state.bet.num_dice
    dice_value = state.bet.dice_value

    actual_num_dice = 0
    for dice in state.dice:
        actual_num_dice += dice.count(dice_value)
        if dice_value != STAR:
            actual_num_dice += dice.count(STAR)
    dice_diff = actual_num_dice - num_dice

    # Determine who loses dice
    if dice_diff > 0:
        state = _lose_dice(state, state.player_curr, dice_diff)
        result = Action(
            type=ActionType.RESULT,
            result=(state.player_curr, dice_diff, actual_num_dice, dice_value),
        )
    elif dice_diff < 0:
        state = _lose_dice(state, state.player_prev, -dice_diff)
        result = Action(
            type=ActionType.RESULT,
            result=(state.player_prev, -dice_diff, actual_num_dice, dice_value),
        )
    else:
        for player in state.turn_order:
            if player != state.player_prev:
                state = _lose_dice(state, player, 1)
        result = Action(
            type=ActionType.RESULT, result=(-1, 1, actual_num_dice, dice_value)
        )

    # Determine next player
    if dice_diff >= 0:
        state.player_curr = state.player_prev
    state.player_prev = None
    state.action_log.append(result)
    return state


def new_round(state: State) -> State:
    """
    Starts a new round.

    Args:
        state (State): The state of the game.

    Returns:
        State: The new state of the game.
    """
    state.bet.index = NO_BET_INDEX
    state.player_prev = None
    state.action_log = []
    for player in range(state.num_players):
        state.dice[player] = np.random.randint(0, 6, state.num_dice[player]).tolist()
        state.dice_locked[player] = np.zeros(
            state.num_dice[player], dtype=bool
        ).tolist()
    return state


def _reroll(state: State, dice_to_lock: List[bool]):
    for dice_index, lock_dice in enumerate(dice_to_lock):
        if not lock_dice and not state.dice_locked[state.player_curr][dice_index]:
            state.dice[state.player_curr][dice_index] = np.random.randint(0, 6)
        else:
            state.dice_locked[state.player_curr][dice_index] = True
    return state


def _bet(state: State, bet: Bet):
    state.bet = bet
    state.player_prev = state.player_curr
    state.player_curr = state.turn_order[
        (state.turn_order.index(state.player_curr) + 1) % len(state.turn_order)
    ]
    return state


def player_action(state: State, action: Action) -> State:
    """
    Returns the new state after the player takes the specified action.

    Args:
        state (State): The state of the game.
        action (Action): The action to take.

    Returns:
        State: The new state of the game.

    Raises:
        ValueError: If the action is invalid.
    """
    _validate_action(state, action)
    action.player = state.player_curr
    state.action_log.append(action)
    if action.type == ActionType.CALL:
        state = _call(state)
    elif action.type == ActionType.BET:
        state = _bet(state, action.bet)
    elif action.type == ActionType.REROLL_BET:
        state = _reroll(state, action.dice_to_lock)
        state = _bet(state, action.bet)
    else:
        raise ValueError("Invalid action type.")
    return state


def render(state: State):
    """
    Renders the current state of the game.

    Args:
        state (State): The state of the game.
    """
    if len(state.action_log) == 1:
        print("####################")
        print("New round started.")
        print(f"Players have {state.num_dice} dice.")
    if state.action_log[-1].type == ActionType.RESULT:
        indices = [-2, -1]
    else:
        indices = [-1]
    for action in [state.action_log[i] for i in indices]:
        if (
            action.type == ActionType.REROLL_BET
            or action.type == ActionType.BET
            or action.type == ActionType.CALL
        ):
            print("--------------------")
            print(
                f"Player {action.player}'s turn, their dice are: {state.dice[action.player]}"
            )
        if action.type == ActionType.REROLL_BET:
            locked_dice = []
            for dice_index, dice_value in enumerate(action.dice_to_lock):
                if dice_value:
                    locked_dice.append(state.dice[action.player][dice_index])
            num_rerolled = state.num_dice[action.player] - state.dice_locked[
                action.player
            ].count(True)
            print(
                f"Player {action.player} locked in {len(locked_dice)} dice: {locked_dice} and rerolled {num_rerolled} dice."
            )
        if action.type == ActionType.BET or action.type == ActionType.REROLL_BET:
            print(
                f"Player {action.player} bet {action.bet.num_dice} {action.bet.dice_value}s"
            )
        if action.type == ActionType.CALL:
            print(f"Player {action.player} called the bet.")
        if action.type == ActionType.RESULT:
            print(f"There are actually {action.result[2]} {action.result[3]}s.")
            if action.result[0] == -1:
                print("Everyone but the better loses a die.")
            else:
                print(f"Player {action.result[0]} loses {action.result[1]} dice.")

            if game_over(state):
                print("!!!!!!!!!!!!!!!!!!!")
                print("Game over!")
                print(f"Player {state.player_curr} wins!")
                print("!!!!!!!!!!!!!!!!!!!")
