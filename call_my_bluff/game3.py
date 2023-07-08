"""
This module contains the Game class, which is responsible for keeping track of
the state of the game.
"""
import random
from typing import Dict, Tuple
import numpy as np


class Game:
    """
    This class is responsible for keeping track of the state of the game.


    Args:
        num_players (int): The number of players in the game.

    """

    def __init__(self, num_players: int):
        self.num_players = num_players

        self.players = list(range(self.num_players))
        self.player_turn = 0
        self.player_turn_prev = -1
        self.wins = {player: 0 for player in self.players}

        self.num_dice_per_player = None
        self.dice_per_player = None
        self.locked_dice = None
        self.log = None
        self.log_index = 0
        self.bet_index = None
        self.new_game()

    def new_game(self):
        """
        Starts a new game.
        """
        self.num_dice_per_player = {player: 5 for player in self.players}
        random.shuffle(self.players)
        self.log = []
        self._new_round()

    def _new_round(self):
        self.dice_per_player = {
            player: np.random.randint(
                low=0, high=6, size=(self.num_dice_per_player[player])
            )
            for player in self.players
        }
        self.locked_dice = {
            player: np.zeros((self.num_dice_per_player[player]), dtype=bool)
            for player in self.players
        }
        self.bet_index = None
        self.log.append(["new", self.num_dice_per_player])

    def _game_over(self):
        players_with_dice = 0
        winning_player = None
        for player in self.players:
            if self.num_dice_per_player[player] > 0:
                players_with_dice += 1
                winning_player = player

        if players_with_dice <= 1:
            self.wins[winning_player] += 1
            return True
        return False

    def next_player(self):
        """
        Returns the next player in the game.
        """
        while not self._game_over():
            # Skip players with no dice
            if self.num_dice_per_player[self.players[self.player_turn]] == 0:
                self.player_turn = (self.player_turn + 1) % self.num_players

            yield self.players[self.player_turn]
            self.player_turn = (self.player_turn + 1) % self.num_players

    def get_state(self, player: int):
        """
        Returns a dictionary containing the state of the game for the given
        player.

        Args:
            player (int): The player to return the state for.

        Returns:
            dict: A dictionary containing the state of the game for the given
            player.
        """
        unknown_dice_per_player = {}
        known_dice_per_player = {}
        for other_player in self.players:
            if other_player != player:
                unknown_dice_per_player[other_player] = self.num_dice_per_player[
                    other_player
                ] - np.sum(self.locked_dice[other_player])
                known_dice_per_player[other_player] = self.dice_per_player[
                    other_player
                ][self.locked_dice[other_player]]

        state = {
            "player": player,
            "dice_roll": self.dice_per_player[player],
            "locked_dice": self.locked_dice[player],
            "unknown_dice_per_competitor": unknown_dice_per_player,
            "known_dice_per_competitor": known_dice_per_player,
            "log": self.log,
            "bet_index": self.bet_index,
        }
        return state

    def _bet_difference(self):
        (num_dice_bet, dice_value) = bet_index_to_bet(self.bet_index)
        num_dice_actual = 0
        for player in self.players:
            num_dice_actual += np.sum(self.dice_per_player[player] == dice_value)
            if dice_value < 5:
                num_dice_actual += np.sum(self.dice_per_player[player] == 5)
        return int(num_dice_actual - num_dice_bet)

    def _call_action(self):
        dice_difference = self._bet_difference()
        if dice_difference > 0:
            self.num_dice_per_player[self.players[self.player_turn]] = max(
                [
                    self.num_dice_per_player[self.players[self.player_turn]]
                    - dice_difference,
                    0,
                ]
            )
        elif dice_difference == 0:
            for player in self.players:
                if player != self.players[self.player_turn_prev]:
                    self.num_dice_per_player[player] = max(
                        [
                            self.num_dice_per_player[player] - 1,
                            0,
                        ]
                    )
        else:
            self.num_dice_per_player[self.players[self.player_turn_prev]] = max(
                [
                    self.num_dice_per_player[self.players[self.player_turn_prev]]
                    + dice_difference,
                    0,
                ]
            )
        self.log.append(
            [
                "call",
                self.players[self.player_turn],
                self.bet_index,
                dice_difference,
                self.dice_per_player,
            ]
        )
        self._new_round()

    def _invalid_action(self, message: str = ""):
        self.num_dice_per_player[self.players[self.player_turn]] = 0
        self.log.append(["invalid", self.players[self.player_turn], message])
        self._new_round()

    def turn(self, action: Dict):
        # Call
        if action["call"]:
            if self.bet_index is None:
                self._invalid_action("call before first bet")
            else:
                self._call_action()
        else:
            # Reroll
            if action["reroll"]:
                dice_to_lock = action["dice_to_lock"]
                if len(dice_to_lock) == 0:
                    self._invalid_action("didn't lock any dice")
                    return
                for i in dice_to_lock:
                    if self.locked_dice[self.players[self.player_turn]][i]:
                        self._invalid_action("locked a dice that was already locked")
                        return
                    self.locked_dice[self.players[self.player_turn]][i] = True
                # Reroll all unlocked dice
                for i in range(
                    self.num_dice_per_player[self.players[self.player_turn]]
                ):
                    if not self.locked_dice[self.players[self.player_turn]][i]:
                        self.dice_per_player[self.players[self.player_turn]][
                            i
                        ] = np.random.randint(low=0, high=6)
                self.log.append(
                    [
                        "reroll",
                        self.players[self.player_turn],
                        len(dice_to_lock),
                    ]
                )
            # Bet
            new_bet_index = action["bet"]
            if new_bet_index > 109 or new_bet_index < 0:
                self._invalid_action("invalid bet index")
                return
            else:
                self.bet_index = new_bet_index
                self.player_turn_prev = self.player_turn
                self.log.append(["bet", self.players[self.player_turn], self.bet_index])
        self._render()

    def _render(self):
        for i in range(self.log_index, len(self.log)):
            if self.log[i][0] == "call":
                print(
                    f"player {self.log[i][1]} called {bet_index_to_bet(self.log[i][2])} with a difference of {self.log[i][3]} dice"
                )
            elif self.log[i][0] == "invalid":
                print(
                    f"player {self.log[i][1]} made an invalid action: {self.log[i][2]}"
                )
            elif self.log[i][0] == "reroll":
                print(f"player {self.log[i][1]} rerolled {self.log[i][2]} dice")
            elif self.log[i][0] == "bet":
                print(f"player {self.log[i][1]} bet {bet_index_to_bet(self.log[i][2])}")
            elif self.log[i][0] == "new":
                print("####################")
                print(f"new round")
                print(f"remaining dice per player {self.num_dice_per_player}")
        self.log_index = len(self.log)


def bet_index_to_bet(bet_index: int):
    """
    Converts a bet index to a bet. The bet index is an integer between 0 and
    109, inclusive.

    Args:
        bet_index (int): The bet index to convert. Must be between 0 and 109.

    Returns:
        tuple: The bet corresponding to the bet index. (num_dice, dice_value)
    """
    if bet_index is None:
        return (None, None)
    if bet_index < 0 or bet_index >= 110:
        raise ValueError("Invalid bet index")

    group = bet_index // 11
    position = bet_index % 11
    if position == 5:
        bet = (group + 1, 5)
    elif position < 5:
        bet = (2 * group + 1, position)
    else:
        bet = (2 * group + 2, position - 6)

    return bet


def bet_to_bet_index(bet: Tuple):
    """
    Converts a bet to a bet index.

    Args:
        bet (tuple): The bet to convert. (num_dice, dice_value)

    Returns:
        int: The bet index corresponding to the bet.
    """
    if bet is None:
        return None
    if len(bet) != 2:
        raise ValueError("Invalid bet")
    if bet[1] < 0 or bet[1] > 5:
        raise ValueError("Invalid bet")
    if bet[1] == 5 and (bet[0] < 1 or bet[0] > 10):
        raise ValueError("Invalid bet")
    if bet[1] < 5 and (bet[0] < 1 or bet[0] > 20):
        raise ValueError("Invalid bet")

    if bet[1] == 5:
        group = bet[0] - 1
        position = 5
    else:
        group = (bet[0] - 1) // 2
        position = bet[1]
        if bet[0] % 2 == 0:
            position += 6

    bet_index = group * 11 + position
    return bet_index
