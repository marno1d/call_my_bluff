"""
This is an agent that does dumb stuff
"""
import math
from typing import List
import numpy as np
import call_my_bluff as cmb


class MatthewAgentV1:
    """
    This is an agent that does smart stuff
    """

    def __init__(self):
        self.num_players = None
        self.log_index = 0
        self.had_first_turn = False
        self.predict_dice = None
        self.bluff_prob_per_player = None
        self.bluff_action_log_indices = None
        self.my_player_index = None

    def _update_player_dice(
        self, observation: cmb.game.Observation, action_log: List[cmb.game.Action]
    ):
        if len(action_log) > 0:
            for i in range(self.log_index, len(action_log)):
                action = action_log[i]
                bet = action.bet
                player = action.player

                if player != observation.player:
                    # Dice info from action player
                    player_num_dice = observation.unknown_dice[player]
                    unknown_dice = 0
                    known_dice = observation.known_dice[player].count(bet.dice_value)

                    # Dice info from self
                    for j, locked in enumerate(observation.player_locked_dice):
                        if locked:
                            known_dice += int(
                                observation.known_dice[observation.player][j]
                                == bet.dice_value
                            )
                        else:
                            unknown_dice += 1
                else:
                    # Dice info from action player
                    player_num_dice = 0
                    unknown_dice = 0
                    known_dice = 0
                    for j, locked in enumerate(observation.player_locked_dice):
                        if locked:
                            known_dice += int(
                                observation.known_dice[observation.player][j]
                                == bet.dice_value
                            )
                        else:
                            player_num_dice += 1

                # Dice info from other players
                for j in range(self.num_players):
                    if j not in (player, observation.player):
                        unknown_dice += observation.unknown_dice[j]
                        known_dice += observation.known_dice[j].count(bet.dice_value)

                # Predict dice
                pred_dice = 0
                for j in range(self.num_players):
                    if j != player:
                        pred_dice += self.predict_dice[j, bet.dice_value]

                if action.type == cmb.game.ActionType.BET:
                    guesses = []
                    num_dice = []
                    for j in range(player_num_dice + 1):
                        prob = self._prob_at_least_min_successes(
                            bet.num_dice - known_dice - pred_dice - j,
                            unknown_dice,
                            self._prob_dice_value(bet),
                        )
                        guesses.append(prob)
                        num_dice.append(j)
                    guesses = np.array(guesses)
                    num_dice = np.array(num_dice)
                    if np.max(guesses) > 0.5:
                        num_dice = num_dice[guesses > 0.5]
                        guesses = guesses[guesses > 0.5]
                        self.predict_dice[player, bet.dice_value] = num_dice[
                            np.argmin(guesses)
                        ]

    def _first_turn(self, observation: cmb.game.Observation):
        self.had_first_turn = True
        self.log_index = 0
        self.num_players = len(observation.unknown_dice)
        self.predict_dice = np.zeros((self.num_players, 6), dtype=np.int32)
        if self.bluff_prob_per_player is None:
            self.bluff_prob_per_player = 0.01 * np.ones(
                (self.num_players), dtype=np.float32
            )
        self.my_player_index = observation.player
        self.bluff_action_log_indices = []

    def policy(self, observation: cmb.game.Observation):
        """
        Play a turn.

        Args:
            observation (Observation): The observation of the current state of the game.

        Returns:
            action (Action): The action to take.
        """
        debug = False

        if not self.had_first_turn:
            self._first_turn(observation)
        self._update_player_dice(observation, observation.action_log)

        if observation.bet.index == cmb.game.MAX_BET_INDEX:
            return cmb.game.Action(cmb.game.ActionType.CALL)

        total_unknown = sum(observation.unknown_dice)
        total_pred = np.zeros((6), dtype=np.int32)
        for i in range(self.num_players):
            if i != observation.player:
                total_pred += self.predict_dice[i, :]
        total_known = np.zeros((6), dtype=np.int32)

        for i in range(6):
            for dice in observation.known_dice:
                total_known[i] += dice.count(i)
        total_known[:5] += total_known[cmb.game.STAR]

        min_successes = (
            observation.bet.num_dice
            - total_known[observation.bet.dice_value]
            - total_pred[observation.bet.dice_value]
        )
        call_prob = 1.0 - self._prob_at_least_min_successes(
            min_successes,
            total_unknown - total_pred[observation.bet.dice_value],
            self._prob_dice_value(observation.bet),
        )

        probs = []
        indices = []
        for index in range(observation.bet.index + 1, cmb.game.MAX_BET_INDEX + 1):
            indices.append(index)
            bet = cmb.game.Bet(index=index)
            min_successes = (
                bet.num_dice - total_known[bet.dice_value] - total_pred[bet.dice_value]
            )
            probs.append(
                self._prob_at_least_min_successes(
                    min_successes,
                    total_unknown - total_pred[bet.dice_value],
                    self._prob_dice_value(bet),
                )
            )
        probs = np.array(probs)
        indices = np.array(indices, dtype=np.int32)

        if debug:
            print("pred_dice:", self.predict_dice)
            print("best prob:", np.max(probs))
            print("call prob:", call_prob)

        bluff_thresh = 0.75
        turn_index = observation.turn_order.index(observation.player)
        next_player = observation.turn_order[
            (turn_index + 1) % len(observation.turn_order)
        ]
        bluff_prob = self.bluff_prob_per_player[next_player]
        if np.max(probs) > call_prob and np.max(probs) > bluff_thresh:
            if np.random.rand() < bluff_prob:
                if debug:
                    print("bluffing!")
                # Find the dice value to bluff
                if np.sum(total_pred) > 0:
                    bluff_dice_value = np.argmax(total_pred[:5])
                else:
                    min_dice_value = 0
                    min_count = 10
                    for i in range(5):
                        count = observation.known_dice[observation.player].count(i)
                        if count < min_count:
                            min_count = count
                            min_dice_value = i
                    bluff_dice_value = min_dice_value
                # Find the num dice to bluff
                num_player_dice = len(observation.known_dice[observation.player])
                if num_player_dice <= 5 and num_player_dice >= 3:
                    num_bluff_dice = np.random.randint(2, num_player_dice)
                else:
                    num_bluff_dice = np.random.randint(1, num_player_dice + 1)
                # Find the index of the bet to bluff
                actual_num_dice = observation.known_dice[observation.player].count(
                    bluff_dice_value
                )
                actual_num_dice += observation.known_dice[observation.player].count(5)
                total_known_bluff_dice = (
                    num_bluff_dice + total_known[bluff_dice_value] - actual_num_dice
                )
                bluff_indices = []
                bluff_probs = []
                for test_num_dice in range(1, 21):
                    bet = cmb.game.Bet(
                        dice_value=bluff_dice_value, num_dice=test_num_dice
                    )
                    if bet.index > observation.bet.index:
                        min_successes = (
                            bet.num_dice
                            - total_known_bluff_dice
                            - total_pred[bluff_dice_value]
                        )
                        bluff_indices.append(bet.index)
                        bluff_probs.append(
                            self._prob_at_least_min_successes(
                                min_successes,
                                total_unknown - total_pred[bluff_dice_value],
                                self._prob_dice_value(bet),
                            )
                        )
                bluff_probs = np.array(bluff_probs)
                bluff_indices = np.array(bluff_indices, dtype=np.int32)
                if np.max(bluff_probs) > bluff_thresh:
                    possible_bluff_indices = bluff_indices[bluff_probs > bluff_thresh]
                    index = possible_bluff_indices[np.argmax(possible_bluff_indices)]
                    action = cmb.game.Action(
                        type=cmb.game.ActionType.BET, bet=cmb.game.Bet(index=index)
                    )
                    self.bluff_action_log_indices.append(len(observation.action_log))
                    return action

        # If bet and call are below threshold, see if rerolling is better
        reroll_thresh = 0.0
        if np.max(probs) < reroll_thresh and call_prob < reroll_thresh:
            if observation.player_locked_dice.count(False) > 1:
                reroll_probs = []
                reroll_indices = []
                reroll_locks = []
                for index in range(
                    observation.bet.index + 1, cmb.game.MAX_BET_INDEX + 1
                ):
                    reroll_indices.append(index)
                    bet = cmb.game.Bet(index=index)
                    num_reroll = 0
                    reroll_lock = []
                    lock_last_index = 0
                    for i, locked in enumerate(observation.player_locked_dice):
                        if not locked:
                            if (
                                observation.known_dice[observation.player][i]
                                == bet.dice_value
                                or observation.known_dice[observation.player][i]
                                == cmb.game.STAR
                            ):
                                reroll_lock.append(True)
                            else:
                                reroll_lock.append(False)
                                num_reroll += 1
                                lock_last_index = i
                        else:
                            reroll_lock.append(False)
                    if reroll_lock.count(True) == 0:
                        reroll_lock[lock_last_index] = True
                        num_reroll -= 1
                    min_successes = (
                        bet.num_dice
                        - total_known[bet.dice_value]
                        - total_pred[bet.dice_value]
                    )
                    reroll_probs.append(
                        self._prob_at_least_min_successes(
                            min_successes,
                            total_unknown - total_pred[bet.dice_value] + num_reroll,
                            self._prob_dice_value(bet),
                        )
                    )
                    reroll_locks.append(reroll_lock)
                reroll_probs = np.array(reroll_probs)
                reroll_indices = np.array(reroll_indices, dtype=np.int32)
                if debug:
                    print("reroll probs:", np.max(reroll_probs))
                if (
                    np.max(reroll_probs) > np.max(probs)
                    and np.max(reroll_probs) > call_prob
                ):
                    index = reroll_indices[np.argmax(reroll_probs)]
                    action = cmb.game.Action(
                        type=cmb.game.ActionType.REROLL_BET,
                        bet=cmb.game.Bet(index=index),
                        dice_to_lock=reroll_locks[np.argmax(reroll_probs)],
                    )
                    return action

        prob_threshold = 0.75
        if np.max(probs) > prob_threshold:
            possible_indices = indices[probs > prob_threshold]
            index = np.max(possible_indices)
            action = cmb.game.Action(
                cmb.game.ActionType.BET, bet=cmb.game.Bet(index=index)
            )
        else:
            index = indices[np.argmax(probs)]
            if np.max(probs) > call_prob:
                action = cmb.game.Action(
                    cmb.game.ActionType.BET, bet=cmb.game.Bet(index=index)
                )
            else:
                action = cmb.game.Action(cmb.game.ActionType.CALL)

        return action

    def _prob_dice_value(self, bet: cmb.game.Bet):
        if bet.dice_value == cmb.game.STAR:
            return 1.0 / 6
        return 1.0 / 3

    def round_results(self, state: cmb.game.State):
        """
        Update the agent's internal state based on the results of the round.

        Args:
            state (State): The state of the game at the end of the round.
        """
        # pylint: disable=unnecessary-pass
        if self.had_first_turn:
            self.had_first_turn = False

            # Update the bluff probability per player
            for i, action in enumerate(state.action_log):
                if action.player == self.my_player_index:
                    if i in self.bluff_action_log_indices:
                        next_action = state.action_log[i + 1]
                        j = i + 2
                        found_result = False
                        while not found_result:
                            if state.action_log[j].type == cmb.game.ActionType.RESULT:
                                found_result = True
                            else:
                                j += 1
                        result_action = state.action_log[j]
                        lose_tie = (
                            result_action.result[0] == -1
                            and state.action_log[j - 2].player != self.my_player_index
                        )
                        if result_action.result[0] == self.my_player_index or lose_tie:
                            self.bluff_prob_per_player[next_action.player] *= 0.75
                        else:
                            self.bluff_prob_per_player[next_action.player] *= 1.25
            self.bluff_prob_per_player = np.clip(self.bluff_prob_per_player, 0.0, 1.0)

    def _prob_at_least_min_successes(self, min_successes, num_trials, success_prob):
        """
        Computes the probability of getting at least a certain number of successful dice
        rolls with a given number of tries.

        Parameters:
        min_successes (int): The minimum number of successful dice rolls.
        num_trials (int): The total number of dice rolls.
        success_prob (float): The probability of success on a single dice roll.

        Returns:
        float: The probability of getting at least the minimum number of successful dice rolls
        in the given number of trials.
        """
        if min_successes <= 0:
            return 1.0

        # Initialize the cumulative probability
        cum_prob = 0

        # Loop over the range of possible successes
        for k in range(min_successes, num_trials + 1):
            # Calculate the combination
            comb = math.comb(num_trials, k)
            # Calculate the binomial probability
            prob = comb * (success_prob**k) * ((1 - success_prob) ** (num_trials - k))
            # Add the binomial probability to the cumulative probability
            cum_prob += prob

        return cum_prob
