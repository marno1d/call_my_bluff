""" This file contains some example agents. """
import numpy as np

from call_my_bluff import game


class SimpleAgent:
    """
    This is a simple agent that plays near the total dice average.
    """

    def __init__(self):
        pass

    def policy(self, observation: game.Observation):
        """
        Play a turn.

        Args:
            observation (Observation): The observation of the current state of the game.

        Returns:
            action (Action): The action to take.
        """

        total_unknown_dice = sum(observation.unknown_dice)
        total_known_dice = 0
        for dice in observation.known_dice:
            total_known_dice += len(dice)
        total_dice = total_unknown_dice + total_known_dice

        dice_value = np.random.randint(0, 5)
        num_dice_8 = int(0.8 * total_dice / 3)
        num_dice_8 = max([num_dice_8, 1])
        bet_8 = game.Bet(num_dice=num_dice_8, dice_value=dice_value)

        if observation.bet.index == game.NO_BET_INDEX:
            action = game.Action(type=game.ActionType.BET, bet=bet_8)
        elif observation.bet.index == game.MAX_BET_INDEX:
            action = game.Action(type=game.ActionType.CALL)
        elif bet_8.index > observation.bet.index:
            action = game.Action(game.ActionType.BET, bet=bet_8)
        else:
            bet_plus_1 = game.Bet(index=observation.bet.index + 1)
            num_dice = bet_plus_1.num_dice
            dice_value = bet_plus_1.dice_value
            expected_dice = total_dice / 3
            if dice_value == game.STAR:
                expected_dice = total_dice / 6
            if expected_dice >= num_dice:
                action = game.Action(game.ActionType.BET, bet=bet_plus_1)
            else:
                action = game.Action(game.ActionType.CALL)

        return action

    def round_results(self, state: game.State):
        """
        Update the agent's internal state based on the results of the round.

        Args:
            state (State): The state of the game at the end of the round.
        """
        # pylint: disable=unnecessary-pass
        pass


class MaxAgent:
    """
    This is an agent that always bets the maximum.
    """

    def __init__(self):
        pass

    def policy(self, observation: game.Observation):
        """
        Play a turn.

        Args:
            observation (Observation): The observation of the current state of the game.

        Returns:
            action (Action): The action to take.
        """
        expected_bet_indices = [0]
        for dice_value in range(0, 6):
            expected_dice = sum(observation.unknown_dice)
            if dice_value == game.STAR:
                expected_dice /= 6
            else:
                expected_dice /= 3
            for known_dice in observation.known_dice:
                expected_dice += known_dice.count(dice_value)
                if dice_value != game.STAR:
                    expected_dice += known_dice.count(game.STAR)
            if int(expected_dice) > 0:
                expected_bet = game.Bet(
                    num_dice=int(expected_dice), dice_value=dice_value
                )
                expected_bet_indices.append(expected_bet.index)
        max_index = max(expected_bet_indices)
        if max_index > observation.bet.index and max_index <= game.MAX_BET_INDEX:
            action = game.Action(
                type=game.ActionType.BET, bet=game.Bet(index=max_index)
            )
        else:
            action = game.Action(type=game.ActionType.CALL)

        return action

    def round_results(self, state: game.State):
        """
        Update the agent's internal state based on the results of the round.

        Args:
            state (State): The state of the game at the end of the round.
        """
        # pylint: disable=unnecessary-pass
        pass
