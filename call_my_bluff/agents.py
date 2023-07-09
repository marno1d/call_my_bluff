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
            total_known_dice += sum(dice)
        total_dice = total_unknown_dice + total_known_dice

        dice_value = np.random.randint(0, 5)
        num_dice_8 = int(0.8 * total_dice / 3)
        num_dice_8 = max([num_dice_8, 1])
        bet_8 = game.Bet(num_dice=num_dice_8, dice_value=dice_value)

        if observation.bet.index == game.NO_BET_INDEX:
            action = game.Action(type=game.ActionType.BET, bet=bet_8)
        # TODO: MAX_BET_INDEX is an off by one hour and it might occur in other places
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
