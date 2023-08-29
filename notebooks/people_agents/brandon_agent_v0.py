"""
This is an agent that does dumb stuff
"""
import call_my_bluff as cmb
import numpy as np

class BrandonAgentV0:
    """
    This is an agent that does dumb stuff
    """

    def __init__(self):
        pass

    def policy(self, observation: cmb.game.Observation):
        """
        Play a turn.

        Args:
            observation (Observation): The observation of the current state of the game.

        Returns:
            action (Action): The action to take.
        """
        action = cmb.game.Action(type=cmb.game.ActionType.CALL)
        my_dice = np.array(observation.known_dice[observation.player])
        my_locked_dice = np.array(observation.player_locked_dice)
        avg = self.average_number(observation)
        #print(avg)
        if observation.bet.dice_value == 5: # star
            if observation.bet.num_dice > avg[observation.bet.dice_value] + 1:
                action = cmb.game.Action(type=cmb.game.ActionType.CALL)
            else:
                max_avg = max(avg)
                max_avg_dice_num = np.argmax(avg)
                bet = cmb.game.Bet(num_dice=round(max_avg), dice_value=max_avg_dice_num)
                count = 0
                while bet.index <= observation.bet.index:
                    count += 1
                    if count > 2:
                        action = cmb.game.Action(type=cmb.game.ActionType.CALL)
                        return action
                    bet = cmb.game.Bet(num_dice=bet.num_dice + 1, dice_value=bet.dice_value)
                action = cmb.game.Action(type=cmb.game.ActionType.BET, bet=bet)
                # print(bet)
        else:
            if observation.bet.num_dice > avg[observation.bet.dice_value] + 1:
                action = cmb.game.Action(type=cmb.game.ActionType.CALL)
            else:
                max_avg = max(avg)
                max_avg_dice_num = np.argmax(avg)
                bet = cmb.game.Bet(num_dice=round(max_avg), dice_value=max_avg_dice_num)
                count = 0

                dice_to_lock = my_dice == 10  # make all false
                while bet.index <= observation.bet.index:
                    count += 1
                    if count > 1:
                        dice_to_lock = my_dice == bet.dice_value
                    if count > 2:
                        action = cmb.game.Action(type=cmb.game.ActionType.CALL)
                        return action
                    bet = cmb.game.Bet(num_dice=bet.num_dice+1, dice_value=bet.dice_value)

                #print(dice_to_lock)
                dice_to_lock = dice_to_lock & ~my_locked_dice
                #print(dice_to_lock)
                #print(my_locked_dice)
                if np.any(dice_to_lock):
                    action = cmb.game.Action(type=cmb.game.ActionType.REROLL_BET, dice_to_lock=dice_to_lock, bet=bet)
                else:
                    action = cmb.game.Action(type=cmb.game.ActionType.BET, bet=bet)
                #print(bet)


        return action

    def round_results(self, state: cmb.game.State):
        """
        Update the agent's internal state based on the results of the round.

        Args:
            state (State): The state of the game at the end of the round.
        """
        # pylint: disable=unnecessary-pass
        pass

    def num_unknown_dice(self, observation: cmb.game.Observation):
        return sum(observation.unknown_dice)

    def average_number(self, observation: cmb.game.Observation):
        #print(observation.known_dice)
        known_counts = [sum([sublist.count(number) for sublist in observation.known_dice]) for number in
                        [0, 1, 2, 3, 4, 5]]
        #print(known_counts)
        average_numbers = np.array(known_counts) + known_counts[-1] + sum(observation.unknown_dice) / 3
        average_numbers[-1] = known_counts[-1] + sum(observation.unknown_dice) / 6
        return average_numbers
