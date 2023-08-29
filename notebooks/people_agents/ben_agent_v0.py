import copy
import numpy as np
from call_my_bluff import game

class BenAgentV0:
    """
    This is Ben's version 0 agent.
    """

    def __init__(self):
        self.probabilities = np.load('probabilities.npz')['probabilities']

    def policy(self, observation: game.Observation):
        """
        Play a turn.

        Args:
            observation (Observation): The observation of the current state of the game.

        Returns:
            action (Action): The action to take.
        """

        # create copy of observation with a python deepcopy
        observation_copy = copy.deepcopy(observation)

        if observation.bet.index == game.NO_BET_INDEX:
            action = game.Action(game.ActionType.BET, bet=game.Bet(num_dice=1, dice_value=0))
        elif observation.bet.index == game.MAX_BET_INDEX:
            action = game.Action(game.ActionType.CALL)
        else:
            action = self.action_to_minimize_lost_dice(observation_copy)

        return action

    def round_results(self, state: game.State):
        """
        Update the agent's internal state based on the results of the round.

        Args:
            state (State): The state of the game at the end of the round.
        """
        pass

    def action_to_minimize_lost_dice(self, observation: game.Observation):
        """
        Get the action that minimizes the number of dice lost.

        Args:
            observation (Observation): The observation of the current state of the game.

        Returns:
            action (Action): The action to take.
        """
        original_index = observation.bet.index

        # get the number of dice and face of the current bet
        num_dice = observation.bet.num_dice
        dice_face = observation.bet.dice_value

        # what do I know about the known dice?
        num_correct = self.count_known_dice_for_bet(observation)

        # how many unknown dice are there?
        num_unknown = self.get_total_unknown_dice(observation)

        # get the average dice lost if I call the bet
        dice_lost_by_calling = calculate_mean_dice_lost_by_caller(
            unknown_dice=num_unknown, bid_quantity=num_dice-num_correct,
            bid_face=dice_face, probabilities=self.probabilities)
        
        # get the average dice lost if I raise the bet for up to 10 raises
        dice_lost_by_raising = self.get_dice_lost_by_raising(observation, 10)

        # determine the action that minimizes the number of dice lost
        if dice_lost_by_calling < np.min(dice_lost_by_raising):
            action = game.Action(game.ActionType.CALL)
        else:
            # get the number of raises that minimizes the number of dice lost
            num_raises = np.argmin(dice_lost_by_raising) + 1
            action = game.Action(
                game.ActionType.BET, 
                bet=game.Bet(index=original_index+num_raises))

        return action

    def get_dice_lost_by_raising(self, observation: game.Observation, num_raises: int):
        """
        Get the average number of dice lost by raising the bet.

        Args:
            observation (Observation): The observation of the current state of the game.
            num_raises (int): The number of raises to consider.

        Returns:
            dice_lost_by_raising (np.array): The average number of dice lost for each raise.
        """
        dice_lost = np.zeros(num_raises)

        for i in range(num_raises):
            observation.bet.index += 1
            num_correct = self.count_known_dice_for_bet(observation)
            num_unknown = self.get_total_unknown_dice(observation)
            num_dice = observation.bet.num_dice
            dice_face = observation.bet.dice_value

            dice_lost[i] = calculate_mean_dice_lost_by_bidder(
                unknown_dice=num_unknown, bid_quantity=num_dice-num_correct,
                bid_face=dice_face, probabilities=self.probabilities)
            
        return dice_lost

    def count_known_dice_for_bet(self, observation: game.Observation):
        """
        Count the number of known dice that match the current bet.

        Args:
            observation (Observation): The observation of the current state of the game.

        Returns:
            num_correct (int): The number of known dice that match the current bet.
        """

        # get the face of the current bet
        dice_face = observation.bet.dice_value

        # get the number of known dice that match the current bet
        num_correct = 0
        for player_dice in observation.known_dice:
            for dice in player_dice:
                if dice == dice_face or dice == 5:
                    num_correct += 1

        return num_correct
    
    def get_total_unknown_dice(self, observation: game.Observation):
        """
        Get the total number of unknown dice.

        Args:
            observation (Observation): The observation of the current state of the game.

        Returns:
            num_unknown (int): The total number of unknown dice.
        """

        # get the total number of unknown dice
        num_unknown = 0
        for player_unknown in observation.unknown_dice:
            num_unknown += player_unknown

        return num_unknown

# --------------------------------------------------------------
# Functions for calculating the number of dice lost by each player

def calculate_mean_dice_lost_by_caller(unknown_dice=1, bid_quantity=1, bid_face=1, probabilities=None):
    '''
    Calculate the mean number of dice lost for a given bid
    '''

    face_index = bid_face

    # probability of each bid quantity
    bid_quantity_probabilities = probabilities[unknown_dice-1, :unknown_dice+1, face_index]

    # probability of each relevant bid quantity being called incorrectly
    bid_quantity_called_incorrect_probabilities = bid_quantity_probabilities[bid_quantity:]

    # number of dice lost by caller for each relevant bid quantity
    dice_lost_by_caller = np.arange(1, bid_quantity_called_incorrect_probabilities.shape[0])
    dice_lost_by_caller = np.concatenate((np.ones(1), dice_lost_by_caller))

    # mean number of dice lost by caller
    mean_dice_lost_by_caller = np.sum(bid_quantity_called_incorrect_probabilities * dice_lost_by_caller)

    return mean_dice_lost_by_caller

def calculate_mean_dice_lost_by_bidder(unknown_dice=1, bid_quantity=1, bid_face=1, probabilities=None):
    '''
    Calculate the mean number of dice lost for a given bid
    '''

    face_index = bid_face

    # probability of each bid quantity
    bid_quantity_probabilities = probabilities[unknown_dice-1, :unknown_dice+1, face_index]

    # probability of each relevant bid quantity being called incorrectly
    bid_quantity_called_incorrect_probabilities = bid_quantity_probabilities[:bid_quantity]

    # number of dice lost by bidder for each relevant bid quantity
    dice_lost_by_bidder = np.arange(1, bid_quantity_called_incorrect_probabilities.shape[0] + 1)
    dice_lost_by_bidder = np.flip(dice_lost_by_bidder)

    # mean number of dice lost by bidder
    mean_dice_lost_by_bidder = np.sum(bid_quantity_called_incorrect_probabilities * dice_lost_by_bidder)

    return mean_dice_lost_by_bidder


# --------------------------------------------------------------
# Functions for calculating statistical outcomes

def calculate_probabilities(max_num_dice=1, num_simulations=10):
    '''
    Calculate the probabilities of a specific bid value
    showing up given the number of dice
    '''

    # total counts for dice faces 0 to 5
    total_counts = np.zeros((max_num_dice, max_num_dice + 1, 6))

    for _ in range(num_simulations):
        x = np.random.randint(6, size=(max_num_dice))
        for i in range(max_num_dice):
            bins = np.bincount(x[:i+1], minlength=6)
            for face in range(5):
                total_counts[i, bins[face] + bins[5], face] += 1
            total_counts[i, bins[5], 5] += 1

    probabilities = total_counts / num_simulations

    return probabilities

if __name__ == '__main__':
    # calculate probabilities and save to npz file
    probabilities = calculate_probabilities(max_num_dice=30, num_simulations=100000)
    np.savez('probabilities.npz', probabilities=probabilities)