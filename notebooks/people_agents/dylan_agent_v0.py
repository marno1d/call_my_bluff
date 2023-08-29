"""
This is Dylan's agent that (still) does dumb stuff
"""
import call_my_bluff as cmb
import itertools
from collections import Counter
import math
import sys


def P(r, n, p):
    nCr = math.comb(n, r)
    return nCr * math.pow(p, r) * math.pow((1 - p), n - r)


def getChance(unknownDice, guess, face):
    if face == 5:
        p = 1 / 6
    else:
        p = 1 / 3

    n = unknownDice
    r = guess
    nCr = math.comb(n, r)

    sum = 0
    for ri in range(r, unknownDice + 1):
        prob = P(ri, n, p)
        sum = sum + prob

    return sum


def buildChances(numUnknown, face):
    chanceTuples = []
    for i in range(1, numUnknown + 1):
        chanceTuples.append((i, getChance(numUnknown, i, face)))
    return chanceTuples


def bet(number, face):
    # print(f"Dylan is betting {number} {face}")
    action = cmb.game.Action(
        type=cmb.game.ActionType.BET, bet=cmb.game.Bet(num_dice=number, dice_value=face)
    )
    return action


def call():
    action = cmb.game.Action(type=cmb.game.ActionType.CALL)
    return action


# A function to get the most common face from the known dice.
# Should retrun the most common face and the number of times it appears.
# If the most common face is not 5 (wild) it DOES take wilds into account
def getMyMostCommonFace(observation):
    knownDice = list(itertools.chain.from_iterable(observation.known_dice))
    diceCounter = Counter(knownDice)
    mostCommon = diceCounter.most_common(2)
    mostNum = mostCommon[0][1]
    mostFace = mostCommon[0][0]
    if len(mostCommon) > 1:
        secNum = mostCommon[1][1]
        secFace = mostCommon[1][0]
    else:
        secNum = 0
        secFace = mostFace

    if mostFace != 5:
        numWilds = diceCounter[5]
        mostNum += numWilds
    else:
        mostNum = secNum + mostNum
        mostFace = secFace

    return mostFace, mostNum


# Takes the current bet and figures out how many of that number should be expected base on
# my know info.
def getBetExpected(observation):
    knownDice = list(itertools.chain.from_iterable(observation.known_dice))
    dicecounter = Counter(knownDice)
    betFace = observation.bet.dice_value
    numUnknownDice = sum(observation.unknown_dice)
    if betFace == 5:
        expectedAverageOfUnknown = int(numUnknownDice / 6)
        numKnownOfBetFace = dicecounter[betFace]
        numTotalExpected = expectedAverageOfUnknown + numKnownOfBetFace
    else:
        expectedAverageOfUnknown = int(numUnknownDice / 3)
        numKnownOfBetFace = dicecounter[betFace]
        numKnownWilds = dicecounter[5]
        numTotalExpected = expectedAverageOfUnknown + numKnownOfBetFace + numKnownWilds
    return numTotalExpected


class DylanAgentV0:
    """
    This is Dylan's agent that (still) does dumb stuff
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

        # Knumber of known dice
        numKnownDice = sum(
            [len(observation.known_dice[i]) for i in range(len(observation.known_dice))]
        )
        numUnknownDice = sum(observation.unknown_dice)
        mostFace, mostNum = getMyMostCommonFace(observation)

        # Never use 5s, always use additiona linfo in hand to bet on a different face
        # chanceTabsle = buildChances(numUnknownDice, mostFace)
        expectedAverageOfUnknown = int(numUnknownDice / 3)
        numToBet = mostNum + expectedAverageOfUnknown
        try:
            myBestBet = cmb.game.Bet(num_dice=numToBet, dice_value=mostFace)
        except:
            myBestBet = cmb.game.Bet(index=cmb.game.MAX_BET_INDEX)

        # If no bet has been made, it is the start of the round
        # and a bet must be made.
        # Use the following formula to make a starting bet:
        # -Find the average number of a given face in the unknown set of dice by
        #  dividing the remaing number of dice by 2. This is the 'expected average of unknown'
        # -Round this number down to the nearest integer.
        # -Look at your own dice, identify the face of which you have the most,
        #  including wilds. This is the 'known most', and the face you will bet on.
        # -Add the 'exepected average of unknown' to the 'known most' - 1 (to be 'safe').
        # -Bet that aomunt of dice of the 'known most' face.
        # -Doesnt take into own wilds yet
        if observation.bet.index == cmb.game.NO_BET_INDEX:
            if numToBet <= 0:
                numToBet = 1
            action = bet(numToBet, mostFace)

        # Need to decide to bet or call
        else:
            currentBetFace = observation.bet.dice_value
            currentBetNum = observation.bet.num_dice

            # Detirmine the expected number of the current bet based on our known info
            numExpectedBet = getBetExpected(observation)
            # Allow for the bet to go 1 over the expected before calling
            if observation.bet.num_dice > numExpectedBet:
                # print(f"Dylan expected a max of {numExpectedBet} (+1) {observation.bet.dice_value}s")
                action = call()
            elif observation.bet.num_dice < numExpectedBet:
                action = bet(numExpectedBet, observation.bet.dice_value)
            # Bet -
            else:
                if myBestBet.index > observation.bet.index:
                    action = bet(myBestBet.num_dice, myBestBet.dice_value)
                else:
                    action = bet(
                        observation.bet.num_dice + 1, observation.bet.dice_value
                    )

        return action

    def round_results(self, state: cmb.game.State):
        """
        Update the agent's internal state based on the results of the round.

        Args:
            state (State): The state of the game at the end of the round.
        """
        # pylint: disable=unnecessary-pass
        pass
