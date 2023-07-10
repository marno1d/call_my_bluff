"""
This is an agent that does dumb stuff
"""
import call_my_bluff as cmb


class MatthewAgentV0:
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

        if observation.bet.index == cmb.game.NO_BET_INDEX:
            action = cmb.game.Action(
                type=cmb.game.ActionType.BET, bet=cmb.game.Bet(index=0)
            )
        elif observation.bet.index == cmb.game.MAX_BET_INDEX:
            action = cmb.game.Action(type=cmb.game.ActionType.CALL)
        else:
            action = cmb.game.Action(
                cmb.game.ActionType.BET,
                bet=cmb.game.Bet(index=observation.bet.index + 1),
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
