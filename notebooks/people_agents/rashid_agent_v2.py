"""
This is an agent that does dumb stuff
"""
import math
import call_my_bluff as cmb


class RashidAgentV2:
    """
    This is an agent that does dumb stuff
    """

    def __init__(self, external_var1=None, external_var2=None, external_var3=None):
        self.external_var1 = external_var1
        self.external_var2 = external_var2
        self.external_var3 = external_var3

        pass

    def policy(self, observation: cmb.game.Observation):
        """
        Play a turn.
            observation (Observation): The observation of the current state of the game.

        Returns:
            action (Action): The action to take.
        """
        prob_call = self.external_var1
        prob_threshold_no_bet = self.external_var2
        prob_threshold_bet = self.external_var3
        known_dice = [dice for sublist in observation.known_dice for dice in sublist]
        n_dice = sum(observation.unknown_dice)
        prob_number = 1 / 3
        prob_star = 1 / 6

        current_bet = [observation.bet.dice_value, observation.bet.num_dice]
        dice_count = current_bet[1]

        expected_bet_indices = [0]
        expected_bet_prob = [0]
        if observation.bet.index == cmb.game.NO_BET_INDEX:
            prob_threshold = prob_threshold_no_bet
        else:
            prob_threshold = prob_threshold_bet

        if observation.bet.index == cmb.game.MAX_BET_INDEX:
            action = cmb.game.Action(type=cmb.game.ActionType.CALL)
        else:
            # Initialize the probability matrix with zeros
            prob_matrix = [[0 for _ in range(n_dice)] for _ in range(6)]

            for dice_number in range(6):
                if dice_number == 5:
                    match_dice = [dice for dice in known_dice if dice == dice_number]
                else:
                    match_dice = [
                        dice for dice in known_dice if dice == dice_number or dice == 5
                    ]
                if match_dice:
                    num_match_dice = len(match_dice)
                else:
                    num_match_dice = 0

                # if dice_count > len(num_match_dice) and num_match_dice:
                #     dice_count_int_offset = dice_count - len(num_match_dice)
                # else:
                #     dice_count_int_offset = dice_count

                # Calculate the probability for each dice count
                for dice_count_int in range(1, n_dice + 1):
                    prob = []
                    if dice_count_int <= num_match_dice:
                        prob_matrix[dice_number][dice_count_int - 1] = 1

                    else:
                        # Adjust dice_count_int with the offset for actual computation
                        # adjusted_dice_count = dice_count_int + dice_count_int_offset

                        if dice_number == 5:
                            for idice in range(
                                dice_count_int - num_match_dice, n_dice + 1
                            ):
                                prob.append(
                                    self.binomial_probability(n_dice, idice, prob_star)
                                )
                        else:
                            for idice in range(
                                dice_count_int - num_match_dice, n_dice + 1
                            ):
                                prob.append(
                                    self.binomial_probability(
                                        n_dice, idice, prob_number
                                    )
                                )

                        prob_matrix[dice_number][dice_count_int - 1] = sum(prob)
                    if prob_matrix[dice_number][dice_count_int - 1] > prob_threshold:
                        expected_bet = cmb.game.Bet(
                            num_dice=int(dice_count_int), dice_value=dice_number
                        )
                        expected_bet_indices.append(expected_bet.index)
                        expected_bet_prob.append(
                            prob_matrix[dice_number][dice_count_int - 1]
                        )

            max_index, max_index_for_prob = max(
                (value, idx) for idx, value in enumerate(expected_bet_indices)
            )
            bet = cmb.game.Bet(index=max_index)
            # print(
            #    f"num_dice: {bet.num_dice}, dice_value: {bet.dice_value}, prob: {expected_bet_prob[max_index_for_prob]}"
            # )

            if (
                max_index > observation.bet.index
                and max_index <= cmb.game.MAX_BET_INDEX
            ):
                action = cmb.game.Action(
                    type=cmb.game.ActionType.BET, bet=cmb.game.Bet(index=max_index)
                )
            else:
                prob_max_num_dice = len(prob_matrix[0]) - 1
                if observation.bet.num_dice - 1 > prob_max_num_dice:
                    dices_c = [
                        known_dice.count(0) + known_dice.count(5),
                        known_dice.count(1) + known_dice.count(5),
                        known_dice.count(2) + known_dice.count(5),
                        known_dice.count(3) + known_dice.count(5),
                        known_dice.count(4) + known_dice.count(5),
                        known_dice.count(5),
                    ]
                    if dices_c[current_bet[0]] >= current_bet[1]:
                        action = cmb.game.Action(
                            type=cmb.game.ActionType.BET,
                            bet=cmb.game.Bet(
                                num_dice=observation.bet.num_dice + 1,
                                dice_value=observation.bet.dice_value,
                            ),
                        )
                    else:
                        indices = [
                            idx
                            for idx, num in enumerate(
                                dices_c[observation.bet.dice_value :],
                                start=observation.bet.dice_value,
                            )
                            if num >= observation.bet.num_dice
                        ]
                        if indices:
                            if indices[-1] > observation.bet.dice_value:
                                action = cmb.game.Action(
                                    type=cmb.game.ActionType.BET,
                                    bet=cmb.game.Bet(
                                        num_dice=observation.bet.num_dice,
                                        dice_value=indices[-1],
                                    ),
                                )
                            else:
                                action = cmb.game.Action(type=cmb.game.ActionType.CALL)
                        else:
                            action = cmb.game.Action(type=cmb.game.ActionType.CALL)

                else:
                    if (
                        prob_matrix[observation.bet.dice_value][
                            observation.bet.num_dice - 1
                        ]
                        > prob_call
                    ):
                        action = cmb.game.Action(
                            type=cmb.game.ActionType.BET,
                            bet=cmb.game.Bet(
                                num_dice=observation.bet.num_dice + 1,
                                dice_value=observation.bet.dice_value,
                            ),
                        )
                    else:
                        action = cmb.game.Action(type=cmb.game.ActionType.CALL)

            return action

    def round_results(self, state: cmb.game.State):
        """
        Update the agent's internal state based on the results of the round.

        Args:
            state (State): The state of the game at the end of the round.
        """
        # pylint: disable=unnecessary-pass
        pass

    def binomial_coefficient(self, total_num, combs):
        """Compute the binomial coefficient."""
        return math.factorial(total_num) / (
            math.factorial(combs) * math.factorial(total_num - combs)
        )

    def binomial_probability(self, total_num, combs, prob):
        """Compute the expected probability"""
        return (
            self.binomial_coefficient(total_num, combs)
            * (prob**combs)
            * ((1 - prob) ** (total_num - combs))
        )
