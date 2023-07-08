import unittest

import call_my_bluff as cmb


# no class or method docstrings in tests
# pylint: disable=missing-class-docstring, missing-function-docstring
class TestGame(unittest.TestCase):
    def test_initialize_game_returns_state(self):
        # Ensure that initialize_game returns a State object
        state = cmb.initialize_game(2)
        self.assertIsInstance(state, cmb.game.State)

    def test_initialize_game_num_players(self):
        # Ensure that the num_players attribute of the State object is correct
        state = cmb.initialize_game(3)
        self.assertEqual(state.num_players, 3)

    def test_initialize_game_turn_order(self):
        # Ensure that the turn_order attribute of the State object is a list of integers
        state = cmb.initialize_game(4)
        self.assertIsInstance(state.turn_order, list)
        self.assertTrue(all(isinstance(player, int) for player in state.turn_order))

    def test_initialize_game_dice(self):
        # Ensure that the dice attribute of the State object is a list of lists of integers
        state = cmb.initialize_game(2)
        self.assertIsInstance(state.dice, list)
        self.assertTrue(
            all(isinstance(player_dice, list) for player_dice in state.dice)
        )
        self.assertTrue(
            all(
                isinstance(die, int)
                for player_dice in state.dice
                for die in player_dice
            )
        )

    def test_initialize_game_dice_locked(self):
        # Ensure that the dice_locked attribute of the State object is a list of lists of booleans
        state = cmb.initialize_game(3)
        self.assertIsInstance(state.dice_locked, list)
        self.assertTrue(
            all(
                isinstance(player_dice_locked, list)
                for player_dice_locked in state.dice_locked
            )
        )
        self.assertTrue(
            all(
                isinstance(die_locked, bool)
                for player_dice_locked in state.dice_locked
                for die_locked in player_dice_locked
            )
        )

    def test_initialize_game_bet_index(self):
        # Ensure that the bet_index attribute of the State object is None
        state = cmb.initialize_game(4)
        self.assertTrue(state.bet_index == -1)

    def test_game_over(self):
        # Ensure that game_over returns True if there is only one player with dice
        state = cmb.initialize_game(2)
        state.num_dice = [0, 1]
        self.assertTrue(cmb.game.game_over(state))

    def test_player_observation(self):
        # Ensure that player_observation returns an Observation object
        state = cmb.initialize_game(2)
        observation = cmb.player_observation(state)
        self.assertIsInstance(observation, cmb.Observation)

    def test_player_observation_known_and_unknown_dice(self):
        # Ensure that player_observation correctly identifies known and unknown dice
        state = cmb.initialize_game(2)
        state.player_curr = 0
        state.dice_locked[0] = [True, False, False, True, False]  # player 0
        state.dice_locked[1] = [False, False, True, False, True]  # player 1
        observation = cmb.player_observation(state)
        self.assertEqual(observation.unknown_dice[0], 0)
        self.assertEqual(observation.unknown_dice[1], 3)
        self.assertEqual(len(observation.known_dice[0]), 5)
        self.assertEqual(len(observation.known_dice[1]), 2)

    def test_player_observation_current_player(self):
        # Ensure that player_observation correctly identifies the current player
        state = cmb.initialize_game(3)
        state.player_curr = 1
        observation = cmb.player_observation(state)
        self.assertEqual(observation.player, 1)
