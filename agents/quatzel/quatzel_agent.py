import numpy as np
import random
from agents import BaselineAgent
from agents.utility import GroundTruthType
from agents.utility.exceptions import *

import socket
from math import cos, sin, degrees, pi

from src.client.agent_client import AgentClient, GameState, RequestCodes
from src.trajectory_planner.trajectory_planner import SimpleTrajectoryPlanner
from src.computer_vision.GroundTruthReader import GroundTruthReader, NotVaildStateError
from src.computer_vision.game_object import GameObjectType
from src.computer_vision.GroundTruthTest import GroundTruthTest
from src.utils.point2D import Point2D
from agents.utility.degrees import *
from src.computer_vision.game_object import GameObject

from keras.layers import Conv2D, Conv3D, Dropout, MaxPool2D, Activation, Dense, Softmax, Conv1D
import tensorflow as tf
import logging


class QuatzelAgent(BaselineAgent):
    """Birds in boots (server/client version)"""

    def __init__(self, agent_ind, agent_configs, min_deg: int = -4, max_deg: int = 78, deg_step: float = 0.3):

        super().__init__(
            agent_ind=agent_ind,
            agent_configs=agent_configs)

        self.n_actions = None
        self.n_states = None
        self.alpha = 0.1  # Learning rate
        self.n_episodes = 100
        self.epsilon = 0.1
        self.gamma = 0.9  # discount factor

        # degrees
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.deg_step = deg_step

        self.Q = None

    def solve(self):
        """
        * Solve a particular level by using q learning
        * @return GameState: the game state after shots.
        """
        # Initialization
        ground_truth_type = GroundTruthType.ground_truth_screenshot
        vision = self._update_reader(ground_truth_type.value, self.if_check_gt)
        # states
        self.n_states = len(vision.find_birds().keys())
        # actions
        self.n_actions = get_n_degrees_possibilities(self.min_deg, self.max_deg, self.deg_step)
        # Set up the Q-table with all zeros
        self.Q = np.zeros([self.n_states, self.n_actions])
        current_state = 0  # start from the first bird
        # For each episode
        for episode in range(self.n_episodes):
            # Retake vision for episode
            vision = self._update_reader(ground_truth_type.value, self.if_check_gt)
            sling = vision.find_slingshot_mbr()[0]
            # Choose an action according to an exploration policy (e.g. Epsilon-Greedy)
            action = self.choose_action(current_state)
            # Take the action and observe the reward and next state
            try:
                reward, next_state = self.take_action(current_state, action, sling)
                # Update the Q-value of the current state and action using the Bellman equation
                self.Q[current_state, action] = self.alpha * self.Q[current_state, action] + (1 - self.alpha) * \
                                                (reward + self.gamma * np.max(self.Q[next_state, :]))

            except (GameSessionLossException, GameSessionWonException) as e:
                # Update the Q-value of the current state and action using the Bellman equation
                self.Q[current_state, action] = self.alpha * self.Q[current_state, action] + (1 - self.alpha) * e.reward
                #self.ar.load_next_available_level()
                return

    def choose_action(self, state: int):
        if np.random.uniform() < self.epsilon:
            # Explore: choose a random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: choose the action with the highest Q-value
            return np.argmax(self.Q[state, :])

    def take_action(self, state: int, action: int, sling):
        score = self.check_current_level_score()
        # Shoot the bird
        deg = get_deg_from_index(self.min_deg, self.deg_step, action)
        self.shoot_bird_by_angle(deg, sling)
        # Check score diff
        score = self.check_current_level_score() - score
        # Get next_state
        game_state = self.ar.get_game_state()
        # If the level is solved , go to the next level
        if game_state == GameState.WON:
            raise GameSessionWonException(reward=score)
        if game_state == GameState.LOST:
            raise GameSessionLossException(reward=-1000)  # TODO: constant reward for loss, need to think this differently
        next_state = state + 1
        return score, next_state

    def run(self):
        self.ar.configure(self.id)
        # do not use observer
        # self.observer_ar.configure(self.id)
        self.ar.set_game_simulation_speed(self.sim_speed)
        # n_levels = self.update_no_of_levels()

        # self.solved = [0 for x in range(n_levels)]

        # load next available level
        # self.current_level = self.ar.load_next_available_level()
        # self.novelty_existence = self.ar.get_novelty_info()

        '''
        Uncomment this section to run TEST for requesting groudtruth via different thread
        '''
        # gt_thread = Thread(target=self.sample_state)
        # gt_thread.start()
        '''
        END TEST
        '''

        change_from_training = False

        while True:
            state = self.ar.get_game_state()
            # If the level is solved , go to the next level
            if state == GameState.WON:
                self.repeated_gt_counter = 0
                # check for change of number of levels in the game
                n_levels = self.update_no_of_levels()

                # /System.out.println(" loading the level " + (self.current_level + 1) )
                # self.check_current_level_score()
                self.current_level = self.ar.load_next_available_level()
                self.novelty_existence = self.ar.get_novelty_info()

                # make a new trajectory planner whenever a new level is entered
                self.tp = SimpleTrajectoryPlanner()

            elif state == GameState.LOST:
                self.repeated_gt_counter = 0
                # check for change of number of levels in the game
                # n_levels = self.update_no_of_levels()

                self.check_current_level_score()

                # If lost, then restart the level
                self.failed_counter += 1
                if self.failed_counter > 60:  # for testing , go directly to the next level

                    self.failed_counter = 0
                    self.current_level = self.ar.load_next_available_level()
                    self.novelty_existence = self.ar.get_novelty_info()

                else:
                    self.logger.info("fail level count does not reach the limit, restart the level")
                    self.ar.restart_level()

            elif state == GameState.LEVEL_SELECTION:
                self.logger.info(
                    "unexpected level selection page, go to the last current level : " + self.current_level)
                self.current_level = self.ar.load_next_available_level()
                self.novelty_existence = self.ar.get_novelty_info()

            elif state == GameState.MAIN_MENU:
                self.repeated_gt_counter = 0
                self.logger.info("unexpected main menu page, reload the level : %s" % self.current_level)
                self.current_level = self.ar.load_next_available_level()
                self.novelty_existence = self.ar.get_novelty_info()

            elif state == GameState.EPISODE_MENU:
                self.logger.info("unexpected episode menu page, reload the level:  %s" % self.current_level)
                self.current_level = self.ar.load_next_available_level()
                self.novelty_existence = self.ar.get_novelty_info()

            elif state == GameState.PLAYING:
                self.solve()


            elif state == GameState.REQUESTNOVELTYLIKELIHOOD:
                # Require report novelty likelihood and then playing can be resumed
                # dummy likelihoods:
                self.logger.info("received request of novelty likelihood")
                novelty_likelihood = 0.9
                non_novelty_likelihood = 0.1
                novel_obj_ids = {1, -2, -398879789}
                novelty_level = 0
                novelty_description = "";
                self.ar.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood, novel_obj_ids,
                                                  novelty_level, novelty_description)

            elif state == GameState.NEWTRIAL:
                self.repeated_gt_counter = 0
                # Make a fresh agents to continue with a new trial (evaluation)
                self.logger.critical("new trial state received")
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
                 allowNoveltyInfo) = self.ar.ready_for_new_set()
                self.current_level = 0
                self.training_level_backup = 0

            elif state == GameState.NEWTESTSET:
                self.repeated_gt_counter = 0
                self.logger.critical("new test set state received")
                # DO something to clone a test-only agents that does not learn
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
                 allowNoveltyInfo) = self.ar.ready_for_new_set()

                if change_from_training:
                    self.training_level_backup = self.current_level
                self.current_level = 0
                change_from_training = False

            elif state == GameState.NEWTRAININGSET:
                self.repeated_gt_counter = 0
                # DO something to start a fresh agents for a new training set
                self.logger.critical("new training set state received")
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
                 allowNoveltyInfo) = self.ar.ready_for_new_set()
                self.current_level = 0
                self.training_level_backup = 0
                change_from_training = True

            elif state == GameState.RESUMETRAINING:
                self.repeated_gt_counter = 0
                # DO something to resume the training agents to the previous training
                self.logger.critical("resume training set state received")
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
                 allowNoveltyInfo) = self.ar.ready_for_new_set()
                change_from_training = True
                self.current_level = self.training_level_backup

            elif state == GameState.EVALUATION_TERMINATED:
                # store info and disconnect the agents as the evaluation is finished
                self.logger.critical("Evaluation terminated.")
                exit(0)