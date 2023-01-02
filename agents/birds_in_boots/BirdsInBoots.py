import time
import os
import numpy as np
import random
import socket
from math import cos, sin, degrees, pi
from src.client.agent_client import AgentClient, GameState, RequestCodes
from src.trajectory_planner.trajectory_planner import SimpleTrajectoryPlanner
from src.computer_vision.GroundTruthReader import GroundTruthReader, NotVaildStateError
from src.computer_vision.game_object import GameObjectType
from src.computer_vision.GroundTruthTest import GroundTruthTest
from src.utils.point2D import Point2D

from src.computer_vision.game_object import GameObject
from agents import BaselineAgent
from agents.utility import GroundTruthType
from keras.layers import Conv2D, Conv3D, Dropout, MaxPool2D, Activation, Dense, Softmax, Conv1D
import tensorflow as tf
import logging


class BirdsInBoots(BaselineAgent):
    """Birds in boots (server/client version)"""

    def __init__(self, agent_ind, agent_configs, min_deg: int = -4, max_deg: int = 78, deg_step: float = .3):

        super().__init__(
            agent_ind=agent_ind,
            agent_configs=agent_configs)

        # Degrees
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.deg_step = deg_step
        self.deg_possibilities = self.get_n_degrees_possibilities()
        self.factor = 4
        self.nn_model = None
        self.streak = list()

    def solve(self):
        """
        * Solve a particular level by shooting birds directly to pigs
        * @return GameState: the game state after shots.
        """
        self.logger.info("sovling level %s" % (self.current_level))
        # TODO: sagiv
        ground_truth_type = 'noisy_ground_truth'
        ground_truth_type = GroundTruthType.ground_truth_screenshot
        game_state = self.ar.get_game_state()

        #####################################################################
        # for skipping the level
        # if agents receives over 10 gt truth per level, skip the level

        if game_state != GameState.PLAYING:
            return game_state,-1

        else:
            self.repeated_gt_counter += 1
        ################################################################

        vision = self._update_reader(ground_truth_type.value, self.if_check_gt)
        object_hot_matrix = self.get_hot_object_matrix(vision)

        image_shpae = self.get_image_shape(vision)
        model = self.create_NN(object_hot_matrix.shape)

        angles = model.predict(np.expand_dims(object_hot_matrix, 0))
        angle_indexes = np.argmax(angles, -1)
        release_angle = self.get_degree(possibility=np.max(angle_indexes))

        if not vision.is_vaild():
            self.logger.info("no pig or birds found")
            return self.ar.get_game_state()

        # if self.show_ground_truth:  # TODO: check what is show_ground_truth, refotmat it, it does not work
        #     vision.showResult()

        sling = vision.find_slingshot_mbr()[0]
        # TODO: look into the width and height issue of Traj planner, a bug of replaced height and width
        sling.width, sling.height = sling.height, sling.width
        #
        # # get all the pigs
        pigs = vision.find_pigs_mbr()
        # self.logger.info("no of pigs: " + str(len(vision.find_pigs_mbr())))
        # for pig in pigs:
        #     self.logger.info("pig location: " + str(pig.get_centre_point()))  # Get pigs center location
        # state = self.ar.get_game_state()

        # TODO: move to a module
        # if there is a sling, then play, otherwise skip.
        if sling != None:
            # If there are pigs, we pick up a pig randomly and shoot it.
            if pigs:
                # # TODO change computer_vision.cv_utils.Rectangle
                ################estimate the trajectory###################
                self.logger.info('################estimate the trajectory###################')

                release_point = self.tp.find_release_point(sling, release_angle * pi/180.0)  #neeed to tranfer to radians
                tap_time = 0
                if release_point != None:
                    # release_angle = self.tp.get_release_angle(sling, release_point)
                    self.logger.info("Release Point: %s" % release_point)
                    self.logger.info("Release Angle: %s" % release_angle)
                    print(release_angle)

                    # tap_time = self.get_taptime(release_point, _tpt, vision, sling)
                else:
                    self.logger.warning("No Release Point Found")
                    return self.ar.get_game_state()


                vision = self._update_reader(ground_truth_type.value, self.if_check_gt)

                if isinstance(vision, GameState):
                    return vision
                # if self.show_ground_truth:
                #     vision.showResult()

                _sling = vision.find_slingshot_mbr()[0]
                _sling.width, _sling.height = _sling.height, _sling.width

                if _sling != None:
                    hint = self.ar.get_novelty_hint(0)
                    print(hint)
                    hint = self.ar.get_novelty_hint(1)
                    print(hint)
                    hint = self.ar.get_novelty_hint(2)
                    print(hint)
                    hint = self.ar.get_novelty_hint(3)
                    print(hint)

                    batch_gt = self.ar.shoot_and_record_ground_truth(release_point.X, release_point.Y, 0,
                                                                                 tap_time, 1, 0)
                    self.shot_done = True
                    vision = self._update_reader(ground_truth_type.value, self.if_check_gt)
                    time.sleep(2 / self.sim_speed)
                    state = self.ar.get_game_state()
                    if state == GameState.PLAYING:
                        vision = self._update_reader(ground_truth_type.value, self.if_check_gt)
                        if isinstance(vision, GameState):
                            return vision
                                # if self.show_ground_truth:
                                #     vision.showResult()
                    else:
                        self.logger.info("Scale is changed, can not execute the shot, will re-segement the image")
                else:
                    self.logger.info("no sling detected, can not execute the shot, will re-segement the image")
        return (state, release_angle)

    @property
    def segments(self):
        return {
            GameObjectType.ICE: [GameObjectType.ICE],
            GameObjectType.WOOD: [GameObjectType.WOOD],
            GameObjectType.STONE: [GameObjectType.STONE],
            "unbreak": [GameObjectType.HILL, GameObjectType.SLING],
            "tnt": [GameObjectType.TNT],
            GameObjectType.PIG: [GameObjectType.PIG],
        }


    def get_hot_object_matrix(self, vision: GroundTruthReader):
        object_hot_m = dict()  # Get dimension and remove RGB
        for segment in self.segments:
            hot_m = np.zeros(shape=self.get_image_shape(vision))
            for game_object_type in self.segments[segment]:
                try:
                    for game_object in vision.allObj[game_object_type.value]:
                        hot_m = self.fill_hot_matrix(game_object, hot_m)
                except KeyError:
                    pass
            object_hot_m[segment] = hot_m
        # reduce matrix by factor
        # for object in object_hot_m:
        #     object_hot_m[object] = object_hot_m[object][::self.factor, ::self.factor]
        return np.dstack(object_hot_m.values())

    def fill_hot_matrix(self, game_object: GameObject, hot_m: np.ndarray):
        # TODO: check for not rectangualr data
        for i in range(game_object.top_left[0], game_object.bottom_right[0],self.factor):
            for j in range(game_object.top_left[1], game_object.bottom_right[1],self.factor):
                hot_m[j//self.factor, i//self.factor, 0] = 1
        return hot_m

    def create_NN(self, shape: tuple):
        if self.nn_model is not None:
            return self.nn_model
        fst_dim = shape
        sec_dim = tuple(int(x / 2) for x in shape[:-1]) + (shape[-1],)
        thd_dim = tuple(int(x / 4) for x in shape[:-1]) + (shape[-1],)
        model = tf.keras.models.Sequential([
            Conv2D(16, 3,batch_size=1, input_shape=shape),
            Dropout(.15),
            MaxPool2D((3, 3)),
            Activation("relu"),

            Conv2D(16, 3, batch_size=1, input_shape=sec_dim),
            Dropout(.20),
            MaxPool2D((5, 5)),
            Activation('relu'),

            Conv2D(16, 3, batch_size=1, input_shape=thd_dim),
            Dropout(.30),
            MaxPool2D((3, 3)),
            Activation('relu'),

            Dense(1024),
            Dense(self.deg_possibilities),  # number of possibilities
            Softmax()
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        self.nn_model = model
        return model

    def get_n_degrees_possibilities(self):
        return int((self.max_deg - self.min_deg) / self.deg_step)

    def get_degree(self, possibility: int):
        return self.min_deg + self.deg_step * possibility

    def get_image_shape(self, vision):
        return tuple(int(dim/self.factor) for dim in vision.screenshot.shape)[:-1] + (1,)


    def run(self):
        self.ar.configure(self.id)
        #do not use observer
        #self.observer_ar.configure(self.id)
        self.ar.set_game_simulation_speed(self.sim_speed)
        #n_levels = self.update_no_of_levels()

        #self.solved = [0 for x in range(n_levels)]

        #load next available level
        #self.current_level = self.ar.load_next_available_level()
        #self.novelty_existence = self.ar.get_novelty_info()

        '''
        Uncomment this section to run TEST for requesting groudtruth via different thread
        '''
        #gt_thread = Thread(target=self.sample_state)
        #gt_thread.start()
        '''
        END TEST
        '''

        change_from_training = False

        while True:
            (state, release_angle) = self.solve()
            #If the level is solved , go to the next level
            if state == GameState.WON:
                self.repeated_gt_counter = 0
                #check for change of number of levels in the game
                n_levels = self.update_no_of_levels()

                #/System.out.println(" loading the level " + (self.current_level + 1) )
                #self.check_current_level_score()
                self.current_level = self.ar.load_next_available_level()
                self.novelty_existence = self.ar.get_novelty_info()

                # make a new trajectory planner whenever a new level is entered
                self.tp = SimpleTrajectoryPlanner()

            elif state == GameState.LOST:
                self.repeated_gt_counter = 0
                #check for change of number of levels in the game
                #n_levels = self.update_no_of_levels()

                self.check_current_level_score()

                #If lost, then restart the level
                self.failed_counter += 1
                if self.failed_counter > 0: #for testing , go directly to the next level

                    self.failed_counter = 0
                    self.current_level = self.ar.load_next_available_level()
                    self.novelty_existence = self.ar.get_novelty_info()

                else:
                    self.logger.info("fail level count does not reach the limit, restart the level")
                    self.ar.restart_level()

            elif state == GameState.LEVEL_SELECTION:
                self.logger.info("unexpected level selection page, go to the last current level : " \
                , self.current_level)
                self.current_level = self.ar.load_next_available_level()
                self.novelty_existence = self.ar.get_novelty_info()

            elif state == GameState.MAIN_MENU:
                self.repeated_gt_counter = 0
                self.logger.info("unexpected main menu page, reload the level : %s"%self.current_level)
                self.current_level = self.ar.load_next_available_level()
                self.novelty_existence = self.ar.get_novelty_info()

            elif state == GameState.EPISODE_MENU:
                self.logger.info("unexpected episode menu page, reload the level:  %s"%self.current_level)
                self.current_level = self.ar.load_next_available_level()
                self.novelty_existence = self.ar.get_novelty_info()

            elif state == GameState.REQUESTNOVELTYLIKELIHOOD:
                #Require report novelty likelihood and then playing can be resumed
                #dummy likelihoods:
                self.logger.info("received request of novelty likelihood")
                novelty_likelihood=0.9
                non_novelty_likelihood=0.1
                novel_obj_ids = {1,-2,-398879789}
                novelty_level = 0
                novelty_description = "";
                self.ar.report_novelty_likelihood(novelty_likelihood,non_novelty_likelihood,novel_obj_ids,novelty_level,novelty_description)

            elif state == GameState.NEWTRIAL:
                self.repeated_gt_counter = 0
                #Make a fresh agents to continue with a new trial (evaluation)
                self.logger.critical("new trial state received")
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo) = self.ar.ready_for_new_set()
                self.current_level = 0
                self.training_level_backup = 0

            elif state == GameState.NEWTESTSET:
                self.repeated_gt_counter = 0
                self.logger.critical("new test set state received")
                #DO something to clone a test-only agents that does not learn
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo) = self.ar.ready_for_new_set()

                if change_from_training:
                    self.training_level_backup = self.current_level
                self.current_level = 0
                change_from_training = False

            elif state == GameState.NEWTRAININGSET:
                self.repeated_gt_counter = 0
                #DO something to start a fresh agents for a new training set
                self.logger.critical("new training set state received")
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo) = self.ar.ready_for_new_set()
                self.current_level = 0
                self.training_level_backup = 0
                change_from_training = True

            elif state == GameState.RESUMETRAINING:
                self.repeated_gt_counter = 0
                #DO something to resume the training agents to the previous training
                self.logger.critical("resume training set state received")
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo) = self.ar.ready_for_new_set()
                change_from_training = True
                self.current_level = self.training_level_backup

            elif state == GameState.EVALUATION_TERMINATED:
                #store info and disconnect the agents as the evaluation is finished
                self.logger.critical("Evaluation terminated.")
                exit(0)
            if state == GameState.WON or state ==GameState.LOST:
                v = 1 if state == GameState.WON else 0
                self.streak.append((release_angle, v))
            if len(self.streak) == 10: # epoch
                self.train_model()
    def train_model(self):
        x=0