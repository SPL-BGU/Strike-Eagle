from agents.pddl.pddl_files.world_model.params import Params
from agents.pddl.pddl_files.world_model.world_model import WorldModel
from src.computer_vision.GroundTruthReader import GroundTruthReader
from src.computer_vision.game_object import GameObjectType


def get_birds(vision, sling, tp,agent_world_model: WorldModel):
    BIRD_TYPES = [GameObjectType.REDBIRD, GameObjectType.YELLOWBIRD, GameObjectType.BLACKBIRD,
                  GameObjectType.WHITEBIRD, GameObjectType.BLUEBIRD]
    bird_id = 0
    problem_data = dict()
    birds_types = vision.find_birds()
    ref = tp.get_reference_point(sling)
    for bird_type, birds in birds_types.items():
        for bird in birds:
            problem_data[f"bird_{bird_id}"] = {
                "x_bird": ref.X,
                # "x_bird": 89,
                "y_bird": 640 - ref.Y,
                # "y_bird": 377,
                "bird_id": bird_id,
                "bird_type": BIRD_TYPES.index(GameObjectType(bird_type)),
                "m_bird": bird.width * bird.height,  # check this because it is not mandatory
                "bird_radius": min(bird.width, bird.height) / 2,  # check this
                "v_bird": agent_world_model.hyperparams_values[Params.velocity],
                "bounce_count": 0,

                # "v_bird": 190.5
            }
            bird_id += 1
    return problem_data


def get_pigs(vision, sling, tp):
    pig_id = 0
    problem_data = dict()
    pigs = vision.find_pigs_mbr()
    for pig in pigs:
        temp_pt = pig.get_centre_point()

        # TODO change computer_vision.cv_utils.Rectangle
        # to be more intuitive
        problem_data[f"pig_{pig_id}"] = {
            "x_pig": pig.X + pig.width,
            # "y_pig": 640 - pig.Y - 354,
            "y_pig": 640 - pig.Y - pig.height / 2,
            "m_pig": pig.width * pig.height,  # check this because it is not mandatory
            "pig_radius": min(pig.width, pig.height) / 2,  # check this
            "pig_life": 1  # check

        }
        pig_id += 1

    return problem_data


def get_blocks(vision, sling, tp):
    block_types = vision.find_blocks()
    x = 0
    blocks_data = {
        'wood': {
            'life': 0.75,
            'mass_coef': 0.375,
            'multi': 1
        },
        'ice': {
            'life': 0.75,
            'mass_coef': 0.375,
            'multi': 0.5
        },
        'TNT': {
            'life': 0.75,
            'mass_coef': 0.375,
            'multi': 0.5
        },

        'stone': {
            'life': 1.2,
            'mass_coef': 0.375,
            'multi': 2
        }
    }
    problem_data = dict()

    block_id = 0
    if not block_types:
        return {}
    for block_type, blocks in block_types.items():
        for block in blocks:
            problem_data[f"block_{block_id}"] = {
                "x_block": block.X + block.width / 2,
                "y_block": 640 - block.Y - block.height / 2,
                "block_width": block.width,
                "block_height": block.height,
                "block_life": blocks_data[block_type]['life'] * blocks_data[block_type]['multi'],
                "block_mass": block.width * block.height * blocks_data[block_type]['mass_coef'],
                "block_stability": 1

            }
            block_id += 1

    return problem_data



def get_platforms(vision: GroundTruthReader, sling, tp):
    platform_id = 0
    problem_data = dict()
    platforms = vision.find_hill_mbr() #this how do they call platforms apperently
    if not platforms:
        return {}
    for platform in platforms:
        # to be more intuitive
        platform.width,platform.height = platform.height, platform.width
        problem_data[f"platform_{platform_id}"] = {
            "x_platform": platform.X + platform.width/2,
            "y_platform": 640 - platform.Y - platform.height / 2,
            "platform_width": platform.width, # objects get mixed
            "platform_height": platform.height

        }
        platform_id += 1

    return problem_data
