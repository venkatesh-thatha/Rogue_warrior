"""Shared action IDs for the game and teaching models."""

from enum import IntEnum


class Action(IntEnum):
    """Keep these IDs consistent for human controls and future RL agents."""

    ACCELERATE = 0
    ACCELERATE_LEFT = 1
    ACCELERATE_RIGHT = 2
    BRAKE = 3
    COAST = 4
    PUNCH_LEFT = 5
    PUNCH_RIGHT = 6
    KICK = 7
    DEFEND = 8
    ACCELERATE_ATTACK_LEFT = 9
    ACCELERATE_ATTACK_RIGHT = 10


TEACHING_ACTIONS = (Action.ACCELERATE, Action.ACCELERATE_LEFT, Action.ACCELERATE_RIGHT)
DRIVING_ACTIONS = (*TEACHING_ACTIONS, Action.BRAKE, Action.COAST)
