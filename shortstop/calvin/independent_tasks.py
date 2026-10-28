"""CALVIN's initial world states, for drawing single-task trials."""
from itertools import product


def _enumerate_initial_states():
    """The initial-state population of mdt.evaluation.multistep_sequences.
    get_sequences (copied: that function does not expose it): every
    combination of led/lightbulb/slider/drawer states and block placements
    with 1-2 blocks on the table and at most one per slider side."""
    possible_conditions = {
        "led": [0, 1], "lightbulb": [0, 1], "slider": ["right", "left"],
        "drawer": ["closed", "open"], "red_block": ["table", "slider_right", "slider_left"],
        "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"], "grasped": [0],
    }
    keep = lambda l: l.count("table") in [1, 2] and l.count("slider_right") < 2 and l.count("slider_left") < 2  # noqa: E731
    value_combinations = filter(keep, product(*possible_conditions.values()))
    return [dict(zip(possible_conditions.keys(), vals)) for vals in value_combinations]
