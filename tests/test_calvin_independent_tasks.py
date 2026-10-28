from shortstop.calvin.independent_tasks import _enumerate_initial_states


def test_initial_states_follow_calvin_rules():
    states = _enumerate_initial_states()
    assert len(states) > 0
    blocks = ("red_block", "blue_block", "pink_block")
    for s in states:
        assert s["grasped"] == 0
        places = [s[b] for b in blocks]
        assert places.count("table") in (1, 2)
        assert places.count("slider_right") < 2 and places.count("slider_left") < 2
