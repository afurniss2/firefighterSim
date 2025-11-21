"""
Run with:
    python tests.py
"""

import math

from model import WildfireModel


def make_params(
    width=10,
    height=10,
    torus=False,
    ignitions=1,
    base_spread_p=0.3,
    firefighters=4,
    perception_r=4,
    comm_r=4,
    comm_freq=2,
    max_water=5,
    max_ticks=200,
    random_seed=42,
    log_dir="output_test",
):
    """Create a plain dict of params compatible with WildfireModel."""
    return {
        # world geometry
        "width": width,
        "height": height,
        "torus": torus,

        # fire environment
        "ignitions": ignitions,
        "base_spread_p": base_spread_p,
        # unused but included for completeness
        "intensity_decay": 0.25,
        "reignite_allowed": False,
        "wind_dx": 0,
        "wind_dy": 0,
        "wind_bias": 0.0,

        # firefighters
        "firefighters": firefighters,
        "perception_r": perception_r,
        "comm_r": comm_r,
        "comm_freq": comm_freq,
        "max_water": max_water,

        # run control
        "max_ticks": max_ticks,
        "random_seed": random_seed,
        "log_dir": log_dir,
    }


def test_fire_burns_out_without_firefighters():
    """
    With no spread and no firefighters, a single ignition should burn and then burn out.
    """
    params = make_params(
        ignitions=1,
        base_spread_p=0.0,   # no spread
        firefighters=0,      # no agents
        max_ticks=20,
        random_seed=1,
        log_dir="output_test_no_ff",
    )
    model = WildfireModel(params)
    model.run()

    burning = model.count_burning()
    burnt = model.count_burnt()
    extinguished = model.count_extinguished()

    assert burning == 0, "Fire should not be burning at end of run"
    assert burnt >= 1, "At least one cell should have burnt"
    assert extinguished == 0, "No firefighters => nothing should be extinguished"

    print("[OK] test_fire_burns_out_without_firefighters")


def test_firefighters_extinguish_some_fire():
    """
    With firefighters and a modest spread probability, we expect them to
    extinguish at least some cells and eventually contain the fire.
    """
    params = make_params(
        ignitions=2,
        base_spread_p=0.3,
        firefighters=6,
        perception_r=4,
        comm_r=4,
        comm_freq=2,
        max_water=5,
        max_ticks=200,
        random_seed=2,
        log_dir="output_test_ff_basic",
    )
    model = WildfireModel(params)
    model.run()

    burnt = model.count_burnt()
    extinguished = model.count_extinguished()

    # Key expectations:
    # - firefighters do extinguish something
    # - the fire gets contained in finite time
    assert extinguished > 0, "Firefighters should extinguish at least one cell"

    assert not math.isnan(model.contained_at), "Fire should be contained (finite contained_at)"
    assert model.contained_at <= params["max_ticks"], "Containment time should be within max_ticks"

    print(
        "[INFO] burnt =", burnt,
        "extinguished =", extinguished,
        "contained_at =", model.contained_at,
    )
    print("[OK] test_firefighters_extinguish_some_fire")



def test_communications_get_sent():
    """
    Messages get sent when comms enabled
    """
    base_common = {
        "width": 20,
        "height": 20,
        "ignitions": 3,
        "base_spread_p": 0.35,
        "firefighters": 10,
        "perception_r": 4,
        "max_water": 6,
        "max_ticks": 200,
        "random_seed": 3,
    }

    # No comms
    params_no_comm = make_params(
        **base_common,
        comm_r=0,
        comm_freq=0,
        log_dir="output_test_no_comm",
    )
    model_no_comm = WildfireModel(params_no_comm)
    model_no_comm.run()

    # With comms
    params_comm = make_params(
        **base_common,
        comm_r=6,
        comm_freq=2,
        log_dir="output_test_with_comm",
    )
    model_comm = WildfireModel(params_comm)
    model_comm.run()

    # Both should contain the fire
    assert not math.isnan(model_no_comm.contained_at), "No-comm run should still contain the fire"
    assert not math.isnan(model_comm.contained_at), "Comm run should contain the fire"

    # With comms we expect some messages
    assert model_comm.messages > 0, "With communication enabled, messages should be sent"

    print(
        "[INFO] contained_at (no comm)  =",
        model_no_comm.contained_at,
    )
    print(
        "[INFO] contained_at (with comm) =",
        model_comm.contained_at,
    )
    print("[OK] test_communications_get_sent")


def run_all_tests():
    test_fire_burns_out_without_firefighters()
    test_firefighters_extinguish_some_fire()
    test_communications_get_sent()
    print("\nAll tests completed.")


if __name__ == "__main__":
    run_all_tests()
