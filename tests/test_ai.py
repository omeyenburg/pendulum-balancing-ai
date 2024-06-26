import ai


def test_activation_functions():
    a = ai.ActivationFunction

    assert ai._activation_function("relu") == a.relu
    assert ai._activation_function("tanh") == a.tanh
    assert ai._activation_function("sigmoid") == a.sigmoid

    assert a.relu(2) == 2
    assert a.relu(-1) == 0
    assert a.relu(0) == 0

    assert a.sigmoid(0) == 0.5
    assert 0 < a.sigmoid(-1) < 0.5
    assert 0.5 < a.sigmoid(1) < 1

    assert a.tanh(0) == 0
    assert -1 < a.tanh(-1) < 0
    assert 0 < a.tanh(1) < 1


def test_generation_saving():
    assert (
        ai.get_newest_generation(
            ["gen5.json", "gen-1.json", "gen.json", "gen3.json", "3.txt", "1"]
        )
        == 5
    )


def test_time_conversion():
    assert ai.seconds_to_str(1) == "1 second"
    assert ai.seconds_to_str(2) == "2 seconds"
    assert ai.seconds_to_str(60) == "1 minute"
    assert ai.seconds_to_str(3600) == "1 hour"
    assert ai.seconds_to_str(3736) == "1 hour, 2 minutes"
    assert ai.seconds_to_str(86400) == "1 day"
    assert ai.seconds_to_str(270000) == "3 days, 3 hours"
