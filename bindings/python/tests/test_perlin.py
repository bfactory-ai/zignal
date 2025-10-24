from __future__ import annotations

import math

import pytest

import zignal


MAX_OCTAVES = 32
MAX_LACUNARITY = 16.0


def test_perlin_defaults_and_amplitude_scaling():
    base = zignal.perlin(0.125, 0.5, 0.25)
    assert isinstance(base, float)

    scale = 7.5
    scaled = zignal.perlin(0.125, 0.5, 0.25, amplitude=scale)
    assert scaled == pytest.approx(base * scale)


def test_perlin_accepts_custom_parameters():
    value = zignal.perlin(
        0.2,
        0.4,
        0.1,
        amplitude=1.2,
        frequency=2.5,
        octaves=3,
        persistence=0.42,
        lacunarity=2.1,
    )
    assert isinstance(value, float)
    # Ensure octaves/persistence influence the result relative to single octave
    single_octave = zignal.perlin(0.2, 0.4, 0.1, octaves=1, persistence=0.5, lacunarity=2.0)
    assert not math.isclose(value, single_octave)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amplitude": 0.0},
        {"frequency": 0.0},
        {"octaves": 0},
        {"persistence": -0.1},
        {"persistence": 1.1},
        {"lacunarity": 0.5},
        {"lacunarity": MAX_LACUNARITY + 1},
        {"octaves": MAX_OCTAVES + 1},
    ],
)
def test_perlin_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        zignal.perlin(0.0, 0.0, **kwargs)
