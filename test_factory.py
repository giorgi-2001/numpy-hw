import pytest
import numpy as np
from factory import SoundWaveFactory


@pytest.fixture
def factory():
    return SoundWaveFactory()


@pytest.fixture
def timeline(factory):
    return factory.timeline


def test_create_note_wave(factory: SoundWaveFactory, timeline):
    timeline = factory.timeline
    wave = factory.create_note_wave('a4', timeline)
    assert len(wave) > 0
    assert np.max(wave) <= 2**13
    assert np.min(wave) >= -2**13


def test_invalid_note_wave(factory: SoundWaveFactory, timeline):
    with pytest.raises(ValueError):
        timeline = factory.timeline
        factory.create_note_wave('invalid_note', timeline)


def test_save_and_read_wave(factory: SoundWaveFactory, tmp_path):
    assert tmp_path.exists()
    wave = factory.wave
    file_path = tmp_path / "wave.txt"
    factory.to_file(str(tmp_path) + "/wave", 'txt')
    assert file_path.exists()
    loaded_wave = factory.read_wave(str(file_path))
    assert np.array_equal(wave, loaded_wave)


def test_wave_details(factory: SoundWaveFactory):
    factory.print_wave_details()


def test_normalize_waves(factory: SoundWaveFactory, timeline):
    wave1 = factory.create_note_wave('a4', timeline)
    wave2 = factory.create_note_wave('b4', timeline)
    normalized = factory.normalize_waves([wave1, wave2])
    assert len(normalized[0]) == len(normalized[1])


def test_combine_waves(factory: SoundWaveFactory, timeline):
    wave1 = factory.create_note_wave('a4', timeline)
    wave2 = factory.create_note_wave('b4', timeline)
    combined = factory.combine_waves([wave1, wave2])
    assert len(combined) == len(wave1) + len(wave2)


def test_adsr(factory: SoundWaveFactory):
    wave = factory.wave
    modified_wave = factory.get_applied_adsr(
        attack=0.1, 
        decay=0.2, 
        sustain_level=0.7, 
        release=0.2
    )
    assert len(modified_wave) == len(wave)
