import numpy as np
from scipy.io import wavfile
from scipy.signal import square, sawtooth


SAMPLING_RATE = 44100  # like 44.1 KHz
DURATION_SECONDS = 5
SOUND_ARRAY_LEN = SAMPLING_RATE * DURATION_SECONDS
MAX_AMPLITUDE = 2 ** 13


# From a list of https://en.wikipedia.org/wiki/Piano_key_frequencies
NOTES = {
    '0': 0, 'e0': 20.60172, 'f0': 21.82676, 'f#0': 23.12465, 'g0': 24.49971, 'g#0': 25.95654, 'a0': 27.50000, 'a#0': 29.13524,
    'b0': 30.86771, 'c0': 32.70320, 'c#0': 34.64783, 'd0': 36.70810, 'd#0': 38.89087,
    'e1': 41.20344, 'f1': 43.65353, 'f#1': 46.24930, 'g1': 48.99943, 'g#1': 51.91309, 'a1': 55.00000, 'a#1': 58.27047,
    'b1': 61.73541, 'c1': 65.40639, 'c#1': 69.29566, 'd1': 73.41619, 'd#1': 77.78175,
    'e2': 82.40689, 'f2': 87.30706, 'f#2': 92.49861, 'g2': 97.99886, 'g#2': 103.8262, 'a2': 110.0000, 'a#2': 116.5409,
    'b2': 123.4708, 'c2': 130.8128, 'c#2': 138.5913, 'd2': 146.8324, 'd#2': 155.5635,
    'e3': 164.8138, 'f3': 174.6141, 'f#3': 184.9972, 'g3': 195.9977, 'g#3': 207.6523, 'a3': 220.0000, 'a#3': 233.0819,
    'b3': 246.9417, 'c3': 261.6256, 'c#3': 277.1826, 'd3': 293.6648, 'd#3': 311.1270,
    'e4': 329.6276, 'f4': 349.2282, 'f#4': 369.9944, 'g4': 391.9954, 'g#4': 415.3047, 'a4': 440.0000, 'a#4': 466.1638,
    'b4': 493.8833, 'c4': 523.2511, 'c#4': 554.3653, 'd4': 587.3295, 'd#4': 622.2540,
    'e5': 659.2551, 'f5': 698.4565, 'f#5': 739.9888, 'g5': 783.9909, 'g#5': 830.6094, 'a5': 880.0000, 'a#5': 932.3275,
    'b5': 987.7666, 'c5': 1046.502, 'c#5': 1108.731, 'd5': 1174.659, 'd#5': 1244.508,
    'e6': 1318.510, 'f6': 1396.913, 'f#6': 1479.978, 'g6': 1567.982, 'g#6': 1661.219, 'a6': 1760.000, 'a#6': 1864.655,
    'b6': 1975.533, 'c6': 2093.005, 'c#6': 2217.461, 'd6': 2349.318, 'd#6': 2489.016,
    'e7': 2637.020, 'f7': 2793.826, 'f#7': 2959.955, 'g7': 3135.963, 'g#7': 3322.438, 'a7': 3520.000, 'a#7': 3729.310,
    'b7': 3951.066, 'c7': 4186.009, 'c#7': 4434.922, 'd7': 4698.636, 'd#7': 4978.032,
}


# get_normed_sin = lambda timeline, frequency: MAX_AMPLITUDE * np.sin(2 * np.pi * frequency * timeline)
# get_soundwave = lambda timeline, note: get_normed_sin(timeline, NOTES[note])
# common_timeline = np.linspace(0, DURATION_SECONDS, num=SOUND_ARRAY_LEN)


class SoundWaveFactory:

    def __init__(
            self,
            note: str = "a4",
            wave_type="sine",
            duration=DURATION_SECONDS
        ):
        self.timeline = self._generate_timeline(duration)
        self.wave = self.create_note_wave(
            note=note,
            timeline=self.timeline,
            wave_type=wave_type
        )
        
    @staticmethod
    def _generate_timeline( duration):
        return np.linspace(
            0, duration,
            num=int(SAMPLING_RATE * duration), 
            endpoint=False
        )

    @classmethod
    def _generate_wave(cls, frequency, timeline, wave_type="sine"):
        if wave_type == "sine":
            return MAX_AMPLITUDE * np.sin(2 * np.pi * frequency * timeline)
        elif wave_type == "square":
            return MAX_AMPLITUDE * square(2 * np.pi * frequency * timeline)
        elif wave_type == "triangle":
            return MAX_AMPLITUDE * sawtooth(2 * np.pi * frequency * timeline, 0.5)
        raise ValueError(
            f"Invalid wave type '{wave_type}'. "
            f"Use 'sine', 'square', or 'triangle'."
        )

    @classmethod
    def create_note_wave(cls, note: str, timeline, wave_type="sine"):
        if note not in NOTES:
            raise ValueError(f"Invalid note '{note}'.")
        frequency = NOTES[note]
        return cls._generate_wave(frequency, timeline, wave_type).astype(np.int16)

    @staticmethod
    def save_wave(wave, filename,  file_type='txt'):
        if file_type.lower() == 'wav':
            wavfile.write(f"{filename}.wav", SAMPLING_RATE, wave)
        elif file_type.lower() == 'txt':
            np.savetxt(f"{filename}.txt", wave)
        else:
            raise ValueError("Unsupported file type. Choose 'txt' or 'wav'.")
        
    def to_file(self, filename, file_type='txt'):
        self.save_wave(
            wave=self.wave,
            filename=filename,
            file_type=file_type
        )
    
    @staticmethod
    def read_wave(filename):
        try:
            return np.loadtxt(filename, dtype=np.int16)
        except Exception as e:
            raise IOError(f"Error reading wave from {filename}: {e}")

    def print_wave_details(self):
        print(
            f"Wave Details:\n - Length: {len(self.wave)}\n" 
            f"- Max Amplitude: {np.max(self.wave)}\n"
            f"- Min Amplitude: {np.min(self.wave)}"
        )

    @staticmethod
    def normalize_waves(waves):
        if not waves:
            raise ValueError("No waves provided for normalization.")
        min_length = min(len(wave) for wave in waves)
        normalized_waves = [(wave[:min_length] * (MAX_AMPLITUDE / np.max(np.abs(wave)))).astype(np.int16) for wave in waves]
        return np.array(normalized_waves)

    @staticmethod
    def combine_waves(wave_sequence):
        return np.concatenate(wave_sequence).astype(np.int16)

    @classmethod
    def read_note_sequence(cls, note_sequence: str):
        waves = []
        tokens = note_sequence.split(" ")
        
        i = 0
        while i < len(tokens):
            if tokens[i].startswith('('):
                chord = []
                for j in range(i, len(tokens)):
                    chord.append(tokens[j].strip("()"))
                    if tokens[j].endswith(")"):
                        break
                dur_index = i + len(chord)
                duration = float(tokens[dur_index].strip('s'))
                new_timeline = cls._generate_timeline(duration)
                wave = sum(cls.create_note_wave(note, new_timeline) for note in chord)
                waves.append(wave)
                i += len(chord) + 1
            else:
                note = tokens[i]
                duration = float(tokens[i + 1].strip('s'))
                new_timeline = cls._generate_timeline(duration)
                print(new_timeline)
                wave = cls.create_note_wave(note, new_timeline, wave_type="sine")
                wave = cls._apply_adsr(wave)
                waves.append(wave)
                i += 2  

        combined_waves = cls.combine_waves(waves)
        return combined_waves

    @classmethod
    def _apply_adsr(cls, wave, attack=0.1, decay=0.2, sustain_level=0.7, release=0.2):
        total_samples = len(wave)
        adsr_env = np.ones(total_samples)

        attack_samples = min(int(attack * SAMPLING_RATE), total_samples)
        decay_samples = min(int(decay * SAMPLING_RATE), total_samples - attack_samples)
        sustain_samples = total_samples - (attack_samples + decay_samples)
        release_samples = min(int(release * SAMPLING_RATE), sustain_samples)
        
        adsr_env[:attack_samples] = np.linspace(0, 1, attack_samples)
        adsr_env[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples)
        adsr_env[-release_samples:] = np.linspace(sustain_level, 0, release_samples)

        return (wave * adsr_env).astype(np.int16)

    def get_applied_adsr(
        self, attack, decay, sustain_level, release
    ):
        return self._apply_adsr(
            self.wave, attack, decay, sustain_level, release
        )
