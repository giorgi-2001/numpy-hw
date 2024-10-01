from factory import SoundWaveFactory


new_factory = SoundWaveFactory()

melody_sequence = "c4 0.5s e4 0.5s g4 0.5s (c4 e4 g4) 1s a4 0.5s b4 0.5s c5 0.5s d5 0.5s e5 0.5s (g4 b4 d5) 1s f5 0.5s e5 0.5s d5 0.5s c5 0.5s (e4 g4 b4) 1s a4 0.5s (g4 e4 c4) 0.5s"


sound = new_factory.read_note_sequence(melody_sequence)


new_factory.save_wave(sound, "./saves/sound", "wav")