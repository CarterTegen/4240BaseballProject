class SillyPitcher:

    def __init__(self):
        pass

    def throw_pitch(self, state=3):
        return  {
            "start_speed": 90,
            "spin_rate": 1300,
            "spin_dir": 159,
            "px": 0,
            "pz": 2.1,
            "pitch_type": "FF"
        }