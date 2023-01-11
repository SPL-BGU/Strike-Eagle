
class GameSessionWonException(Exception):
    def __init__(self, reward):
        super().__init__("WON")
        self.reward = reward


class GameSessionLossException(Exception):
    def __init__(self, reward):
        super().__init__("LOSE")
        self.reward = reward
