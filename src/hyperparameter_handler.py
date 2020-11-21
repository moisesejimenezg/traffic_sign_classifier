class HyperParameters:
    def __init__(self, epochs, learning_rate, low_keep_prob, high_keep_prob, sigma):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.low_keep_prob = low_keep_prob
        self.high_keep_prob = high_keep_prob
        self.sigma = sigma
