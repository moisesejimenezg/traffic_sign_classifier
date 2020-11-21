import tensorflow.compat.v1 as tf


class SessionWrapper:
    def __init__(self, name):
        self.name = name
        self.saver = tf.train.Saver()

    def write(self, session):
        self.saver.save(session, self.name)
        print("Model written.")

    def read(self, session):
        self.saver.restore(session, self.name)
        print("Model read.")
