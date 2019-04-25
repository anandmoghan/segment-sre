from sklearn.metrics import confusion_matrix

import numpy as np


class BaseScorer:
    def __init__(self, labels=None):
        if labels is None:
            labels = [0, 1]
        self.count = 0
        self.labels = labels
        self.total_loss = 0.
        self.num_labels = len(labels)
        self.scores = np.zeros([self.num_labels, self.num_labels], dtype=int)
        self.target_count = 0

    def accuracy(self):
        return self.scores.diagonal().sum() / self.count

    def add(self, true_labels, predicted_labels, loss=0.):
        self.count += len(true_labels)
        self.total_loss += loss * len(true_labels)
        self.scores += confusion_matrix(true_labels, predicted_labels, labels=self.labels)  # tn, fp, fn, tp
        self.target_count += np.sum(true_labels)

    def false_alarm(self):
        return self.scores[0, self.num_labels - 1] / (self.count - self.target_count)

    def loss(self):
        return self.total_loss / self.count

    def miss(self):
        return self.scores[self.num_labels - 1, 0] / self.target_count

    def reset(self):
        self.count = 0
        self.total_loss = 0.
        self.scores = np.zeros([self.num_labels, self.num_labels], dtype=int)
        self.target_count = 0


class ScoreHolder(BaseScorer):
    def __init__(self, labels=None):
        super().__init__(labels)
        self.batch_score = BaseScorer(labels=labels)

    def add(self, true_labels, predicted_labels, loss=0.):
        super().add(true_labels, predicted_labels, loss)
        self.batch_score.add(true_labels, predicted_labels, loss)

    def batch_accuracy(self):
        return self.batch_score.accuracy()

    def batch_false_alarm(self):
        return self.batch_score.false_alarm()

    def batch_loss(self):
        return self.batch_score.loss()

    def batch_miss(self):
        return self.batch_score.miss()

    def batch_start(self):
        return self.batch_score.reset()

    def reset(self):
        super().reset()
        self.batch_start()
