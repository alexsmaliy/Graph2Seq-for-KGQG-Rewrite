class AverageMeter(object):
    def __init__(self):
        self.history = []
        self.last = None
        self.val = self.sum = self.count = 0

    def reset(self):
        self.last = self.mean()
        self.history.append(self.last)
        self.val = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def mean(self):
        return (self.sum / self.count) if self.count != 0 else 0.
