import time

from utils import Logger

class Timer(object):
    def __init__(self, name, logger: Logger):
        self.logger = logger
        self.name = name
        self.running = True
        self.total = 0
        self.start = round(time.time(), 2)
        self.intervalTime = round(time.time(), 2)
        self.logger.log(f"<> <> <> Starting Timer [{self.name}] <> <> <>")

    def reset(self):
        self.running = True
        self.total = 0
        self.start = round(time.time(), 2)
        return self

    def interval(self, intervalName=''):
        intervalTime = self._to_hms(round(time.time() - self.intervalTime, 2))
        self.logger.log(f"<> <> Timer [{self.name}] <> <> Interval [{intervalName}]: {intervalTime} <> <>")
        self.intervalTime = round(time.time(), 2)
        return intervalTime

    def stop(self):
        if self.running:
            self.running = False
            self.total += round(time.time() - self.start, 2)
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = round(time.time(), 2)
        return self

    def time(self):
        if self.running:
            return round(self.total + time.time() - self.start, 2)
        return self.total

    def finish(self):
        if self.running:
            self.running = False
            self.total += round(time.time() - self.start, 2)
            elapsed = self._to_hms(self.total)
            self.logger.log(f"<> <> <> Finished Timer [{self.name}] <> <> <> Total time elapsed: {elapsed} <> <> <>")

    def _to_hms(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%dh %02dm %02ds" % (h, m, s)
