import inspect
import json
import os
import time

import config

CONFIG_FNAME = 'config.file'
METRICS_FNAME = 'metrics.log'
RUNLOG_FNAME = 'run.log'


class Logger(object):
    def __init__(self, dirpath):
        timestamp = int(time.time())
        logdir = os.path.join(dirpath, f"{timestamp}")
        os.makedirs(logdir, exist_ok=True)

        self.dirpath = logdir
        self.metrics_log = os.path.join(logdir, METRICS_FNAME)
        self.run_log = os.path.join(logdir, RUNLOG_FNAME)

        self._log_config()

    def log(self, stuff, fpath, echo=True, as_json=False):
        if as_json:
            stuff = json.dumps(stuff, indent=3, ensure_ascii=False)
        if echo:
            print(stuff)
        with open(fpath, "w") as f:
            f.write(f"{stuff}\n")

    def _log_config(self):
        run_config = (
            (k, v) for (k, v)
            in inspect.getmembers(config)
            if not k.startswith("__") and not inspect.ismodule(v)
        )
        run_config = sorted(run_config, key=lambda kv: kv[0])
        run_config = "\n".join(f"{k} = {v}" for k, v in run_config)
        self.log(
            run_config,
            os.path.join(self.dirpath, CONFIG_FNAME),
            echo=True,
        )
