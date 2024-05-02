import inspect
import json
import os
import time

import config

CONFIG_FNAME = 'config.file'
METRICS_FNAME = 'metrics.log'
RUNLOG_FNAME = 'run.log'


class Logger(object):
    __timestamp = int(time.time())
    log_dir = os.path.join(config.LOG_DIR, f"{__timestamp}")
    metrics_log = os.path.join(log_dir, METRICS_FNAME)
    run_log = os.path.join(log_dir, RUNLOG_FNAME)

    def __init__(self):
        os.makedirs(Logger.log_dir, exist_ok=True)
        self._record_config()

    def log(self, stuff, fpath=run_log, /, echo=True, as_json=False):
        if as_json:
            stuff = json.dumps(stuff, indent=3, ensure_ascii=False)
        if echo:
            print(stuff)
        with open(fpath, "w") as f:
            f.write(f"{stuff}\n")

    def _record_config(self):
        run_config = (
            (k, v) for (k, v)
            in inspect.getmembers(config)
            if not k.startswith("__") and not inspect.ismodule(v)
        )
        run_config = sorted(run_config, key=lambda kv: kv[0])
        run_config = "\n".join(f"{k} = {v}" for k, v in run_config)
        self.log(
            run_config,
            os.path.join(Logger.log_dir, CONFIG_FNAME),
        )
