import time

import torch
import torch.backends.cudnn as cudnn

import config
from model import Model
from utils import AverageMeter, Dataset, DataStream, get_datasets, Logger, Timer, vectorize_input

class ModelManager(object):
    def __init__(self):
        self._dev_loss = AverageMeter()
        self._train_loss = AverageMeter()

        self._train_metrics = {
            "Bleu_1": AverageMeter(),
            "Bleu_2": AverageMeter(),
            "Bleu_3": AverageMeter(),
            "Bleu_4": AverageMeter(),
            "ROUGE_L": AverageMeter(),
        }
        self._dev_metrics = {
            "Bleu_1": AverageMeter(),
            "Bleu_2": AverageMeter(),
            "Bleu_3": AverageMeter(),
            "Bleu_4": AverageMeter(),
            "ROUGE_L": AverageMeter(),
        }

        self.logger = Logger()
        self.run_log = self.logger.run_log
        self.metrics_log = self.logger.metrics_log

        device_count = torch.cuda.device_count()
        self.logger.log(f"USE_CUDA is True, device count = {device_count}")

        if config.USE_CUDA and device_count > 0:
            self.device = torch.device(f"cuda:{config.CUDA_DEVICE_ID}")
            cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")

        datasets = get_datasets(self.logger)
        train_data: Dataset = datasets["train"]
        dev_data: Dataset = datasets["dev"]
        test_data: Dataset = datasets["test"]

        self.vectorize_input = vectorize_input
        self._n_train_examples = 0
        self.model = Model(train_data, self.device, self.logger)
        self.model.network = self.model.network.to(self.device)

        if train_data:
            self.train_loader = DataStream(
                train_data,
                self.model.vocab_model.word_vocab,
                is_shuffle=True, is_loop=True, is_sort=True, ext_vocab=True,
            )
            self._n_train_batches = self.train_loader.get_num_batch()
        else:
            self.train_loader = None

        if dev_data:
            self.dev_loader = DataStream(
                dev_data,
                self.model.vocab_model.word_vocab,
                is_shuffle=False, is_loop=True, is_sort=True, ext_vocab=True,
            )
            self._n_dev_batches = self.dev_loader.get_num_batch()
        else:
            self.dev_loader = None

        if test_data:
            self.test_loader = DataStream(
                test_data,
                self.model.vocab_model.word_vocab,
                is_shuffle=False, is_loop=False, is_sort=True, batch_size=config.BATCH_SIZE, ext_vocab=True,
            )
            self._n_test_batches = self.test_loader.get_num_batch()
            self._n_test_examples = len(test_data)
        else:
            self.test_loader = None

        self.is_test = False

    def metric_to_str(self, metrics):
        format_str = ""
        for k in metrics:
            format_str += f" | {k.upper()} = {metrics[k].mean():0.3f}"
        return format_str

    def _reset_metrics(self):
        self._train_loss.reset()
        self._dev_loss.reset()

        for k in self._train_metrics:
            self._train_metrics[k].reset()
        for k in self._dev_metrics:
            self._dev_metrics[k].reset()

    def _stop_condition(self, epoch, patience=10):
        """
            Checks have not exceeded max epochs and has not gone patience epochs without improvement.
        """
        no_improvement = epoch >= self._best_epoch + patience
        exceeded_max_epochs = epoch >= config.MAX_EPOCHS
        return False if exceeded_max_epochs or no_improvement else True

    def summary(self):
        start = "<<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        info = f"Best epoch = {self._best_epoch}; {self.metric_to_str(self._best_metrics)}"
        end = " <<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        return "\n".join([start, info, end])

    def train(self):
        self.is_test = False
        timer = Timer("Train", self.logger)

        if config.PRETRAINED:
            self._epoch = self._best_epoch = self.model.saved_epoch
        else:
            self._epoch = self._best_epoch = 0
        self._best_metrics = {}

        for k in self._dev_metrics:
            self._best_metrics[k] = self._dev_metrics[k].mean()

        self._reset_metrics()
        while self._stop_condition(self._epoch, config.PATIENCE):
            self._epoch += 1
            rl_ratio = config.RL_RATIO if self._epoch >= config.RL_START_EPOCH else 0
            self.logger.log(f"RL ratio: {rl_ratio}")

            self.logger.log(f">>> Train Epoch: [{self._epoch} / {config.MAX_EPOCHS}]", self.logger.metrics_log)
            self._run_epoch(self.train_loader, training=True, rl_ratio=rl_ratio, verbose=config.VERBOSE)
            train_epoch_time = timer.interval(f"Training Epoch {self._epoch}")
            metrics_message = f"Training Epoch {self._epoch} -- Loss: {self._train_loss.mean():0.5f}"
            metrics_message += self.metric_to_str(self._train_metrics)
            self.logger.log(metrics_message, self.logger.metrics_log)

            self.logger.log(f">>> Dev Epoch: [{self._epoch} / {config.MAX_EPOCHS}]", self.logger.metrics_log)
            self._run_epoch(self.dev_loader, training=False, verbose=config.VERBOSE)
            timer.interval(f"Validation Epoch {self._epoch}")
            metrics_message = f"Validation Epoch {self._epoch} -- Loss: {self._dev_loss.mean():0.5f}"
            metrics_message += self.metric_to_str(self._dev_metrics)
            self.logger.log(metrics_message, self.logger.metrics_log)

            self.model.scheduler.step(self._dev_metrics[config.EARLY_STOP_METRIC].mean())
            if self._best_metrics[config.EARLY_STOP_METRIC] <= self._dev_metrics[config.EARLY_STOP_METRIC].mean():
                self._best_epoch = self._epoch
                for k in self._dev_metrics:
                    self._best_metrics[k] = self._dev_metrics[k].mean()
                if config.SAVE_PARAMS:
                    self.model.save(self.logger.log_dir, self._epoch)
                    print('Saved model to {}'.format(self.logger.log_dir))
                metrics_message = f"!!! Updated: {self.metric_to_str(self._best_metrics)}"
                self.logger.log(metrics_message, self.logger.metrics_log, echo=False)
                self.logger.log(metrics_message, echo=True)
            self._reset_metrics()
            if rl_ratio > 0:
                config.RL_RATIO = min(config.MAX_RL_RATIO, config.RL_RATIO ** config.RL_RATIO_POWER)

        timer.finish()
        self.training_time = timer.total
        metrics_message = f"Finished Training: {self.logger.log_dir}\n{self.summary()}"
        self.logger.log(metrics_message, self.logger.metrics_log, echo=False)
        self.logger.log(metrics_message, echo=True)
        return self._best_metrics

    def _run_epoch(self, data_loader: DataStream, training=True, rl_ratio=0.0, verbose=10, out_predictions=False):
        start_time = time.time()
        mode = "train" if training else ("test" if self.is_test else "dev")

        if training:
            self.model.optimizer.zero_grad()
        output = []
        gold = []
        for step in range(data_loader.get_num_batch()):
            input_batch = data_loader.next_batch()
            x_batch = self.vectorize_input(input_batch, training=training, device=self.device)
            if not x_batch:
                continue  # When there are no examples in the batch

            forcing_ratio = self._set_forcing_ratio(step) if training else 0
            res = self.model.predict(x_batch, step, forcing_ratio=forcing_ratio, rl_ratio=rl_ratio, update=training, out_predictions=out_predictions, mode=mode)

            loss = res['loss']
            metrics = res['metrics']
            self._update_metrics(loss, metrics, x_batch['batch_size'], training=training)

            if training:
                self._n_train_examples += x_batch['batch_size']

            if (verbose > 0) and (step > 0) and (step % verbose == 0):
                summary_str = f"{self.self_report(step, mode)}\nused_time: {time.time() - start_time:0.2f}s"
                self.logger.log(summary_str, self.logger.metrics_log, echo=False)
                self.logger.log(summary_str, echo=True)

            if mode == 'test' and out_predictions:
                output.extend(res['predictions'])
                gold.extend(x_batch['target_src'])
        return output, gold

    def _set_forcing_ratio(self, step):
        return config.FORCING_RATIO * (config.FORCING_DECAY ** step)

    def _update_metrics(self, loss, metrics, batch_size, training=True):
        if training:
            if loss:
                self._train_loss.update(loss)
            for k in self._train_metrics:
                if not k in metrics:
                    continue
                self._train_metrics[k].update(metrics[k] * 100, batch_size)
        else:
            if loss:
                self._dev_loss.update(loss)
            for k in self._dev_metrics:
                if not k in metrics:
                    continue
                self._dev_metrics[k].update(metrics[k] * 100, batch_size)

    def self_report(self, step, mode="train"):
        if mode == "train":
            format_str = f"[train-{self._epoch}] step: [{step} / {self._n_train_batches}] | loss = {self._train_loss.mean():0.5f}"
            format_str += self.metric_to_str(self._train_metrics)
        elif mode == "dev":
            format_str = f"[predict-{self._epoch}] step: [{step} / {self._n_dev_batches}] | loss = {self._dev_loss.mean():0.5f}"
            format_str += self.metric_to_str(self._dev_metrics)
        elif mode == "test":
            format_str = f"[test] | test_exs = {self._n_test_examples} | step: [{step} / {self._n_test_batches}]"
            format_str += self.metric_to_str(self._dev_metrics)
        else:
            raise ValueError(f"mode = {mode} not supported.")
        return format_str
