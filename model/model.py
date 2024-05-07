import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from modules import beam_search, Graph2SeqModule
from modules.vocab import load_or_init
from utils import Dataset, eval_batch_output, eval_decode_batch, evaluate_predictions, Logger, send_to_device

class Model(object):
    def __init__(self, train_data: Dataset, device: torch.device, logger: Logger) -> None:
        self.logger = logger
        self.device = device
        self.module = Graph2SeqModule
        self.logger.log(f"Running {self.module.__name__}")
        self.vocab_model = load_or_init(train_data, logger)
        self.vocab_model.node_id_vocab = None
        self.vocab_model.node_type_id_vocab = None
        self.vocab_model.edge_type_id_vocab = None

        if config.PRETRAINED:
            state_dict_opt = self.init_saved_network(config.PRETRAINED)
        else:
            assert train_data is not None
            self._init_network(self.device)

        num_params = 0
        for name, p in self.network.named_parameters():
            self.logger.log(f"{name}: {p.size()}")
            num_params += p.numel()
        self.logger.log(f"#Parameters = {num_params}")

        self.criterion = nn.NLLLoss(ignore_index=self.vocab_model.word_vocab.pad_ind)
        self._init_optimizer()
        self.wmd = None

    def _init_network(self, device: torch.device):
        word_embedding = self._init_embedding(
            self.vocab_model.word_vocab.vocab_size,
            self.vocab_model.word_vocab.embed_dim,
            self.vocab_model.word_vocab.embeddings,
        )
        self.logger.log(f"Instantiating NN module {self.module.__name__}...")
        self.network = self.module(
            word_embedding,
            self.vocab_model.word_vocab,
            self.logger,
            device,
        )

    def init_saved_network(self, saved_dir):
        fname = os.path.join(saved_dir, config.SAVED_WEIGHTS_FILE)
        self.logger.log(f"Loading saved model: {fname}")

        saved_params = torch.load(str(fname), map_location=lambda storage, loc: storage)
        state_dict = saved_params['state_dict']
        self.saved_epoch = saved_params.get('epoch', 0)

        word_embedding = self._init_embedding(
            self.vocab_model.word_vocab.vocab_size,
            config.WORD_EMBED_DIM,
        )
        self.network = self.module(word_embedding, self.vocab_model.word_vocab, self.logger, self.device)

        if state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

        return state_dict.get('optimizer', None) if state_dict else None

    def save(self, dirname, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
            'dir': dirname,
            'epoch': epoch
        }
        try:
            torch.save(params, str(os.path.join(dirname, config.SAVED_WEIGHTS_FILE)))
        except BaseException:
            self.logger.log(f"Tried to save model to {dirname}/{config.SAVED_WEIGHTS_FILE}, but failed!")

    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=config.LEARNING_RATE)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            verbose=True,
        )

    def _init_embedding(self, vocab_size: int, embedding_size: int, embeddings: np.ndarray = None):
        self.logger.log(f"Initializing embedding for {vocab_size} words of size {embedding_size}...")
        return nn.Embedding(
            vocab_size,
            embedding_size,
            padding_idx=self.vocab_model.word_vocab.pad_ind, # the embedding vector at this index doesn't get updated
            _weight=torch.from_numpy(embeddings).float() if embeddings is not None else None,
        )

    def predict(self,
        batch, step, forcing_ratio=1, rl_ratio=0, update=True, out_predictions=False, mode="train",
    ):
        self.network.train(update)
        decoded_batch = loss_value = metrics = None

        if mode == "train":
            loss, loss_value, metrics = self.train_batch(
                batch, self.criterion, forcing_ratio, rl_ratio, wmd=self.wmd,
            )
            loss.backward()

            # clip gradients
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, config.GRAD_CLIP)

            self.optimizer.step()
            self.optimizer.zero_grad()

        elif mode == "dev":
            decoded_batch, loss_value, metrics = self.dev_batch(
                batch,
                criterion=None,
                show_cover_loss=True,
            )

        elif mode == "test":
            decoded_batch, metrics = self.test_batch(batch)
            loss_value = None

        output = {
            "loss": loss_value,
            "metrics": metrics,
        }

        if mode == "test" and out_predictions:
            output["predictions"] = decoded_batch
        return output

    def dev_batch(self, batch, criterion=None, show_cover_loss=False):
        """Test the `network` on `batch`, return the ROUGE score and the loss."""
        network = self.network
        vocab = self.vocab_model.word_vocab

        network.train(False)
        decoded_batch, out = eval_decode_batch(
            batch,
            network,
            vocab,
            criterion=criterion,
            show_cover_loss=show_cover_loss,
        )
        metrics = evaluate_predictions(batch["target_src"], decoded_batch)
        return decoded_batch, out.loss_value, metrics

    def test_batch(self, batch):
        network = self.network
        vocab = self.vocab_model.word_vocab
        network.train(False)
        decoded_batch = beam_search(batch, network, vocab)
        metrics = evaluate_predictions(batch["target_src"], decoded_batch)
        return decoded_batch, metrics

    def train_batch(self, batch, criterion, forcing_ratio, rl_ratio, wmd=None):
        network = self.network
        vocab = self.vocab_model.word_vocab
        network.train(True)
        with torch.set_grad_enabled(True):
            ext_vocab_size = batch["oov_dict"].ext_vocab_size if batch["oov_dict"] else None
            network_out = network(
                batch,
                batch["targets"],
                criterion,
                forcing_ratio=forcing_ratio,
                partial_forcing=True,
                sample=False,
                ext_vocab_size=ext_vocab_size,
                include_cover_loss=True,
            )

            if rl_ratio > 0:
                batch_size = batch["batch_size"]
                sample_out = network(
                    batch,
                    saved_out=network_out,
                    criterion=criterion,
                    criterion_reduction=False,
                    criterion_nll_only=True,
                    sample=True,
                    ext_vocab_size=ext_vocab_size,
                )
                baseline_out = network(
                    batch,
                    saved_out=network_out,
                    visualize=False,
                    ext_vocab_size=ext_vocab_size,
                )

                sample_out_decoded = sample_out.decoded_tokens.transpose(0, 1)
                baseline_out_decoded = baseline_out.decoded_tokens.transpose(0, 1)

                neg_reward = []
                rl_reward_metric_list = config.RL_REWARD_METRIC.split(',')
                rl_reward_metric_ratio = None

                for i in range(batch_size):
                    scores = eval_batch_output(
                        [batch["target_src"][i]],
                        vocab,
                        batch["oov_dict"],
                        [sample_out_decoded[i]],
                        [baseline_out_decoded[i]],
                    )
                    reward_ = 0
                    for index, rl_reward_metric in enumerate(rl_reward_metric_list):
                        greedy_score = scores[1][rl_reward_metric]
                        tmp_reward_ = (scores[0][rl_reward_metric] - greedy_score)
                        if rl_reward_metric_ratio is not None:
                            tmp_reward_ = tmp_reward_ * rl_reward_metric_ratio[index]
                        reward_ += tmp_reward_

                    neg_reward.append(reward_)

                neg_reward = send_to_device(torch.Tensor(neg_reward), network.device)

                rl_loss = torch.sum(neg_reward * sample_out.loss) / batch_size
                rl_loss_value = torch.sum(neg_reward * sample_out.loss_value).item() / batch_size
                loss = (1 - rl_ratio) * network_out.loss + rl_ratio * rl_loss
                loss_value = (1 - rl_ratio) * network_out.loss_value + rl_ratio * rl_loss_value
                metrics = eval_batch_output(
                    batch["target_src"],
                    vocab,
                    batch["oov_dict"],
                    baseline_out.decoded_tokens,
                )[0]

            else:
                loss = network_out.loss
                loss_value = network_out.loss_value
                metrics = eval_batch_output(
                    batch["target_src"],
                    vocab,
                    batch["oov_dict"],
                    network_out.decoded_tokens,
                )[0]
        return loss, loss_value, metrics
