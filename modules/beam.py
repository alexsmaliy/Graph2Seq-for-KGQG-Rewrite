import pdb

import torch

import config
from utils import batch_decoded_index2word

def beam_search(batch, network, vocab):
    with torch.no_grad():
        ext_vocab_size = batch["oov_dict"].ext_vocab_size if batch["oov_dict"] else None

        hypotheses = batch_beam_search(
            network,
            batch,
            ext_vocab_size,
            config.BEAM_SIZE,
            min_out_len=config.MIN_OUT_LENGTH,
            max_out_len=config.MAX_OUT_LENGTH,
            len_in_words=False,
            block_ngram_repeat=config.BLOCK_NGRAM_REPEAT,
        )

        try:
            to_decode = [each[0].tokens[1:] for each in hypotheses] # the first token is SOS
        except:
            pdb.set_trace()

        decoded_batch = batch_decoded_index2word(to_decode, vocab, batch['oov_dict'])
        return decoded_batch
