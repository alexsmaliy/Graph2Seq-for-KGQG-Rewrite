import torch

def vectorize_input(batch, config, training=True, device=None):
    if not batch:
        return None

    batch_size = len(batch.out_seqs)

    in_graphs = {}
    for k, v in batch.in_graphs.items():
        if k in ['node2edge', 'edge2node', 'max_num_graph_nodes']:
            in_graphs[k] = v
        else:
            in_graphs[k] = torch.LongTensor(v).to(device) if device else torch.LongTensor(v)

    out_seqs = torch.LongTensor(batch.out_seqs)
    out_seq_lens = torch.LongTensor(batch.out_seq_lens)

    with torch.set_grad_enabled(training):
        return {
            "batch_size": batch_size,
            "in_graphs": in_graphs,
            "targets": out_seqs.to(device) if device else out_seqs,
            "target_lens": out_seq_lens.to(device) if device else out_seq_lens,
            "target_src": batch.out_seq_src,
            "oov_dict": batch.oov_dict,
        }

class DataStream(object):
    pass # TODO