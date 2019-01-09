from torch.nn import CrossEntropyLoss
import config.args as args


def loss_fn(logits, labels):
    loss_f = CrossEntropyLoss()
    loss = loss_f(logits.view(-1, len(args.labels)), labels.view(-1))
    return loss