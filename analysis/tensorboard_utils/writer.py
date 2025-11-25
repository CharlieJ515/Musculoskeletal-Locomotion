from torch.utils.tensorboard import SummaryWriter

_writer = None


def get_writer() -> SummaryWriter:
    global _writer

    if _writer is None:
        _writer = SummaryWriter()
    return _writer
