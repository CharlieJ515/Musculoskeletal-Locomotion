from torch.utils.tensorboard import SummaryWriter

_writer = SummaryWriter()


def get_writer() -> SummaryWriter:
    return _writer
