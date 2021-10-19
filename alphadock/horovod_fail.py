import torch
import torch.optim as optim
from torch import nn
import sys
import horovod.torch as hvd


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10, bias=False) for _ in range(2000)])

    def forward(self, x):
        return x


if __name__ == '__main__':
    hvd.init()
    HOROVOD_RANK = hvd.rank()
    torch.set_num_threads(1)

    print(HOROVOD_RANK, ': 1')
    sys.stdout.flush()
    model = Toy().to('cuda:0')

    print(HOROVOD_RANK, ': 2')
    print('Num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Num param sets:', len([p for p in model.parameters() if p.requires_grad]))
    for x in range(torch.cuda.device_count()):
        print('cuda:' + str(x), ':', torch.cuda.memory_stats(x)['allocated_bytes.all.peak'] / 1024**2)
    sys.stdout.flush()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    print(HOROVOD_RANK, ': 3')
    sys.stdout.flush()

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    print(HOROVOD_RANK, ': 4')
    sys.stdout.flush()

    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    print(HOROVOD_RANK, ': 5')
    sys.stdout.flush()

    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), op=hvd.Average)
    print(HOROVOD_RANK, ': 6')
    sys.stdout.flush()
