from torch import nn
import torch

from nflows.transforms.coupling import AdditiveCouplingTransform
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.flows.base import Flow


T, L = 16, 8
lattice_shape = (T,L)

class Net(nn.Module):
    def __init__(self, num_id, num_trf) -> None:
        super().__init__()
        mid_dim=1000
        nblocks=4

        self.net = nn.Sequential(
            nn.Linear(num_id, mid_dim, bias=False),
            nn.Tanh(),
            nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(mid_dim, mid_dim, bias=False),
                    nn.Tanh(),
                ) for _ in range(nblocks)
            ]),
            nn.Linear(mid_dim, num_trf, bias=False)
        )
    
    def forward(self, x, ctx):
        return self.net(x)

class Z2NICE(AdditiveCouplingTransform):
    def __init__(self, mask, transform_net_create_fn, unconditional_transform=None, scale_activation=...):
        super().__init__(mask, transform_net_create_fn, unconditional_transform, scale_activation)

        self.scale = nn.Parameter(
            torch.zeros((1, torch.tensor(list(lattice_shape)).prod()//2)),
            requires_grad=True
        )

    def _scale_and_shift(self, transform_params):
        shift = transform_params
        scale = torch.exp(self.scale)
        return scale, shift
    
def make_checker_mask(shape, parity):
    checker = torch.ones(shape, dtype=torch.uint8) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    return checker

def make_mask(lattice_shape, parity):
    """
    Input: B x T, flattened 8*8
    """
    mask = make_checker_mask(lattice_shape, parity)
    return mask.flatten()

def make_nflows_nice(device):
    num_layers = 6
    base_dist = StandardNormal(shape=[128]).to(device)

    transforms = []
    for i in range(num_layers):
        transforms.append(
            Z2NICE(
                make_mask(lattice_shape, i % 2), 
                Net
            ),
        )
    transform = CompositeTransform(transforms)

    flow_nflows = Flow(transform, base_dist).to(device)

    return flow_nflows