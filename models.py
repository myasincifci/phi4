import math
import torch
from torch import nn
from torch.nn import functional as F
from nflows.transforms.coupling import AdditiveCouplingTransform, AffineCouplingTransform
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.normalization import BatchNorm

class MLP(nn.Module):
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
    
class ACTwithScaling(AdditiveCouplingTransform):
    """ Version of AdditiveCouplingTransform with added 
    scaling component.
    """
    
    def __init__(self, mask, transform_net_create_fn, unconditional_transform=None, scale_activation=...):
        super().__init__(mask, transform_net_create_fn, 
                         unconditional_transform=unconditional_transform,
                         scale_activation=scale_activation
                        )

        self.scale = nn.Parameter(
            torch.zeros((1, torch.tensor(list(mask.shape)).prod()//2)),
            requires_grad=True
        )

    def _scale_and_shift(self, transform_params):
        shift = transform_params
        scale = torch.exp(self.scale)
        return scale, shift

class Z2Nice(Flow):
    """
    """
    def __init__(
        self,
        lat_shape,
        hidden_features=1000,
        num_layers=4,  
        activation=F.relu,    
        device="cpu"
    ):
        coupling_constructor = ACTwithScaling

        def create_mlp(in_features, out_features):
            return nets.MLP(
                [in_features],
                [out_features],
                [hidden_features],
                activation=activation
            )
        
        layers = []
        for i in range(num_layers):
            mask = self._make_checker_mask(lat_shape, i % 2)

            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=MLP
            )
            layers.append(transform)

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([math.prod(lat_shape)]).to(device),
        )
    
    def _make_checker_mask(self, shape, parity):
        checker = torch.ones(shape, dtype=torch.uint8) - parity
        checker[::2, ::2] = parity
        checker[1::2, 1::2] = parity

        return checker.flatten()

class SimpleRealNVP(Flow):
    """An simplified version of Real NVP for 1-dim inputs.

    This implementation uses 1-dim checkerboard masking but doesn't use multi-scaling.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(
        self,
        lat_shape,
        num_layers=20,
        num_blocks_per_layer=4,
        hidden_features=None,
        use_volume_preserving=True,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
        device="cpu"
    ):

        if use_volume_preserving:
            # coupling_constructor = AdditiveCouplingTransform
            coupling_constructor = ACTwithScaling
            # coupling_constructor = splines.linear_spline
        else:
            coupling_constructor = AffineCouplingTransform

        mask = torch.ones(math.prod(lat_shape))
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=math.prod(lat_shape)))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([math.prod(lat_shape)]).to(device),
        )

model_dict = {
    "real_nvp": SimpleRealNVP,
    "z2nice": Z2Nice   
}