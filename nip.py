import h5py
import torch
import os
import logging
from argparse import Namespace

import click

from neulat.models import Z2Nice, COUPLINGS
from neulat.action import Phi4Action, QuantumRotor
from neulat.observable import ObservableMeter, Action
from neulat.observable import AbsMagnetization, TopologicalCharge, get_impsamp_statistics
from neulat.importsamp import ImportanceSampler, estimate_reverse_ess

import model_nflows

def geometry(string):
    return [int(obj) for obj in string.split('x')]


@click.command()
@click.option('--output', type=click.Path(dir_okay=False, writable=True))
@click.option('--seed', type=int, default=0xDEADBEEF)
@click.option('--lat-shape', type=geometry, default='8x8', help='lattice dimensions TxL')
@click.option('--kappa', type=float, default=0.3, help='kappa of action')
@click.option('--lamb', type=float, default=0.022, help='lambda of action')
@click.option('--n-batch', type=int, default=100, help='num configs per each iteration')
@click.option('--n-iter', type=int, default=500, help='num iterations')
@click.option('--ncouplings', type=int, default=6, help='number of coupling layers')
@click.option('--coupling', type=click.Choice(list(COUPLINGS)), default='fc', help='coupling layer for nice')
@click.option('--global-scaling/--local-scaling', help='global scaling mode for z2nice')
@click.option('--cuda/--cpu', default=False, help='runs the code on GPU if available')
@click.option('--sampler', type=click.Choice(['z2nice']), default="z2nice")
@click.option('--action', type=click.Choice(['phi4', 'rotor']), default="phi4")
@click.option('--bias/--no-bias', help='turn on bias for breaking z2 symmetry')
@click.option('--mom-inertia', type=float, default=1.0, help='moment of inertia of quantum rotor')
@click.option('--checkpoint', type=click.Path(dir_okay=False), help='path to model checkpoint')
@click.option('--obs/--no-obs', default=False, help='compute full set of observables')
def main(**kwargs):
    args = Namespace(**kwargs)
    logging.basicConfig(level=logging.INFO)
    torch.set_printoptions(precision=10)
    logging.info(vars(args))
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    config_path = args.output
    batch_size = args.n_batch
    n_iter = args.n_iter
    # try:
    #     os.unlink(config_path)
    # except FileNotFoundError:
    #     pass
    if args.action == "phi4":
        action = Phi4Action(args.kappa, args.lamb)

        meter = ObservableMeter(
            {
                "S": Action(action),
                "|M|": AbsMagnetization()
            },
            stat_func=get_impsamp_statistics
        )
    elif args.action == "rotor":
        action = QuantumRotor(args.mom_inertia)

        meter = ObservableMeter(
            {
                "S": Action(action),
                "Q": TopologicalCharge()
            },
            stat_func=get_impsamp_statistics
        )
    else:
        raise TypeError("Action \"{}\" is not valid".format(args.action))

    lat_shape = args.lat_shape
    logging.info(
        f'\nlatshape:{args.lat_shape}\nkappa:{args.kappa}\nn_iter:{args.n_iter}\nn_batch:{args.n_batch}'
    )
    if args.sampler == "z2nice":
        # coupling_factory = COUPLINGS[args.coupling]
        # model = Z2Nice(
        #     lat_shape=args.lat_shape,
        #     coupling_factory=coupling_factory,
        #     ncouplings=args.ncouplings,
        #     global_scaling=args.global_scaling,
        #     bias=args.bias
        # ).to(device)
        model = model_nflows.make_nflows_nice(device)

    else:
        raise TypeError("Sampler \"{}\" is not valid".format(args.sampler))

    # load model if specified
    if args.checkpoint is not None:
        # model.load(args.checkpoint, device)
        weights = torch.load(args.checkpoint, device)
        model.load_state_dict(weights)
    else:
        logging.warning("No sampler checkpoint provided. Using randomly init sampler.")

    obs = estimate_reverse_ess(model, action, lat_shape, batch_size, n_iter)
    print(obs)
    if args.output:
        with h5py.File(config_path, 'a') as fd:
            for key, value in obs.items():
                fd[key] = value

    if args.obs:
        sampler = ImportanceSampler(model, meter, action, lat_shape, args.n_batch)
        full_obs = sampler.run(args.n_iter)
        logging.info(f'{full_obs}')
        with h5py.File(config_path, 'a') as fd:
            for key, value in full_obs.items():
                fd[key] = value

    logging.info(f'weight histories have been  saved at {config_path}.')
    logging.info(f"The effective sampling ratio ESS was {obs['ess']:.4f}.")


if __name__ == "__main__":
    main()
