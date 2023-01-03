import torch
import bootstrap.lib.utils as utils
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from bootstrap.models.networks.data_parallel import DataParallel

from .causal_transformer import CEVAE, Encoder, Decoder


def factory(engine=None):
    Logger()('Creating CEVAE network...')

    if Options()['model']['network']['name'] == 'cevae':
        encoder = Encoder(in_size=Options()['model']['network']['in_size'],
                          in2_size=Options()['model']['network']['in2_size'], 
                          d=Options()['model']['network']['d'],
                          nh=Options()['model']['network']['nh'],
                          h=Options()['model']['network']['h'],
                          activation=torch.nn.functional.elu)
        decoder = Decoder(n_z=Options()['model']['network']['d'],
                          nh=Options()['model']['network']['nh'],
                          h=Options()['model']['network']['h'],
                          activation=torch.nn.functional.elu)
        network = CEVAE(encoder=encoder, decoder=decoder, n_z=Options()['model']['network']['d'])

    else:
        raise ValueError()
    return network

