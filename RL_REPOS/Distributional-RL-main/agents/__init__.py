from .iqn import IQN
from .c51 import C51
from .qrdqn import QRDQN
from .fqf import FQF

AGENTS = dict(IQN=IQN,
              C51=C51,
              QRDQN=QRDQN,
              FQF=FQF
              )


def get_agent_configs(**configs):
    agent_configs = AGENTS[configs["agent_name"]].get_configs()
    return {**configs, **agent_configs}


def get_agent(cfg_agent):
    name = cfg_agent.agent.name.upper()
    return AGENTS[name](cfg_agent)
