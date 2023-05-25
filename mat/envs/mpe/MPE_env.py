from .environment import MultiAgentEnv
from .scenarios import load


def MPEEnv(args):
    # load scenario from script
    scenario = load(args.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args)

    # create multiagent environment
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        scenario.info,
    )

    return env
