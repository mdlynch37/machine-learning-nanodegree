import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None,
        #  and a default color
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color

        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.next_waypoint = random.choice(Environment.valid_actions)

    def update(self, t):
        # Gather inputs
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        # from route planner, also displayed by simulator
        # self.next_waypoint = self.planner.next_waypoint()

        # TODO: Update state


        # TODO: Select action according to your policy
        # Execute action and get reward
        action = self.next_waypoint
        self.next_waypoint = random.choice(Environment.valid_actions)
        reward = self.env.act(self, action)


        # TODO: Learn policy based on state, action, reward

        # print ("LearningAgent.update(): deadline = {}, inputs = {}, action = "
        #       "{}, reward = {}".format(deadline, inputs,
        #                                action, reward))  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow
    # longer trials

    # create simulator (uses pygame when display=True, if available)
    sim = Simulator(e, update_delay=0.5,
                    display=True)
    # NOTE: To speed up simulation, reduce update_delay and/or set
    # display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C
    # on the command-line


if __name__ == '__main__':
    run()
