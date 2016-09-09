import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple, defaultdict
from copy import deepcopy

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None,
        #  and a default color
        super(LearningAgent, self).__init__(env)
        # seems inconsequential, just for starting display
        #self.next_waypoint = random.choice(Environment.valid_actions[1:])
        self.color = 'red'

        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)

        # TODO: Initialize any additional variables here
        SEED = 42
        random.seed(SEED)

        self.State = namedtuple('State', ['next_waypoint', 'okay_moves'])

        self.epsilon = 0.5  # starting expolation probability
        self.decay_rate = self.epsilon / 400

        self.learning_rate = 0.2  # alpha, incremental update of Q estimate
        self.discount = 0.2  # gamma for value iteration for Q estimate update

        self.INITIAL_Q_ESTIMATE = 0.0
        self.q_table = defaultdict(
                            lambda:defaultdict(lambda:self.INITIAL_Q_ESTIMATE))

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.reward = None
        self.action = None
        self.state = None

    def determine_okay_moves(self):
        forward_okay = (False if self.inputs['light'] == 'red'
                        else True)
        right_okay = (False if self.inputs['light'] == 'red'
                            and self.inputs['left'] == 'forward'
                      else True)
        left_okay = (False if self.inputs['light'] == 'red' or
                              (self.inputs['oncoming'] == 'forward' or
                               self.inputs['oncoming'] == 'right')
                     else True)

        if all([forward_okay, right_okay, left_okay]):
            okay_moves = 'all'
        elif all([forward_okay, right_okay]):
            okay_moves = 'all but left'
        elif right_okay:
            okay_moves = 'right on red'
        else:
            okay_moves = 'none'

        return okay_moves

    def update_q_table(self, s, a, r, s_prime):

        try:
            q_a_prime = max(self.q_table[s_prime].itervalues())
        except ValueError:  # if first time in state
            q_a_prime = self.INITIAL_Q_ESTIMATE

        try:
            q_estimate_update = r + self.discount * q_a_prime
        except TypeError:
            # if start of new trial where r == None, skip to next iteration
            return

        self.q_table[s][a] = ((1 - self.learning_rate) * self.q_table[s][a]
                               + q_estimate_update * self.learning_rate)

    def update(self, t):

        # Gather inputs
        self.inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)  # only for display purposes
        self.next_waypoint = self.planner.next_waypoint()

        # TODO: Update state as per value iteration method
        state_prime = self.State(next_waypoint=self.next_waypoint,
                                 okay_moves=self.determine_okay_moves())
        self.update_q_table(self.state, self.action, self.reward, state_prime)
        self.state = state_prime

        # TODO: Select action according to your policy
        # Execute action and get reward
        if random.random() < self.epsilon or not(self.q_table[self.state]):
            self.action = random.choice(Environment.valid_actions)
        else:
            actions = self.q_table[self.state]
            self.action = max(actions.iterkeys(),
                              key=(lambda key:actions[key]))
        self.epsilon -= self.decay_rate

        self.reward = self.env.act(self, self.action)

        # TODO: Learn policy based on state, action, reward
        # See LearningAgent.update_q_table()


def run():
    """Run the agent for a finite number of trials."""

        # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow
    # longer trials

    # create simulator (uses pygame when display=True, if available)
    sim = Simulator(e, update_delay=0,
                    display=False)
    # NOTE: To speed up simulation, reduce update_delay and/or set
    # display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C
    # on the command-line


if __name__ == '__main__':
    run()
