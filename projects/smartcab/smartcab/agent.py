import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple, defaultdict
from copy import deepcopy

# Note that 'right' input variable is not included since this would have no
# impact on the reward since, on its own, it has no impact on the legality on
# any action the learning agent might take.



class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    State = namedtuple('State', ['next_waypoint', 'light',
                                 'oncoming', 'left'])

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None,
        #  and a default color
        super(LearningAgent, self).__init__(env)
        self.color = 'red'

        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)

        # TODO: Initialize any additional variables here
        SEED = 42
        random.seed(SEED)

        self.epsilon = 0.5  # starting exploration probability
        self.decay_divisor = 400  # estimated total number of updates
        self.decay_rate = self.epsilon / self.decay_divisor

        self.learning_rate = 0.2  # alpha, incremental update of Q estimate
        self.discount = 0.2  # gamma for value iteration for Q estimate update

        self.INITIAL_Q_ESTIMATE = 0.0
        self.q_table = defaultdict(
                            lambda:defaultdict(lambda:self.INITIAL_Q_ESTIMATE))

        # 1-based index due to first elem being set to 0 at start of 1st trial
        self.num_moves = []
        self.num_moves_this_trial = 0
        self.num_violations = []
        self.num_violations_this_trial = 0
        self.missed_deadline = []


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.reward = None
        self.action = None
        self.state = None

        self.num_moves.append(self.num_moves_this_trial)
        self.num_violations.append(
            self.num_violations_this_trial)

        self.num_moves_this_trial = 0
        self.num_violations_this_trial = 0

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
        self.num_moves_this_trial += 1
        # Gather inputs
        self.inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)  # only for display purposes
        self.next_waypoint = self.planner.next_waypoint()

        # TODO: Update state as per value iteration method
        # Declares current state of agent, referenced temporarily as
        # state_prime, so that q_table can be updated with it and other
        # necessary variables carried over from previous update.
        # This should be clear from the code here and in update_q_table().
        state_prime = LearningAgent.State(next_waypoint=self.next_waypoint,
                                          light=self.inputs['light'],
                                          oncoming=self.inputs['oncoming'],
                                          left=self.inputs['left'])
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

        if self.reward == -1.0:
            self.num_violations_this_trial += 1

        # TODO: Learn policy based on state, action, reward
        # See LearningAgent.update_q_table()


# Udacity reviewer can ignore this, requires modfied smartcab dependencies
# Needs testing along with simpler performance metrics.
def grid_search(trials=50, epsilons=(.4, .5, .6),
                           decay_divisors=None,
                           learning_rates=(.2, .4, .6),
                           discounts=(.2, .4, .6),
                           only_Q4=True):
    import pandas as pd

    if trials < 4:
        return "Min. number of trials is 4!"

    if decay_divisors is None:
        decay_divisors = (10*trials, 15*trials, 20*trials)



    results = []

    for epsilon in epsilons:
        for decay_divisor in decay_divisors:
            for learning_rate in learning_rates:
                for discount in discounts:
                    # Set up environment and agent
                    e = Environment()  # create environment (also adds some
                    # dummy traffic)
                    a = e.create_agent(LearningAgent)  # create agent
                    a.epsilon = epsilon
                    a.estimated_total_num_updates = decay_divisor
                    a.learning_rate = learning_rate
                    a.discount = discount

                    e.set_primary_agent(a,
                                        enforce_deadline=True)  # specify
                    # agent to track
                    # NOTE: You can set enforce_deadline=False while
                    # debugging to allow
                    # longer trials

                    # create simulator (uses pygame when display=True,
                    # if available)
                    sim = Simulator(e, update_delay=0,
                                    display=False)
                    # NOTE: To speed up simulation, reduce update_delay
                    # and/or set
                    # display=False

                    sim.run(
                        n_trials=trials)  # run for a specified number of
                    # trials
                    # NOTE: To quit midway, press Esc or close pygame window,
                    # or hit Ctrl+C on the command-line


                    # For each combination of params, determine results
                    # based on averages of various performance metrics across
                    # each quartile of trials conducted.
                    result = [epsilon, decay_divisor, learning_rate, discount]
                    num_moves_avgs, num_violations_avgs = [], []
                    num_missed_deadline_avgs = []

                    for i in xrange(4):
                        lo = i * (trials / 4)
                        hi = (i+1) * (trials / 4)
                        q_m = a.num_moves[lo:hi]
                        q_v = a.num_violations[lo:hi]
                        q_d = a.missed_deadline[lo:hi]
                        num_moves_avgs.append(1.*sum(q_m)/len(q_m))
                        num_violations_avgs.append(1.* sum(q_v) / len(q_v))
                        num_missed_deadline_avgs.append(
                            1. * sum(q_d) / len(q_d))

                    result += num_missed_deadline_avgs
                    result += num_violations_avgs
                    result += num_moves_avgs

                    results.append(result)

    data = pd.DataFrame(results, columns=['epsilon', 'decay_divisor',
                                    'learning_rate', 'discount',
                                    'avg_missed__Q1', 'avg_missed_Q2',
                                    'avg_missed_Q3', 'avg_missed_Q4',
                                    'avg_violations_Q1', 'avg_violations_Q2',
                                    'avg_violations_Q3', 'avg_violations_Q4',
                                    'avg_moves_Q1', 'avg_moves_Q2',
                                    'avg_moves_Q3', 'avg_moves_Q4',])

    if only_Q4:
        columns = set(data.columns[4:])
        columns -= {'avg_missed_Q4', 'avg_violations_Q4', 'avg_moves_Q4'}
        data.drop(list(columns), axis=1, inplace=True)

    return data


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
