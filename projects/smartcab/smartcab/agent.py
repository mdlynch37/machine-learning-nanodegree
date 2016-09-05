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
        self.OKAY_MOVES = ('all', 'all but left', 'right on red', 'none')

        # as per GLIE method(greedy limit and infinite exploration)
        self.epsilon = 0.8  # starting expolation probability
        self.decay_rate = self.epsilon / 1300

        self.learning_rate = 0.2  # alpha, incremental update of Q estimate
        self.discount = 0.2  # gamma for value iteration for Q estimate update

        # [debug]
        self.num_updates = 0
        self.sample_states = []
        self.log = []

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

    # def update_state(self):
    #     self.state = self.State(
    #         location=self.env.agent_states[self]['location'],
    #         next_waypoint=self.next_waypoint,
    #         headings_legality=self.get_headings_legality())

    # def best_action(self):
    #     actions = self.q_table[self.state]
    #     if len(actions) is not 0:
    #         return max(actions.iterkeys(), key=(lambda key: actions[key]))
    #     else:
    #         # return random.choice(Environment.valid_actions)
    #         return self.next_waypoint

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

        # [debug]
        self.num_updates += 1
        # select states to sample at sampled updates
        if 1 < self.num_updates < 5:
            self.sample_states.append(self.state)
            # print self.sample_states
            # print len(self.sample_states)
        elif self.num_updates % 5 == 0:
            sample_values = []
            for state in self.sample_states:
                sample_values.append(self.q_table[state])
            sample_q_table = {state: actions for state, actions
                              in zip(self.sample_states, sample_values)}
            self.log.append([self.num_updates, len(self.q_table),
                             deepcopy(sample_q_table)])

        # Gather inputs
        self.inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)  # only for display purposes?
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
        # See above after state update for accurate s' at next step

        # print ("LearningAgent.update(): deadline = {}, inputs = {}, action = "
        #       "{}, reward = {}".format(deadline, self.inputs,
        #                                action, reward))  # [debug]


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


    # [debug]
    # # create simulator (uses pygame when display=True, if available)
    # sim = Simulator(e, update_delay=1.,
    #                 display=True)
    # # NOTE: To speed up simulation, reduce update_delay and/or set
    # # display=False

    print a.epsilon

    # num_states = []
    # for i in xrange(25):
    #     e = Environment()
    #     a = e.create_agent(LearningAgent)
    #     e.set_primary_agent(a, enforce_deadline=True)
    #     sim = Simulator(e, update_delay=0,
    #                     display=False)
    #     sim.run(n_trials=100)
    #     num_states.append(len(a.q_table))
    #
    # print sum(num_states)*1. / len(num_states)


    # log[0:timestep][n_updates, num states, {sample_q_table}]

    # for x in a.log:
    #     print x[0], ': ', x[1], x[3]  # sample :  num states, recorded n_random
    # print len(a.q_table)
    # #print a.cnt
    # print
    # print 'num sample snapshots:', len(a.log)
    #
    # print
    # print '****'
    # print
    # for i in xrange(0,len(a.sample_states)):
    #     print "*******SAMPLE ", i
    #     print a.sample_states[i]
    #     for action in e.valid_actions:
    #         print
    #         print '* ', action
    #         prev_value = 0.
    #         for sample in a.log:
    #             sample_q_table = sample[2]
    #             state = a.sample_states[i]
    #             current_value = sample_q_table[state][action]
    #             if current_value != prev_value:
    #                 print sample_q_table[state][action]  #, state, '-->>', sample_q_table[state]
    #
    #             prev_value = sample_q_table[state][action]
    #
    #
    # for i in xrange(0, len(a.sample_states)):
    #     print
    #     print "*******SAMPLE ", i
    #     print a.sample_states[i]
    #     for action in e.valid_actions:
    #         print action, a.log[-1][2][a.sample_states[i]][action]

    # print len(a.q_table)
    # for state in a.q_table.keys():
    #     print state
    # pass

if __name__ == '__main__':
    run()
