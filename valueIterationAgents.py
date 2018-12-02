# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import copy

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in xrange(self.iterations):
            copy_values = copy.deepcopy(self.values)
            for state in self.mdp.getStates():
                if state == 'TERMINAL_STATE':
                    self.values[state] = 0
                else:
                    q_list = []
                    for action in mdp.getPossibleActions(state):
                        trans_states_probs = self.mdp.getTransitionStatesAndProbs(state, action)
                        q_value = 0
                        for t in trans_states_probs:
                            q_value += t[1] * (self.mdp.getReward(state, action, t[0]) + (self.discount * copy_values[t[0]]))
                        q_list.append(q_value)
                    self.values[state] = max(q_list)
        debug = 10

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        trans_states_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        q = 0
        for t in trans_states_probs:
            q += t[1]*(self.mdp.getReward(state, action, t[0])+(self.discount*self.values[t[0]]))
        return q


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if state == 'TERMINAL_STATE':
            return None
        else:
            q = [self.computeQValueFromValues(state, a) for a in self.mdp.getPossibleActions(state)]
            actions = [a for a in self.mdp.getPossibleActions(state)]
            i = q.index(max(q))
            return actions[i]
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
