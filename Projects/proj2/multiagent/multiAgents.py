# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Deduct points if ghost is close.
        # (a) If timers are off and the dist to ghost is 0, you're screwed. Deduct a LOT.
        # (b) If timers are off and the dist to ghost is not zero, it's still bad but you can deduct less compared to (a)
        # (c) If timers are on and the dist to ghost is 0, this is actually good! You can eat it. Add some points.
        # (d) If timers are on and the dist to ghost is not zero, try to run anyway. Deduct a few points, but less than (b).
        dist = 0
        for ghost_state in newGhostStates:
            if min(newScaredTimes) == 0:
                dist = util.manhattanDistance(
                    newPos, ghost_state.getPosition())
                if dist == 0:
                    ghost_distance = -10000
                else:
                    ghost_distance = - 5 / dist

            elif min(newScaredTimes) != 0:
                if dist == 0:
                    ghost_distance = 1
                else:
                    ghost_distance = - 1 / dist

        # Deduct points if a lot of food left
        food_left = -len(newFood.asList())

        # Add a few points if nearest food is far, add a lot of points if nearest food is close
        if food_left != 0:
            closest_food = min([util.manhattanDistance(newPos, foodPos)
                                for foodPos in newFood.asList()])
            closest_food = 1 / closest_food
        else:
            closest_food = 0

        return food_left + ghost_distance + closest_food


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def value(agent_index, game_state, depth):
            # if the state is a terminal state, return the state's utility and the stop action
            if (depth == self.depth) or (game_state.isWin()) or (game_state.isLose()):
                ans = self.evaluationFunction(game_state), Directions.STOP

            # if the agent is pacman --> maximize utility
            elif agent_index == 0:
                ans = max_value(agent_index, game_state, depth)

            # if the agent is a ghost --> minimize utility
            else:
                ans = min_value(agent_index, game_state, depth)

            return ans

        def max_value(agent_index, game_state, depth):

            v = float("-inf")
            maximizer_action = Directions.STOP

            for action in game_state.getLegalActions(agent_index):

                successor_state = game_state.generateSuccessor(
                    agent_index, action)

                new_agent, new_depth = getNewAgentandDepth(
                    game_state, agent_index, depth)

                new_state_score = value(
                    new_agent, successor_state, new_depth)[0]

                if new_state_score > v:
                    v = new_state_score
                    maximizer_action = action

            return v, maximizer_action

        def min_value(agent_index, game_state, depth):

            v = float("inf")
            minimizer_action = Directions.STOP

            for action in game_state.getLegalActions(agent_index):
                successor_state = game_state.generateSuccessor(
                    agent_index, action)

                new_agent, new_depth = getNewAgentandDepth(
                    game_state, agent_index, depth)

                new_state_score = value(
                    new_agent, successor_state, new_depth)[0]

                if new_state_score < v:
                    v = new_state_score
                    minimizer_action = action

            return v, minimizer_action

        def getNewAgentandDepth(game_state, curr_agent_index, curr_depth):
            # CHANGING DEPTH
            # From the spec:  A single search ply is considered to be one Pacman move and all the ghosts’ responses
            # Thus, increment the depth only after we look at the last ghost

            # CHANGING AGENT INDEX
            # If you are at the last ghost, the next agent is Pacman itself i.e. index 0
            # Otherwise, increment the agent index by 1 every time

            if curr_agent_index == game_state.getNumAgents() - 1:
                new_agent_index, new_depth = 0, curr_depth + 1
            else:
                new_agent_index, new_depth = curr_agent_index + 1, curr_depth

            return new_agent_index, new_depth

        return value(0, gameState, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def value(agent_index, game_state, depth, alpha, beta):
            # if the state is a terminal state, return the state's utility and the stop action
            if (depth == self.depth) or (game_state.isWin()) or (game_state.isLose()):
                ans = self.evaluationFunction(game_state), Directions.STOP

            # if the agent is pacman --> maximize utility
            elif agent_index == 0:
                ans = max_value(agent_index, game_state, depth, alpha, beta)

            # if the agent is a ghost --> minimize utility
            else:
                ans = min_value(agent_index, game_state, depth, alpha, beta)

            return ans

        def max_value(agent_index, game_state, depth, alpha, beta):

            v = float("-inf")
            maximizer_action = Directions.STOP

            for action in game_state.getLegalActions(agent_index):

                successor_state = game_state.generateSuccessor(
                    agent_index, action)

                new_agent, new_depth = getNewAgentandDepth(
                    game_state, agent_index, depth)

                new_state_score = value(
                    new_agent, successor_state, new_depth, alpha, beta)[0]

                if new_state_score > v:
                    v = new_state_score
                    maximizer_action = action

                if new_state_score > beta:
                    return new_state_score, action

                alpha = max(alpha, v)

            return v, maximizer_action

        def min_value(agent_index, game_state, depth, alpha, beta):

            v = float("inf")
            minimizer_action = Directions.STOP

            for action in game_state.getLegalActions(agent_index):
                successor_state = game_state.generateSuccessor(
                    agent_index, action)

                new_agent, new_depth = getNewAgentandDepth(
                    game_state, agent_index, depth)

                new_state_score = value(
                    new_agent, successor_state, new_depth, alpha, beta)[0]

                if new_state_score < v:
                    v = new_state_score
                    minimizer_action = action

                if new_state_score < alpha:
                    return new_state_score, action

                beta = min(beta, v)

            return v, minimizer_action

        def getNewAgentandDepth(game_state, curr_agent_index, curr_depth):
            # CHANGING DEPTH
            # From the spec:  A single search ply is considered to be one Pacman move and all the ghosts’ responses
            # Thus, increment the depth only after we look at the last ghost

            # CHANGING AGENT INDEX
            # If you are at the last ghost, the next agent is Pacman itself i.e. index 0
            # Otherwise, increment the agent index by 1 every time

            if curr_agent_index == game_state.getNumAgents() - 1:
                new_agent_index, new_depth = 0, curr_depth + 1
            else:
                new_agent_index, new_depth = curr_agent_index + 1, curr_depth

            return new_agent_index, new_depth

        return value(0, gameState, 0, float("-inf"), float("inf"))[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def value(agent_index, game_state, depth):
            # if the state is a terminal state, return the state's utility and the stop action
            if (depth == self.depth) or (game_state.isWin()) or (game_state.isLose()):
                ans = self.evaluationFunction(game_state), Directions.STOP

            # if the agent is pacman --> maximize utility
            elif agent_index == 0:
                ans = max_value(agent_index, game_state, depth)

            # if the agent is a ghost --> minimize utility
            else:
                ans = expected_value(agent_index, game_state, depth)

            return ans

        def max_value(agent_index, game_state, depth):

            v = float("-inf")
            maximizer_action = Directions.STOP

            for action in game_state.getLegalActions(agent_index):

                successor_state = game_state.generateSuccessor(
                    agent_index, action)

                new_agent, new_depth = getNewAgentandDepth(
                    game_state, agent_index, depth)

                new_state_score = value(
                    new_agent, successor_state, new_depth)[0]

                if new_state_score > v:
                    v = new_state_score
                    maximizer_action = action

            return v, maximizer_action

        def expected_value(agent_index, game_state, depth):

            v = 0

            for action in game_state.getLegalActions(agent_index):
                successor_state = game_state.generateSuccessor(
                    agent_index, action)

                new_agent, new_depth = getNewAgentandDepth(
                    game_state, agent_index, depth)

                new_state_score = value(
                    new_agent, successor_state, new_depth)[0]

                v = v + new_state_score

            return v/len(gameState.getLegalActions(agent_index)), Directions.STOP

        def getNewAgentandDepth(game_state, curr_agent_index, curr_depth):
            # CHANGING DEPTH
            # From the spec:  A single search ply is considered to be one Pacman move and all the ghosts’ responses
            # Thus, increment the depth only after we look at the last ghost

            # CHANGING AGENT INDEX
            # If you are at the last ghost, the next agent is Pacman itself i.e. index 0
            # Otherwise, increment the agent index by 1 every time

            if curr_agent_index == game_state.getNumAgents() - 1:
                new_agent_index, new_depth = 0, curr_depth + 1
            else:
                new_agent_index, new_depth = curr_agent_index + 1, curr_depth

            return new_agent_index, new_depth

        return value(0, gameState, 0)[1]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    My evaluation function primarily keeps track of 4 features. 

    1)  Distance to ghosts
    Rationale:
    (a) If timers are off and the dist to ghost is 0, you're screwed. Deduct a LOT.
    (b) If timers are off and the dist to ghost is not zero, it's still bad but you can deduct less compared to (a)
    (c) If timers are on and the dist to ghost is 0, this is actually good! You can eat it. Add some points.
    (d) If timers are on and the dist to ghost is not zero, try to run anyway. Deduct a few points, but less than (b).

    2) Amount of food left
    Rationale: Deduct points if a lot of food left

    3) Distance to food
    Rationale: Add a few points if nearest food is far, add a lot of points if nearest food is close

    4) Current score
    Rationale: Add a few points if current score is less, add a lot of points if current score is a lot. This penalizes staying in place.

    """
    "*** YOUR CODE HERE ***"

    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    "*** YOUR CODE HERE ***"
    # distance to ghost
    dist = 0
    for ghost_state in ghostStates:
        if min(scaredTimes) == 0:
            dist = util.manhattanDistance(
                position, ghost_state.getPosition())
            if dist == 0:
                ghost_distance = -10000
            else:
                ghost_distance = - 5 / dist

        elif min(scaredTimes) != 0:
            if dist == 0:
                ghost_distance = 1
            else:
                ghost_distance = - 1 / dist

    # amount of food
    food_left = -len(food.asList())

    # distance to food
    if food_left != 0:
        closest_food = min([util.manhattanDistance(position, foodPos)
                            for foodPos in food.asList()])
        closest_food = 0.7 / closest_food
    else:
        closest_food = 0

    # current score
    curr_score = currentGameState.getScore()

    return food_left + ghost_distance + closest_food + curr_score


# Abbreviation
better = betterEvaluationFunction
