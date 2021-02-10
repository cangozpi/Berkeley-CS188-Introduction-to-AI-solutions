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


from pacman import GameState
from util import manhattanDistance
from game import Directions
import random, util

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
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # # manhattan distance to foods
        manhattanScoretoFood = 0.000000001
        for x,y in newFood.asList():
            manhattanScoretoFood += manhattanDistance((x,y), newPos)
        
        #if win return 
        if successorGameState.isWin():
            return 99999999 #placeholder for inf
        
        if action == "Stop":#prevents from stopping
            return -999999

        #manhattan distance to closes food
        closestFoodCoordinate = (float("inf"), float("inf"))
        for xy in newFood.asList():
            closestFoodCoordinate = xy if manhattanDistance(xy, newPos) <= manhattanDistance(newPos, closestFoodCoordinate) else closestFoodCoordinate
        manhattanDistanceToClosestFood = manhattanDistance(closestFoodCoordinate, newPos)
        
        
        #minimize food count
        foodCount = 0
        if len(currentGameState.getFood().asList()) > len(newFood.asList()):
            foodCount = 1000 #compensates for manhattanDistanceToClosestFood increasing after eating a food

        #distance to ghosts
        ghostDistance = 0
        for i in range(len(newGhostStates)):
            ghostPos = newGhostStates[i].getPosition() 
            distance = manhattanDistance(ghostPos, newPos)
            if distance <= 2:
                ghostDistance += -99999999 * 2**-distance #placeholder for negative infinity 
                
                minX, maxX = ghostPos[0],newPos[0] 
                
                if newPos[0] < ghostPos[0]:
                    minX,maxX = newPos[0], ghostPos[0]

                minY, maxY = ghostPos[1],newPos[1] 
                if newPos[1] < ghostPos[1]:
                    minY,maxY = newPos[1], ghostPos[1]
                
                xFlag,yFlag = False, False
                if maxY == minY:
                    for x in range(int(minX), int(maxX)):
                        if not successorGameState.hasWall(int(x), int(maxY)):
                            yFlag = True
                            
                if maxX == minX:
                    for y in range(int(minY), int(maxY)):
                        if successorGameState.hasWall(int(minX), int(y)):
                            xFlag = True

                if xFlag:
                    ghostDistance += -99999999 #further increase if there is no wall between them in same x position
                if yFlag:
                    ghostDistance += -99999999 #further increase if there is no wall between them in same x position

        #print(successorGameState.getWalls())
        # game score
        return -manhattanDistanceToClosestFood  + foodCount + ghostDistance

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        def minimax(state, depth, agentNum):
            if agentNum == state.getNumAgents():
                agentNum = 0
                depth -= 1

            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            else:
                if agentNum == 0: #maximizing Agents turn
                    return maxAgent(state,depth, agentNum)
                else: #minimizing Agents turn
                    return minAgent(state,depth, agentNum)

        
        def maxAgent(state, depth, agentNum):
            maxEvaluation = -float("inf")
            for action in state.getLegalActions(agentNum): 
                successorState = state.generateSuccessor(agentNum, action)
                tempEvaluation = minimax(successorState, depth, agentNum + 1) # take max value of ghost agents choices
                maxEvaluation = max(maxEvaluation, tempEvaluation)
            return maxEvaluation

        def minAgent(state, depth, agentNum):
            minEvaluation = float("inf")
            for action in state.getLegalActions(agentNum):
                successorState = state.generateSuccessor(agentNum, action) 
                tempEvaluation = minimax(successorState, depth, agentNum + 1) #take min value of pacman agents choices
                minEvaluation  = min(minEvaluation, tempEvaluation)
            return minEvaluation

        #start the minimax algrotihm here
        maxValue = -float("inf")
        maxAction = None
        for action in gameState.getLegalActions(0):#first game state
            successorState = gameState.generateSuccessor(0, action)
            if maxAction == None:#initialize maxAction
                maxAction = action
            tempValue = minimax(successorState, self.depth, 1)
            if maxValue < tempValue:
                maxValue = tempValue
                maxAction = action
        return maxAction

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        
        def minimax(state, depth, agentNum, alpha, beta):
            if agentNum == state.getNumAgents():
                agentNum = 0
                depth -= 1

            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            else:
                if agentNum == 0: #maximizing Agents turn
                    return maxAgent(state,depth, agentNum, alpha, beta)
                else: #minimizing Agents turn
                    return minAgent(state,depth, agentNum, alpha, beta)

        
        def maxAgent(state, depth, agentNum, alpha, beta):
            maxEvaluation = -float("inf")
            for action in state.getLegalActions(agentNum): 
                successorState = state.generateSuccessor(agentNum, action)
                tempEvaluation = minimax(successorState, depth, agentNum + 1, alpha, beta) # take max value of ghost agents choices
                maxEvaluation = max(maxEvaluation, tempEvaluation)
                if beta < maxEvaluation:
                    return maxEvaluation
                alpha = max(alpha, maxEvaluation)
            return maxEvaluation

        def minAgent(state, depth, agentNum, alpha, beta):
            minEvaluation = float("inf")
            for action in state.getLegalActions(agentNum):
                successorState = state.generateSuccessor(agentNum, action) 
                tempEvaluation = minimax(successorState, depth, agentNum + 1, alpha, beta) #take min value of pacman agents choices
                minEvaluation  = min(minEvaluation, tempEvaluation)
                if minEvaluation < alpha:
                    return minEvaluation
                beta = min(beta, minEvaluation)
            return minEvaluation

        #start the alpha-beta pruning algrotihm here
        maxValue = -float("inf")
        maxAction = None
        alpha = -float("inf")
        beta = float("inf")
        for action in gameState.getLegalActions(0):#first game state
            successorState = gameState.generateSuccessor(0, action)
            if maxAction == None:#initialize maxAction
                maxAction = action
            tempValue = minimax(successorState, self.depth, 1, alpha, beta)
            if maxValue < tempValue:
                maxValue = tempValue
                maxAction = action
            if beta < maxValue:
                    return maxAction
            alpha = max(alpha, maxValue)
        return maxAction

        
        #util.raiseNotDefined()

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
       
        def minimax(state, depth, agentNum):
            if agentNum == state.getNumAgents():
                agentNum = 0
                depth -= 1

            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            else:
                if agentNum == 0: #maximizing Agents turn
                    return maxAgent(state,depth, agentNum)
                else: #minimizing Agents turn
                    return minAgent(state,depth, agentNum)

        
        def maxAgent(state, depth, agentNum):
            maxEvaluation = -float("inf")
            for action in state.getLegalActions(agentNum): 
                successorState = state.generateSuccessor(agentNum, action)
                tempEvaluation = minimax(successorState, depth, agentNum + 1) # take max value of ghost agents choices
                maxEvaluation = max(maxEvaluation, tempEvaluation)
            return maxEvaluation

        def minAgent(state, depth, agentNum):
            minEvaluation = float("inf")
            for action in state.getLegalActions(agentNum):
                successorState = state.generateSuccessor(agentNum, action) 
                tempEvaluation = minimax(successorState, depth, agentNum + 1) #take weighted average of scores of the successors
                if minEvaluation == float("inf"):
                    minEvaluation = 0
                minEvaluation  += tempEvaluation
            return minEvaluation / len(state.getLegalActions(agentNum)) #average since probablities are assumed to be equal

        #start the expectimax algrotihm here
        maxValue = -float("inf")
        maxAction = None
        for action in gameState.getLegalActions(0):#first game state
            successorState = gameState.generateSuccessor(0, action)
            if maxAction == None:#initialize maxAction
                maxAction = action
            tempValue = minimax(successorState, self.depth, 1)
            if maxValue < tempValue:
                maxValue = tempValue
                maxAction = action
        return maxAction

        
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Useful information you can extract from a GameState (pacman.py)
    pacmanPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()
    numberOfFood = currentGameState.getNumFood()
    ghostStates = currentGameState.getGhostStates()
    numOfAgents = currentGameState.getNumAgents()
    ghostPositions = currentGameState.getGhostPositions()
    scaredGhostTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    gameScore = GameState.getScore(currentGameState)

    #check fundamental steps first like win and loose
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return -float("inf")

    #manhattan distance to closes food
    closestFoodCoordinate = (float("inf"), float("inf"))
    for xy in foodList:
        closestFoodCoordinate = xy if manhattanDistance(xy, pacmanPosition) <= manhattanDistance(pacmanPosition, closestFoodCoordinate) else closestFoodCoordinate
    manhattanDistanceToClosestFood = manhattanDistance(closestFoodCoordinate, pacmanPosition)


    #distance to ghosts
    ghostDistance = 0
    for i in range(len(ghostStates)):
        ghostPos = ghostStates[i].getPosition() 
        distance = manhattanDistance(ghostPos, pacmanPosition)
        if distance <= 2:
            ghostDistance += -99999999 * 2**-distance #placeholder for negative infinity 
                
            minX, maxX = ghostPos[0],pacmanPosition[0] 
                
            if pacmanPosition[0] < ghostPos[0]:
                minX,maxX = pacmanPosition[0], ghostPos[0]

            minY, maxY = ghostPos[1],pacmanPosition[1] 
            if pacmanPosition[1] < ghostPos[1]:
                minY,maxY = pacmanPosition[1], ghostPos[1]
                
                xFlag,yFlag = False, False
                if maxY == minY:
                    for x in range(int(minX), int(maxX)):
                        if not currentGameState.hasWall(int(x), int(maxY)):
                            yFlag = True
                            
                if maxX == minX:
                    for y in range(int(minY), int(maxY)):
                        if currentGameState.hasWall(int(minX), int(y)):
                            xFlag = True

                if xFlag:
                    ghostDistance += -99999999 #further increase if there is no wall between them in same x position
                if yFlag:
                    ghostDistance += -99999999 #further increase if there is no wall between them in same x position

    
    #gameScore - manhattanDistanceToClosestFood -> 4/6 points
    #gameScore - manhattanDistanceToclosestFood -> 4/6 points
    #gameScore - manhattanDistanceToClosestFood - numberOfFood + ghostDistance -> 5/6
    return gameScore - manhattanDistanceToClosestFood - numberOfFood + ghostDistance
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
