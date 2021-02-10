# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    #variables required for the DFS
    frontier = util.Stack()
    exploredSet = set() #this is important since we're implementing using a graph not a tree

    frontier.push((problem.getStartState(), ())) # insert first node of the graph in the format node = (state, actions of type tuple)
    #iterate over the graph as long as frontier is not empty
    while not frontier.isEmpty():
        currentState = frontier.pop()
        #check if node exists in the explored set
        if not exploredSet.__contains__(currentState[0]):
            exploredSet.add(currentState[0])
            #check if the popped goal is the goal state
            if problem.isGoalState(currentState[0]):
                return list(currentState[1]) # return the actions when goal is found
            else:
                #iterate over the successors and add to frontier to be expanded in the future
                for successor in problem.getSuccessors(currentState[0]):
                    successorNode = (successor[0], currentState[1] + (successor[1],))
                    frontier.push(successorNode)
        else:
            continue

    #if no solution is found return None to indicate that no particular solution exists and print an error message
    print("Oops, something went wrong could not found a solution path using DFS function !")
    return None

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    #variables
    frontier = util.Queue() # use queue instead of stack to implement BFS
    exploredSet = set()#needed for graph traversal / holds coordinates as set elements
    
    #initialize the frontier with the starting state
    frontier.push((problem.getStartState(), ())) # tuple of (coordinate, actions) / coordiante e.g (5,4) and actions e.g('South', 'North')
    
    while not frontier.isEmpty():
        currentState = frontier.pop()
        if not exploredSet.__contains__(currentState[0]):
            #check for the goal state
            if problem.isGoalState(currentState[0]):
                return list(currentState[1]) #return actions
            else: #if currentState is not the goal state
                exploredSet.add(currentState[0])

                #add successors of the node to frontier
                for successor in problem.getSuccessors(currentState[0]):
                    frontier.push((successor[0], (currentState[1] + (successor[1] , ))))
        else:
            continue

    #return None and print error , message to indicate error
    print("Oops, something went wrong. No particular solution was found using BFS!")
    return None

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    #variables
    frontier = util.PriorityQueue()
    exploredSet = set() # required for graph implementation

     #initialize the frontier
    frontier.push((problem.getStartState(), () ), 0) # in the format of ((state, actions), priority)
    
    #iterate over the elements of the graph
    while not frontier.isEmpty():
        currentState = frontier.pop()
        
        if not exploredSet.__contains__(currentState[0]):
            #check if its the goal state
            if problem.isGoalState(currentState[0]):
                return list(currentState[1])
            else: #if not the goal state then add successors to expand on later iterations
                exploredSet.add(currentState[0])
                for successor in problem.getSuccessors(currentState[0]):
                    currentNode = ((successor[0], (currentState[1] + (successor[1] ,)))
                        ,problem.getCostOfActions(currentState[1] + (successor[1] ,)))
                    
                    frontier.push(currentNode[0], currentNode[1])
        else:
            continue        

    #return None and print error message if no particular solution is found overall
    print("Oops, something went wrong. No particular solution was found using UCS algorithm!")
    return None

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    #variables
    frontier = util.PriorityQueue() #use PQ because we'll pop the node with the lowest f(s) = h(s) + g(s)
    exploredSet = set()

    #initialize the frontier
    frontier.push((problem.getStartState(), ()), 0) #((state, actions), priority)
    
    #start iterating over the graph elements
    while not frontier.isEmpty():
        currentNode = frontier.pop() # (state, actions)
       
        if problem.isGoalState(currentNode[0]):
            return list(currentNode[1])
        else: # if currentstate is not the goal add successors
            if currentNode[0] not in exploredSet:
                exploredSet.add(currentNode[0])
                
                
                #add successors
                for successor in problem.getSuccessors(currentNode[0]):
                    currentCost = problem.getCostOfActions((currentNode[1] + (successor[1] ,)))
                    currentHeuristic = heuristic(successor[0], problem)
        
                    successorNode = (successor[0], (currentNode[1] + (successor[1] ,))) # ((state, action))
                    
                    frontier.push(successorNode, (currentCost + currentHeuristic))
            else:
                continue

    #return None and print Error message if no solution was found overall
    print("Oops, something went wrong. No particular solution was found using A* algorithm!")
    return None

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
