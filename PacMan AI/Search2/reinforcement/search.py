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
from game import Directions
from util import Queue

s = Directions.SOUTH
w = Directions.WEST
n = Directions.NORTH
e = Directions.EAST

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

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    
    starter = (problem.getStartState(), None, [])
    if problem.isGoalState(problem.getStartState()):
        return starter

    stack = util.Stack()
    stack.push(starter)
    explored = []
    visited = {problem.getStartState(): None}

    while True:
        if stack.isEmpty():
            return None
        parentNode = stack.pop()
        parentCoor = parentNode[0]
        parentDir = parentNode[1]
        print parentNode
        
        if problem.isGoalState(parentCoor):
            #visited[parentCoor] = [parentCoor, parentDir]
            return solution(visited, parentCoor)
        explored.append(parentCoor)
        successors = problem.getSuccessors(parentCoor)
        for child in successors:
            childCoor, childDir, childWeight = child
            print childCoor
            if childCoor not in explored:
                stack.push(child)
                visited[childCoor] = [parentCoor, childDir]
    '''
    # function DEPTH-LIMITED-SEARCH(problem, limit ) returns a solution, or failure/cutoff
    starter = (problem.getStartState(), None, [])   # MAKE-NODE(problem.INITIAL-STATE)
    explored = []
    
    visited = {problem.getStartState(): None}
    return recursiveDepthFirstSearch(starter, problem, visited, explored, float('inf'))
    
    # function RECURSIVE-DLS(node, problem, limit ) returns a solution, or failure/cutoff
def recursiveDepthFirstSearch(parentNode, problem, visited, explored, limit):
    # if problem.GOAL-TEST(node.STATE) then return SOLUTION(node)
    parentCoor = parentNode[0]
    parentDir = parentNode[1]
    print parentNode
    print parentCoor
    print "is goal: ", problem.isGoalState(parentCoor)
    if problem.isGoalState(parentCoor):
    #   visited[parentCoor] = [parentCoor, parentDir]
        print visited
        return solution(visited, parentCoor)
    # else if limit = 0 then return cutoff
    # else
    # cutoff occurred? <- false
    # for each action in problem.ACTIONS(node.STATE) do
    explored.append(parentCoor)
    successors = problem.getSuccessors(parentCoor)
    for child in successors:    # child <- CHILD-NODE(problem, node, action)
        childCoor, childDir, childWeight = child
        print childCoor
        if childCoor not in explored:
            visited[childCoor] = [parentCoor, childDir]
            print visited
            #input('press any key')
            result = recursiveDepthFirstSearch(child, problem, visited, explored, limit - 1)
            return result
        # if result = cutoff then cutoff occurred? <- true
        # else if result != failure then return result
        #if result != []
         #   return result
        # if cutoff occurred? then return cutoff else return failure
    '''
    
   

def breadthFirstSearch(problem):

    # starter <- a node with STATE = problem.INITIAL-STATE, PATH-COST = 0
    starter = (problem.getStartState(), None, [])
    
    # if problem.GOAL-TEST(node.STATE) then return SOLUTION(node)
    if problem.isGoalState(problem.getStartState()):
        return []
    
    #print "Is goal (1,1)? ", problem.isGoalState((1,1))
    # frontier <- a FIFO queue with node as the only element
    queue = util.Queue()  # type: Queue
    queue.push(starter)
    
    # explored <- an empty set
    explored = []
    
    # A dictionary to keep track of the best node to come from for each node visited so far.
    # This dictionary is required so that we can reconstruct the solution.
    # The dictionary is structured like this: {to_node1:best_from_node_for_node1,
    # to_node2:best_from_node_for_node2, ...}
    # ie. it is structured {to_node : from_node}
    visited = {problem.getStartState():None}
    

    while True:
        # if EMPTY?( frontier) then return failure
        if queue.isEmpty():
            return None
        
        # node <- POP( frontier ) /* chooses the shallowest node in frontier */
        parentNode = queue.pop()
        parentCoor = parentNode[0]
        parentDir = parentNode[1]
        parentWeight = parentNode[2]

           # if problem.GOAL-TEST(child .STATE) then
        if problem.isGoalState(parentCoor):
            #visited[childCoor] = [parentCoor, childDir]
            # return SOLUTION(child )
            return solution(visited, parentCoor)
        # add node.STATE to explored
        explored.append(parentCoor)
        
        # for each action in problem.ACTIONS(node.STATE) do
        succesors = problem.getSuccessors(parentCoor)
        for child in succesors:     # child <- CHILD-NODE(problem, node, action)
            childCoor, childDir, childWeight = child
            # if child .STATE is not in explored or frontier then
            if childCoor not in explored and childCoor not in queue:
                # frontier <- INSERT(child , frontier )
                queue.push(child)
                visited[childCoor] = [parentCoor, childDir]


def solution(visited, goal):
    #print "Visited: ", visited
    path = []
    # to_node = goal
    # visited will be a dict. containing the best node to come from. Keys are nodes, value is the best nodes to come from to get to key 
    from_list = visited[goal]

    #print from_list
    while from_list != None:
        from_node = from_list[0]
        dir = from_list[1]
        path = [dir] + path
        to_node = from_node
        from_list = visited[to_node]
    return path

def uniformCostSearch(problem):
    """function UNIFORM-COST-SEARCH(problem) returns a solution, or failure"""
    priQueue = util.PriorityQueue()
    priQueue.push( (problem.getStartState(), [], 0), 0 )
    expanded = []

    while not priQueue.isEmpty():
        coordinate, currentSolution, currentCost = priQueue.pop()

        if not coordinate in expanded:
            expanded.append(coordinate)

            if problem.isGoalState(coordinate):
                return currentSolution

            for child, childDirection, cost in problem.getSuccessors(coordinate):
                priQueue.push((child, currentSolution+[childDirection], currentCost + cost), currentCost + cost)

    return []
   

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    priQueue = util.PriorityQueue()
    priQueue.push( (problem.getStartState(), [], 0), 0 )
    expanded = []

    while not priQueue.isEmpty():
        coordinate, currentSolution, currentCost = priQueue.pop()

        if not coordinate in expanded:
            expanded.append(coordinate)

            if problem.isGoalState(coordinate):
                return currentSolution

            for childCoor, childDirection, cost in problem.getSuccessors(coordinate):
                estCost = heuristic(childCoor, problem)
                totalEstCost = estCost + currentCost + cost
                priQueue.push((childCoor, currentSolution+[childDirection], currentCost + cost), totalEstCost)
                
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
