# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)
from collections import deque
import heapq

#create a data structure to link nodes together in a path
class Node:
    #information about the node
    def __init__(self, coordinates, parent, pathSoFar=0, heuristic=0,waypointsLeft=()):
        self.coordinates = coordinates
        self.parent = parent
        self.pathSoFar = pathSoFar
        self.heuristic = heuristic
        self.cost = pathSoFar + heuristic
        self.waypointsLeft = waypointsLeft
    
    def __lt__(self, other):
        return (self.cost < other.cost)
    def __gt__(self, other):
        return (self.cost > other.cost)
    def __le__(self, other):
        return (self.cost <= other.cost)
    def __ge__(self, other):
        return (self.cost >= other.cost)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = Node(maze.start, None)
    
    Frontier = deque([start.coordinates])
    Explored = deque()
    NodeList = [start]

    finalCoord = (0,0)
    
    while list(Frontier) != []:
        #these popped coordinates will be the Parent for the next node
        currentCoord = Frontier.pop()

        if currentCoord == maze.waypoints[0]:
            print(len(maze.waypoints))
            finalCoord = currentCoord
            break
        
        currentNode = Node(None,None)
        for node in NodeList:
            if node.coordinates == currentCoord:
                currentNode = node
                break
            else:
                continue

        neighbors = maze.neighbors(currentCoord[0],currentCoord[1])
        for coord in neighbors:
            # if the neighbor is a valid space and has yet to be explored
            if maze.navigable(coord[0],coord[1]) and Explored.count(coord) < 1:
                NodeList.append(Node(coord, currentNode))
                Frontier.appendleft(coord)
                Explored.appendleft(coord)


    iterNode = Node(None,None)
    for node in NodeList:
        if node.coordinates == finalCoord:
            iterNode = node
            break
        else:
            continue
    finalPath = []

    while(iterNode != None):
        finalPath.append(iterNode.coordinates)
        iterNode = iterNode.parent


    #explore(currentNode)
        
    finalPath.reverse()


    return finalPath

#returns the manhattan distance between the node and the goal
def Manhattan_Heuristic(maze, state, waypoint):
    goal = maze.waypoints[waypoint]
    x = goal[0] - state[0]
    y = goal[1] - state[1]

    # x = state[0] - goal[0]
    # y = state[1] - goal[1]

    return abs(y) + abs(x)

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = Node(maze.start, None, heuristic=Manhattan_Heuristic(maze,maze.start,0))

    Explored = {start.coordinates : start}
    Frontier = [start]
    currentState = start.coordinates
    goalState = maze.waypoints[0]

    #We want to keep exploring the frontier until we pop the end Node
    while (currentState != goalState):
        #explore the frontier
        currentNode = heapq.heappop(Frontier)
        currentState = currentNode.coordinates
        #update Explored dict
        Explored[currentState] = currentNode

        neighbors = maze.neighbors(currentState[0],currentState[1])
        #check if the neighbors have been explored yet
        #if so, check and see if we haven't found a faster way to get to them
        for coord in neighbors:
            #check that the neighbor is a valid space to move
            if maze.navigable(coord[0],coord[1]):
                newNode = Node(coord, currentNode, currentNode.pathSoFar + 1, Manhattan_Heuristic(maze, coord, 0))
                #if the node has yet to be explored add it straight away
                if newNode.coordinates not in Explored:
                    heapq.heappush(Frontier,newNode)
                #otherwise we only want to re explore it if the cost will be lowered
                elif newNode.cost < Explored[coord].cost:
                    heapq.heappush(Frontier,newNode)
                else:
                    continue

                #print(currentNode.coordinates)



    #breaking should give a Node which when traced back through the parents should give 
    #the quickest path to the endState
    iterNode = Explored[goalState]
    finalPath = []

    while(iterNode != None):
        finalPath.append(iterNode.coordinates)
        iterNode = iterNode.parent
        
    finalPath.reverse()

    return finalPath

class Edge:
    def __init__(self, pointsConnected, distance):
        self.pointsConnected = pointsConnected
        self.distance = distance

    def __lt__(self, other):
        return (self.distance < other.distance)


#takes ints point1 & point 2 which are indices of maze.waypoints
#takes tree which is a list of all edges that currently exist to connect 2 indices
def isConnected(point1, point2, tree):
    #essentially implement a bfs to find if point1 is connected to point 2
    #or in other words if point 2 is in the same tree as point 1
    Explored = []
    Frontier = [point1]

    while len(Frontier) != 0:
        #popped the lowest numbered vertice although it should be arbitrary
        currentPoint = heapq.heappop(Frontier)
        for edge in tree:
            #if the edge contains the currentPoint and another point not in the Explored
            if edge.pointsConnected[0] == currentPoint:
                if edge.pointsConnected[1] not in Explored:
                    Frontier.append(edge.pointsConnected[1])

            elif edge.pointsConnected[1] == currentPoint:
                if edge.pointsConnected[0] not in Explored:
                    Frontier.append(edge.pointsConnected[0])
        Explored.append(currentPoint)

    return point2 in Explored

#I think this condition is unnecessary
#and edge.pointsConnected[1] not in Frontier

def Min_Manhattan_Heuristic(point1, waypoints):
    min_distance = 99999
    for point2 in waypoints:
        x = point1[0] - point2[0]
        y = point1[1] - point2[1]
        if (abs(y) + abs(x)) < min_distance:
            min_distance = abs(y) + abs(x)

    return min_distance


def point_distance(point1, point2):

    x = point1[0] - point2[0]
    y = point1[1] - point2[1]

    return abs(y) + abs(x)


def MST_Heuristic(waypoints):
    #We want to find the minimum spanning tree for the waypoints
    # make a list st we have
    #(tuple containing the two indices of waypoints that are being connected,   distance between each indice)
    edges = []
    #Minimum spanning tree based on manhattan distance
    for i in range(len(waypoints)):
        for j in range(i,len(waypoints)):
            heapq.heappush(edges, Edge((i,j), point_distance(waypoints[i], waypoints[j])) )

    #To get the lowest distanced edge from the set now we can just use
    #heappop since we overrode the __lt__ operator

    #We want a way to represent which points are connected so that we ignore them when tring to connect our tree,
    #and points that are disjoint, in order to connect those to our tree by selecting the smallest edge

    #initialize as empty
    #just a collection of edges with info on what points are connected  
    tree = []
    #need an is connected function
    while len(tree) < (len(waypoints) - 1):
        shortestEdge = heapq.heappop(edges)
        waypoint1 = shortestEdge.pointsConnected[0]
        waypoint2 = shortestEdge.pointsConnected[1]
        #if both aren't in the connected list we want to connect them
        if not isConnected(waypoint1, waypoint2, tree):
            tree.append(shortestEdge)
    MSTLength = 0
    for edge in tree:
        MSTLength += edge.distance

    return MSTLength

def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """


    #Set up tuple which will be used to keep track of different states
    zeroList = []
    for i in maze.waypoints:
        zeroList.append(0)
    visitedTracker = tuple(zeroList)

    #begin implementing A* with new heuristic and states
    startHeuristic=Min_Manhattan_Heuristic(maze.start, maze.waypoints)# + MST_Heuristic(maze.waypoints)

    start = Node(maze.start, None, heuristic=startHeuristic, waypointsLeft=visitedTracker)

    Explored = {(start.coordinates, visitedTracker) : start}
    Frontier = [start]
    currentNode = start

    #We want to keep exploring the frontier until we show 
    #that all waypoints have been discovered
    while (0 in currentNode.waypointsLeft):
        #explore the frontier
        currentNode = heapq.heappop(Frontier)
        currentState = currentNode.coordinates
        
        #update Explored dict
        Explored[(currentState, currentNode.waypointsLeft)] = currentNode

        neighbors = maze.neighbors(currentState[0],currentState[1])
        #check if the neighbors have been explored yet
        #if so, check and see if we haven't found a faster way to get to them
        for coord in neighbors:
            #check that the neighbor is a valid space to move
            if maze.navigable(coord[0],coord[1]):

                waypointsToAdd = list(currentNode.waypointsLeft)
                i = 0
                for waypoint in maze.waypoints:
                    if coord == waypoint:
                        waypointsToAdd[i] = 1
                    else:
                        i+=1
                waypointsToCalculateHeuristic = []
                for i in range(len(waypointsToAdd)):
                    if waypointsToAdd[i] == 0:
                        waypointsToCalculateHeuristic.append(maze.waypoints[i])
                                
                #add the MST + Manhat distance to closed goal Heuristic
                nodeHeuristic = Min_Manhattan_Heuristic(coord, waypointsToCalculateHeuristic)# + MST_Heuristic(waypointsToCalculateHeuristic)
                newNode = Node(coord, currentNode, currentNode.pathSoFar + 1, heuristic=nodeHeuristic,waypointsLeft=tuple(waypointsToAdd))
                #if the node has yet to be explored add it straight away
                if (newNode.coordinates,tuple(waypointsToAdd)) not in Explored:
                    heapq.heappush(Frontier,newNode)
                #otherwise we only want to re explore it if the cost will be lowered
                #tried // or len(newNode.waypointsLeft) < len(Explored[coord].waypointsLeft) // and yes work!!!
                elif newNode.cost < Explored[(coord, tuple(waypointsToAdd))].cost:
                    heapq.heappush(Frontier,newNode)
                else:
                    continue

    iterNode = currentNode
    finalPath = []

    while(iterNode != None):
        finalPath.append(iterNode.coordinates)
        iterNode = iterNode.parent

    finalPath.reverse()

    return finalPath


def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    
    #Set up tuple which will be used to keep track of different states
    zeroList = []
    for i in maze.waypoints:
        zeroList.append(0)
    visitedTracker = tuple(zeroList)

    #begin implementing A* with new heuristic and states
    MST_Start = MST_Heuristic(maze.waypoints)
    startHeuristic=Min_Manhattan_Heuristic(maze.start, maze.waypoints) + MST_Start

    start = Node(maze.start, None, heuristic=startHeuristic, waypointsLeft=visitedTracker)

    Explored = {(start.coordinates, visitedTracker) : start}
    Frontier = [start]
    SolvedMST = {visitedTracker : MST_Start}
    currentNode = start

    #We want to keep exploring the frontier until we pop the end Node
    while (0 in currentNode.waypointsLeft):
        #explore the frontier
        currentNode = heapq.heappop(Frontier)
        currentState = currentNode.coordinates
        
        #update Explored dict
        Explored[(currentState, currentNode.waypointsLeft)] = currentNode

        neighbors = maze.neighbors(currentState[0],currentState[1])
        #check if the neighbors have been explored yet
        #if so, check and see if we haven't found a faster way to get to them
        for coord in neighbors:
            #check that the neighbor is a valid space to move
            if maze.navigable(coord[0],coord[1]):

                waypointsToAdd = list(currentNode.waypointsLeft)
                i = 0
                for waypoint in maze.waypoints:
                    if coord == waypoint:
                        waypointsToAdd[i] = 1
                    else:
                        i+=1
                waypointsToCalculateHeuristic = []
                for i in range(len(waypointsToAdd)):
                    if waypointsToAdd[i] == 0:
                        waypointsToCalculateHeuristic.append(maze.waypoints[i])
                                
                #Will save A LOT of runtime by precalculating and saving already solved MST's
                if tuple(waypointsToAdd) not in SolvedMST:
                    SolvedMST[tuple(waypointsToAdd)] =  MST_Heuristic(waypointsToCalculateHeuristic)
                #add the MST + Manhat distance to closed goal Heuristic
                nodeHeuristic = Min_Manhattan_Heuristic(coord, waypointsToCalculateHeuristic) + SolvedMST[tuple(waypointsToAdd)]
                newNode = Node(coord, currentNode, currentNode.pathSoFar + 1, heuristic=nodeHeuristic,waypointsLeft=tuple(waypointsToAdd))
                #if the node has yet to be explored add it straight away
                if (newNode.coordinates,tuple(waypointsToAdd)) not in Explored:
                    heapq.heappush(Frontier,newNode)
                #otherwise we only want to re explore it if the cost will be lowered
                #tried // or len(newNode.waypointsLeft) < len(Explored[coord].waypointsLeft) // and yes work!!!
                elif newNode.cost < Explored[(coord, tuple(waypointsToAdd))].cost:
                    heapq.heappush(Frontier,newNode)
                else:
                    continue

    iterNode = currentNode
    finalPath = []

    while(iterNode != None):
        finalPath.append(iterNode.coordinates)
        iterNode = iterNode.parent

    finalPath.reverse()

    return finalPath

   # return []

def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
    
            
