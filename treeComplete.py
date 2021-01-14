#!/usr/bin/env python3

import numpy as np, turtle, math, random, matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.edges = []
    
    def distance(self, node):
        xDis = abs(self.x - node.x)
        yDis = abs(self.y - node.y)
        return int(np.sqrt((xDis ** 2) + (yDis ** 2)))
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class Edge:
    def __init__(self, nodeA, nodeB):
        self.nodeA = nodeA
        self.nodeB = nodeB
        self.length = nodeA.distance(nodeB)

    def getLength(self):
        return self.length

    def reverse(self):
        return Edge(self.nodeB, self.nodeA)

    def __repr__(self):
        return "(" + str(self.nodeA.x) + "," + str(self.nodeA.y) + ") " + str(self.length) + " --> (" + str(self.nodeB.x) + "," + str(self.nodeB.y) + ")"

class Route:
    def __init__(self, nodeB):
        self.path = []
        self.edges = []
        self.neededDestination = nodeB
        self.distance = 0.0
        self.invDistMin = 0

    def copy(self):
        route = Route(self.neededDestination)
        for i in self.path:
            route.addNode(i)
        for i in self.edges:
            route.edges.append(i)
        route.routeDistance()
        return route

    def distanceToDest(self):
        min = 0
        for n in reversed(self.path):
            if min != 1:
                dist = 1 / float(n.distance(self.neededDestination)+1)
                if dist > min:
                    min = dist
        self.invDistMin = min

    def addNode(self, node):
        self.path.append(node)

    def removeFromNode(self, i):
        for n in range(len(self.path)-1, i-1, -1):
            self.path.pop(n)
            self.edges.pop(n-1)

    def routeDistance(self):
        self.distance = 0
        for i in range(0, len(self.path)-1):
            self.distance += self.path[i].distance(self.path[i+1])
    
    def __repr__(self):
        return "from " + str(self.path[0]) + " to " + str(self.neededDestination) + " : " + str(self.path) + " distance: " + str(self.distance) + "\n"

class Tree:
    def __init__(self, routes):
        self.routes = []
        for i in range(len(routes)):
            self.routes.append(routes[i].copy())
        self.edges = []
        self.distance = 0
        self.fitness = 0.0
        self.treeFitness()

    def treeLength(self):
        self.distance = 0
        for i in self.edges:
            self.distance += i.length

    def treeEdges(self):
        self.edges = []
        for i in self.routes:
            for j in i.edges:
                if j not in self.edges and j.reverse() not in self.edges:
                    self.edges.append(j)

    def treeFitness(self):
        self.treeEdges()
        self.treeLength()
        self.fitness = 0.0
        for i in self.routes:
            self.fitness += 2 / (float(i.path[-1].distance(i.neededDestination))+1)
            """i.distanceToDest()
            self.fitness += 2 * i.invDistMin"""
        self.fitness += 100 / float(self.distance)

    def __repr__(self):
        repr = "" 
        for i in self.routes:
            repr += str(i)
        return repr + "TOTAL FITNESS= " + str(self.fitness) + " - TOTAL DISTANCE= " + str(self.distance) + "\n"

    def copy(self, otherTree):
        self.routes = []
        for i in otherTree.routes:
            self.routes.append(i)
        self.treeEdges()
        self.treeLength()
        self.treeFitness()


nodeNumber = 25
edgePerNode = 4
maxPathLength = 25
sourceNode = 1
destinationNode = 5

sourceNodes = []
destinationNodes = []
nodeList = []
edgeList = []
tempEdgeList = []

for i in range(0, nodeNumber):
    nodeList.append(Node(x=int(random.random() * 200), y=int(random.random() * 200)))

for i in range(0, sourceNode):
    sourceNodes.append(random.choice(nodeList))

while len(destinationNodes) < destinationNode:
    rand = random.choice(nodeList)
    if rand not in destinationNodes and rand not in sourceNodes:
        destinationNodes.append(rand)

for node in nodeList:
    for otherNode in nodeList:
        if otherNode != node:
            if len(tempEdgeList) < edgePerNode:
                tempEdgeList.append(Edge(node, otherNode))
                tempEdgeList.sort(key=Edge.getLength)
            else:
                n = 0
                while n < edgePerNode:
                    if node.distance(otherNode) < tempEdgeList[n].length:
                        tempEdgeList.insert(n, Edge(node, otherNode))
                        n = edgePerNode
                    n += 1
    for i in range(edgePerNode):
        edgeList.append(tempEdgeList[i])
        node.edges.append(tempEdgeList[i])
    tempEdgeList.clear()

def geneticAlgorithm(edgeList, population, sourceNodes, destinationNode, destinationNodes, maxPathLength, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population, sourceNodes, destinationNode, destinationNodes, maxPathLength)
    pop[0].treeFitness()
    progress = []
    progress.append(pop[0].fitness)
    print(pop[0].fitness)

    # drawGraph(edgeList, pop[0])

    
    for i in range(0, generations):
        rankTrees(pop)
        # drawGraph(edgeList, pop[0])
        progress.append(pop[0].fitness)
        print(pop[0].fitness)
        pop = nextGeneration(nodeList, pop, popSize, eliteSize, mutationRate)
 
    plt.plot(progress)
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.show()
    
def initialPopulation(popSize, nodeList, sourceNodes, destinationNode, destinationNodes, maxPathLength):
    population = []

    for i in range(0, popSize):
        t = createTree(nodeList, sourceNodes[0], destinationNodes, maxPathLength)
        population.append(t)
    return population

def createTree(nodeList, nodeA, destinationNodes, maxPathLength):
    routes = []
    for i in destinationNodes:
        routes.append(createRoute(nodeList, nodeA, i, maxPathLength))
    return Tree(routes)


def createRoute(nodeList, nodeA, nodeB, maxPathLength):
    node = nodeA
    route = Route(nodeB)
    route.addNode(nodeA)
    j = 0
    while node != nodeB and j < maxPathLength-1:
        edge = random.choice(node.edges)
        node = edge.nodeB
        route.edges.append(edge)
        route.addNode(node)
        j += 1
    route.routeDistance()
    return route

def rankTrees(population):
    for i in population:
        i.treeFitness()
    population.sort(key=lambda tree: tree.fitness, reverse= True)

def breedPopulation(pop, eliteSize, popSize, destinationNode):
    children = []
    """print("\nselected")
    print(pop[0:eliteSize])"""
    for i in range(popSize):
        rand = pop[i%eliteSize]
        r = Tree(rand.routes)
        children.append(r)
    return children   

def mutatePopulation(nodeList, pop, mutationRate, maxPathLength):
    for i in pop:
        for j in i.routes:
            if random.random() < mutationRate:
                index = (int)(random.random()*(len(j.path)-1)) + 1
                j.removeFromNode(index)
                node = j.path[index-1]
                k = len(j.path)
                while node != j.neededDestination and k < maxPathLength-1:
                    edge = random.choice(node.edges)
                    node = edge.nodeB
                    j.addNode(node)
                    j.edges.append(edge)
                    k += 1
                j.routeDistance()

def nextGeneration(nodeList, pop, popSize, eliteSize, mutationRate):
    """print("\n\n\ninitial")
    print(pop)"""
    pop = breedPopulation(pop, eliteSize, popSize, destinationNode)
    """print("\nnewly bred")
    print(pop)"""
    mutatePopulation(nodeList, pop, mutationRate, maxPathLength)
    """print("\nmutated")
    print(pop)"""
    return pop

def drawGraph(edgeList, tree):
    turtle.pencolor("black")
    for e in edgeList:
        turtle.penup()
        turtle.goto(1.5*e.nodeA.x, 1.5*e.nodeA.y)
        turtle.pendown()
        turtle.goto(1.5*e.nodeB.x, 1.5*e.nodeB.y)

    turtle.pencolor("red")
    for e in tree.edges:
        turtle.penup()
        turtle.goto(1.5*e.nodeA.x, 1.5*e.nodeA.y)
        turtle.pendown()
        turtle.goto(1.5*e.nodeB.x, 1.5*e.nodeB.y)

    turtle.hideturtle()
    turtle.exitonclick()


geneticAlgorithm(edgeList=edgeList, population=nodeList, sourceNodes=sourceNodes, destinationNode=destinationNode, destinationNodes=destinationNodes, maxPathLength=maxPathLength, popSize=100, eliteSize=15, mutationRate=0.3, generations=250)
