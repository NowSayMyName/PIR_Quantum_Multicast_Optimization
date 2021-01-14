#!/usr/bin/env python3

import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

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

    def setEdges(self, edgeList):
        for i in edgeList:
            self.edges.append(i)

class Edge:
    def __init__(self, nodeA, nodeB):
        self.nodeA = nodeA
        self.nodeB = nodeB
        self.length = nodeA.distance(nodeB)

    def getLength(self):
        return self.length

    def __repr__(self):
        return "(" + str(self.nodeA.x) + "," + str(self.nodeA.y) + ") " + str(self.length) + " --> (" + str(self.nodeB.x) + "," + str(self.nodeB.y) + ")"

class Route:
    def __init__(self, nodeB):
        self.path = []
        self.neededDestination = nodeB
        self.distance = 0.0
        self.fitness = 0.0

    def copy(self, otherRoute):
        for i in otherRoute.path:
            self.addNode(i)
        self.distance = otherRoute.distance
        self.fitness = otherRoute.fitness

    def addNode(self, node):
        self.path.append(node)

    def removeFromNode(self, i):
        for n in range(len(self.path)-1, i-1, -1):
            'print("removing " + str(n))'
            self.path.pop(n)

    def routeDistance(self):
        self.distance = 0
        for i in range(0, len(self.path)-1):
            self.distance += self.path[i].distance(self.path[i+1])

    def routeFitness(self):
        self.fitness = 1/self.distance
        if self.neededDestination in self.path:
            self.fitness += 1
    
    def __repr__(self):
        return "from " + str(self.path[0]) + " to " + str(self.neededDestination) + " : " + str(self.path) + " fitness: " + str(self.fitness) +  " distance: " + str(self.distance) + "\n"


nodeNumber = 25
edgePerNode = 4
maxPathLength = 15
sourceNode = 1
destinationNode = 1

sourceNodes = []
destinationNodes = []
nodeList = []
edgeList = []
tempEdgeList = []

for i in range(0, nodeNumber):
    nodeList.append(Node(x=int(random.random() * 200), y=int(random.random() * 200)))

for i in range(0, sourceNode):
    sourceNodes.append(random.choice(nodeList))

for i in range(0, destinationNode):
    destinationNodes.append(random.choice(nodeList))

for node in nodeList:
    'print("node :" + str(node))'
    for otherNode in nodeList:
        if otherNode != node:
            'print("other node :" + str(otherNode))'
            if len(tempEdgeList) < edgePerNode:
                tempEdgeList.append(Edge(node, otherNode))
                tempEdgeList.sort(key=Edge.getLength)
                'print(tempEdgeList)'
            else:
                n = 0
                while n < edgePerNode:
                    if node.distance(otherNode) < tempEdgeList[n].length:
                        tempEdgeList.insert(n, Edge(node, otherNode))
                        'print("added : " + str(tempEdgeList[n]))'
                        'print("removed : " + str(tempEdgeList.pop(edgePerNode)))'
                        'print(tempEdgeList)'
                        n = edgePerNode
                    n += 1
    'print()'
    for i in tempEdgeList:
        edgeList.append(i)
    node.setEdges(tempEdgeList)
    tempEdgeList.clear()
'print(nodeList)'
'print(edgeList)'

def geneticAlgorithm(population, sourceNodes, destinationNode, destinationNodes, maxPathLength, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population, sourceNodes, destinationNode, destinationNodes, maxPathLength)
    'print(pop)'
    fitnessTotal = 0
    for n in pop:
            fitnessTotal += n.fitness
    print(fitnessTotal)
    
    'print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))'
    
    for i in range(0, generations):
        pop = nextGeneration(nodeList, pop, popSize, eliteSize, mutationRate)
        fitnessTotal = 0
        for n in pop:
            fitnessTotal += n.fitness
            'print(n.fitness)'
        'print("Distance: " + str(1 / rankRoutes(pop)[0][1]))'
        print(fitnessTotal)
    
    'print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))'

def initialPopulation(popSize, nodeList, sourceNodes, destinationNode, destinationNodes, maxPathLength):
    population = []

    for i in range(0, popSize):
        for n in range(0, destinationNode):
            population.append(createRoute(nodeList, sourceNodes[0], destinationNodes[n], maxPathLength))
    return population

def createRoute(nodeList, nodeA, nodeB, maxPathLength):
    node = nodeA
    route = Route(nodeB)
    route.addNode(nodeA)
    j = 0
    while node != nodeB and j < maxPathLength:
        node = random.choice(node.edges).nodeB
        route.addNode(node)
        j += 1
    route.routeDistance()
    route.routeFitness()
    'print(route.path)'
    return route

def rankRoutes(population):
    for i in population:
        i.routeFitness()
    return sorted(population, key=lambda route: route.fitness, reverse= True)
    """print(population[0].fitness > population[50].fitness)
    print()
    for i in population:
        print(i.fitness)
    print()"""

def breedPopulation(elitePop, eliteSize, popSize, destinationNode):
    children = []
    for n in range(0, popSize*destinationNode):
        rand = random.choice(elitePop)
        r = Route(rand.neededDestination)
        r.copy(rand)
        children.append(r)
    return children   

def mutatePopulation(nodeList, pop, mutationRate, maxPathLength):
    j = 0
    for n in pop:
        j += 1
        if random.random() < mutationRate:
            """print()
            print("mutating " + str(j))
            print(n)"""
            index = (int)(random.random()*(len(n.path)-1)) + 1
            n.removeFromNode(index)
            node = n.path[index-1]
            j = len(n.path)
            while node != n.neededDestination and j < maxPathLength:
                node = random.choice(node.edges).nodeB
                n.addNode(node)
                j += 1
            n.routeDistance()
            n.routeFitness()
            'print(n)'

def nextGeneration(nodeList, pop, popSize, eliteSize, mutationRate):
    pop = rankRoutes(pop)
    'print(pop[0])'
    """print("initial")"""
    print(pop)
    elitePop = pop[0: eliteSize]
    'print(elitePop)'
    pop = breedPopulation(elitePop, eliteSize, popSize, destinationNode)
    """print("newly bred")
    print(pop)"""
    mutatePopulation(nodeList, pop, mutationRate, maxPathLength)
    """print("mutated")
    print(pop)"""
    return pop

geneticAlgorithm(population=nodeList, sourceNodes=sourceNodes, destinationNode=destinationNode, destinationNodes=destinationNodes, maxPathLength=maxPathLength, popSize=10, eliteSize=2, mutationRate=0.3, generations=50)
