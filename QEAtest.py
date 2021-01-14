#!/usr/bin/env python3

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
import numpy as np, math, random, matplotlib.pyplot as plt

pi = math.pi

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

class QRoute:
    def __init__(self, qc, nodeA, nodeB, maxPathLength):
        self.qc = qc
        self.maxPathLength = maxPathLength
        self.path = []
        self.binaryPath = ""
        self.edges = []
        self.neededSource = nodeA
        self.neededDestination = nodeB
        self.distance = 0.0
        self.invDistMin = 0

    def distanceToDest(self):
        min = 0
        for n in reversed(self.path):
            if min != 1:
                dist = 1 / float(n.distance(self.neededDestination)+1)
                if dist > min:
                    min = dist
        self.invDistMin = min

    def routeDistance(self):
        self.distance = 0
        for i in range(0, len(self.path)-1):
            self.distance += self.path[i].distance(self.path[i+1])

    def measure(self):
        for i in range(len(self.qc)):
            self.qc[i].barrier(0)
            self.qc[i].measure(0, 0)

    def quantumToClassical(self):
        self.measure()
        backend = Aer.get_backend('qasm_simulator')
        shots = 1
        completeMeasure = ""
        for i in range(len(self.qc)):
            job = execute(self.qc[i], backend=backend, shots=shots)
            results = job.result()
            answer = results.get_counts(self.qc[i]).keys()
            for j in answer:
                completeMeasure += str(j)
        self.path = []
        self.path.append(self.neededSource)
        self.edges = []
        self.binaryPath = ""

        j = 0
        'print(completeMeasure)'
        while j < len(completeMeasure) / 2:
            if self.neededDestination in self.path:
                for k in range(2*j, len(self.qc)):
                    self.binaryPath += "1"
                j = len(self.qc) / 2
            else:
                binaryString = completeMeasure[2*j: 2*j+2]
                decInt = int(binaryString, 2)
                if decInt != 3:
                    self.edges.append(self.path[j].edges[decInt])
                    self.path.append(self.edges[-1].nodeB)
                    self.binaryPath += binaryString
                    j += 1
                elif decInt == 3:
                    r = int(3 * random.random())
                    if r == 0:
                        binaryString = "00"
                    elif r == 1:
                        binaryString = "01"
                    else:
                        binaryString = "10"
                    decInt = int(binaryString, 2)
                    self.edges.append(self.path[j].edges[decInt])
                    self.path.append(self.edges[-1].nodeB)
                    self.binaryPath += binaryString
                    j += 1
        'print(self.binaryPath)'
    
    def __repr__(self):
        return "from " + str(self.path[0]) + " to " + str(self.neededDestination) + " : " + str(self.path) + " distance: " + str(self.distance) + "\n"

class QTree:
    def __init__(self, routes):
        self.routes = routes
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
            i.quantumToClassical()
            for j in i.edges:
                if j not in self.edges and j.reverse() not in self.edges:
                    self.edges.append(j)

    def treeFitness(self):
        self.treeEdges()
        self.treeLength()
        self.fitness = 0.0
        for i in self.routes:
            self.fitness += 2 / (float(i.path[-1].distance(i.neededDestination))+1)
        self.fitness += 100 / float(self.distance)

    def __repr__(self):
        repr = "" 
        for i in self.routes:
            repr += str(i)
        return repr + "TOTAL FITNESS= " + str(self.fitness) + " - TOTAL DISTANCE= " + str(self.distance) + "\n"

nodeNumber = 10
edgePerNode = 3
maxPathLength = 10
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

def initialQPopulation(popSize, nodeList, sourceNodes, destinationNode, destinationNodes, maxPathLength):
    population = []

    for i in range(0, popSize):
        t = createQTree(sourceNodes, destinationNode, destinationNodes, maxPathLength)
        population.append(t)
    return population

def createQTree(sourceNodes, destinationNode, destinationNodes, maxPathLength):
    qroutes = []
    for i in range(destinationNode):
        qb = []
        for j in range(2*maxPathLength):
            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qb.append(qc)
        qroutes.append(QRoute(qb, sourceNodes[0], destinationNodes[i], maxPathLength))
    return QTree(qroutes)

def nextGeneration(nodeList, pop, popSize, mutationRate, generation):
    rankQTrees(pop)
    for i in pop:
        for j in i.routes:
            print(j.binaryPath)
    mutateQPopulation(nodeList, pop, mutationRate, maxPathLength, generation)
    return pop

def rankQTrees(population):
    for i in population:
        i.treeFitness()
    population.sort(key=lambda tree: tree.fitness, reverse= True)

def qGeneticAlgorithm(edgeList, population, sourceNodes, destinationNode, destinationNodes, maxPathLength, popSize, mutationRate, generations):
    pop = initialQPopulation(popSize, population, sourceNodes, destinationNode, destinationNodes, maxPathLength)
    pop[0].treeFitness()
    progress = []
    progress.append(pop[0].fitness)
    print(pop[0].fitness)
    
    for i in range(0, generations):
        print()
        print("generation: " + str(i+1))
        pop = nextGeneration(nodeList, pop, popSize, mutationRate, (i+1))
        print(pop[0].fitness)
        """progress.append(pop[0].fitness)"""
 
    plt.plot(progress)
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.show()

def mutateQPopulation(nodeList, pop, mutationRate, maxPathLength, generation):
    theta = mutationRate * pi / float(16 * generation)
    for i in pop:
        for j in i.routes:
            index = i.routes.index(j)
            for q in range(len(j.qc)):
                sign = 0
                if j.binaryPath[q] == pop[0].routes[index].binaryPath[q] :
                    sign = 0
                elif j.binaryPath[q] == 0 and pop[0].routes[index].binaryPath[q] == 1:
                    sign = -1
                else:
                    sign = 1
                j.qc[q].u3(theta*sign, 0, 0, 0)


qGeneticAlgorithm(edgeList=edgeList, population=nodeList, sourceNodes=sourceNodes, destinationNode=destinationNode, destinationNodes=destinationNodes, maxPathLength=maxPathLength, popSize=10, mutationRate=1.0, generations=250)
