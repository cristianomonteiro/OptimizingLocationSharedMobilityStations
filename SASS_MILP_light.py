#RUN POSTGRESQL BY USING: postgres -D '/Users/cristianomartinsm/Library/Application Support/Postgres/var-14' >logfile 2>&1 &

import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from time import sleep
import pathlib
import bz2
import pickle
import numpy as np
from math import floor

from loadSplitEdges import loadMultiGraphEdgesSplit

class LightEdge:
    def __init__(self, G, u, v, idEdge, utilityValue, parkingExpenses, distanceCutOff):
        self.idEdge = idEdge
        self.u = u
        self.v = v
        self.utilityValue = utilityValue
        self.parkingExpenses = parkingExpenses
        self.tagVariable = self.idEdge + '-' + str(self.u) + '-' + str(self.v)

        self.omegaEdgesSet, self.alphaEdgesSet = self.reachableEdges(G, u, distanceCutOff)
        edgesSetFromV, alphaEdgesSetFromV = self.reachableEdges(G, v, distanceCutOff)
        self.omegaEdgesSet.update(edgesSetFromV)
        self.alphaEdgesSet.update(alphaEdgesSetFromV)

        self.omegaEdgesSet = self.omegaEdgesSet - self.alphaEdgesSet - {self.idEdge}
        self.alphaEdgesSet = self.alphaEdgesSet - {self.idEdge}

        self.omegaEdgesSet = list(self.omegaEdgesSet)
        self.alphaEdgesSet = list(self.alphaEdgesSet)

    #Return all id of edges adjacent to the current vertex
    def reachableEdges(self, G, u, cutoff):
        distances = nx.single_source_dijkstra_path_length(G, u, cutoff=cutoff, weight='length')
        vertices = distances.keys()

        edges = []
        edgesAlpha = []
        for vertex in vertices:
            for item in G.edges(vertex, data=True):
                if item[2]['utilityvalue'] != 0:
                    edges.append(item[2]['idedge'])

                    #cutoff is divided by 2 to allow edges not too far in Omega to also have stations
                    if distances[vertex] < cutoff/2:
                        edgesAlpha.append(item[2]['idedge'])

        return set(edges), set(edgesAlpha)

class Edge:    
    def __init__(self, u, v, idEdge, utilityValue, parkingExpenses, alphaEdgesSet, omegaEdgesSet, variablesAlreadyCreated, model):
        self.idEdge = idEdge
        self.u = u
        self.v = v
        self.utilityValue = utilityValue
        self.parkingExpenses = parkingExpenses
        self.tagVariable = self.idEdge + '-' + str(self.u) + '-' + str(self.v)
        self.variable = Edge.createOrGetEdgeVariable(model, self.idEdge, self.tagVariable)

        self.omegaEdgesSet = omegaEdgesSet
        self.alphaEdgesSet = alphaEdgesSet

        if variablesAlreadyCreated:
            self.omegaVariables, self.alphaVariables = self.getNeededVariables(model)

    #Return all id of edges adjacent to the current vertex
    def reachableEdges(self, G, u, cutoff):
        distances = nx.single_source_dijkstra_path_length(G, u, cutoff=cutoff, weight='length')
        vertices = distances.keys()

        edges = []
        edgesAlpha = []
        for vertex in vertices:
            for item in G.edges(vertex, data=True):
                if item[2]['utilityvalue'] != 0:
                    edges.append(item[2]['idedge'])

                    #cutoff is divided by 2 to allow edges not too far in Omega to also have stations
                    if distances[vertex] < cutoff/2:
                        edgesAlpha.append(item[2]['idedge'])

        return set(edges), set(edgesAlpha)
    
    @staticmethod
    def getVariable(model, varName):
        variable = None
        try:
            variable = model.getVarByName(varName)
        finally:
            return variable

    @staticmethod
    def createOrGetEdgeVariable(model, varName, tagVariable=None):
        #if edgeVariable == None:
        if not varName in Edge.createdVariables:
            Edge.createdVariables.add(varName)

            edgeVariable = model.addVar(name=varName, vtype=GRB.BINARY)
            #VTag is needed for finding the variable in the json file after optimizing the model
            edgeVariable.VTag = tagVariable
        else:
            edgeVariable = Edge.getVariable(model, varName)
        
        return edgeVariable

    def getNeededVariables(self, model):
        #self.variable = Edge.createOrGetEdgeVariable(model, self.idEdge)
        alphaVariables = []
        omegaVariables = []
        
        for reachedIdEdge in self.alphaEdgesSet:
            alphaVariables.append(Edge.createOrGetEdgeVariable(model, reachedIdEdge))

        for reachedIdEdge in self.omegaEdgesSet:
            omegaVariables.append(Edge.createOrGetEdgeVariable(model, reachedIdEdge))

        return omegaVariables, alphaVariables

def buildGurobiModel(budget=float('inf'), nIterations=0, distanceCutOff=200):
    #Managing the created variables and constraints outside the model to avoid calling the expensive "model.update()"
    Edge.createdVariables = set()
    Edge.createdOmegaConstraints = set()

    G = loadMultiGraphEdgesSplit(nIterations=nIterations)

    count = 0
    nextPrint = 1
    #Defines a list that requires less RAM than a graph
    print("Creating the light Graph")
    GList = []
    for u, v, data in G.edges(data=True):
        if data['utilityvalue'] == 0:
            continue

        count += 1
        if count == nextPrint:
            print(count)
            nextPrint *= 2

        edge = LightEdge(G, u, v, data['idedge'], data['utilityvalue'], data['parkingexpenses'], distanceCutOff)
        GList.append((edge.u, edge.v, edge.idEdge, edge.utilityValue, edge.parkingExpenses, edge.alphaEdgesSet, edge.omegaEdgesSet))

    G = None

    #Create a Gurobi Model
    model = gp.Model("SASS")

    count = 0
    nextPrint = 1
    #Create the variables
    print("Creating the variables")
    for u, v, idEdge, utilityValue, parkingExpenses, alphaEdgesSet, omegaEdgesSet in GList:
        if utilityValue == 0:
            continue

        count += 1
        if count == nextPrint:
            print(count)
            nextPrint *= 2

        edge = Edge(u, v, idEdge, utilityValue, parkingExpenses, alphaEdgesSet, omegaEdgesSet, False, model)

    model.update()
    count = 0
    nextPrint = 1
    objective = 0
    sumVarsCost = 0
    #Define the constraints
    print("Defining the constraints", len(model.getVars()), len(Edge.createdVariables))
    while len(GList) > 0:
        u, v, idEdge, utilityValue, parkingExpenses, alphaEdgesSet, omegaEdgesSet = GList.pop(0)
        if utilityValue == 0:
            continue

        count += 1
        if count == nextPrint:
            print(count)
            nextPrint *= 2

        edge = Edge(u, v, idEdge, utilityValue, parkingExpenses, alphaEdgesSet, omegaEdgesSet, True, model)

        objective += edge.utilityValue * edge.variable
        sumVarsCost += edge.parkingExpenses * edge.variable

        model.addConstr(edge.variable + sum(edge.alphaVariables) <= 1, 'alpha_' + edge.idEdge)
        for involvedVariable in edge.omegaVariables:
            currentAndOmega = sorted([edge.idEdge, involvedVariable.VarName])
            currentAndOmega = currentAndOmega[0] + '|' + currentAndOmega[1]
            if currentAndOmega not in Edge.createdOmegaConstraints:
                Edge.createdOmegaConstraints.add(currentAndOmega)
                model.addConstr(edge.variable + involvedVariable <= 1, 'leq_1_' + edge.idEdge + '_' + involvedVariable.VarName)
        
    #Adding the costs constraint
    model.addConstr(sumVarsCost <= budget, "budget_constraint")
    
    #Set objective: maximize the utility value by allocating stations on edges
    model.setObjective(objective, GRB.MAXIMIZE)

    #Setting an upper bound with a previous bound found to speed up the MIP
    #model.addConstr(objective <= 1.9998e+08, 'boundary_value')

    print("MODEL BUILT!!", len(model.getVars()), len(Edge.createdVariables))

    #Freeing space of data structures not needed anymore
    Edge.createdVariables = None
    Edge.createdOmegaConstraints = None

    return model

nIterationsList = np.arange(0, 3, 0.5) 

#folderSaveModel = 'SASS_1_Thread'
folderSaveModel = 'SASS'

folderSaveModel += '/Costs/'
numRuns = 40
for nIter in nIterationsList:
    # The modelSSMS is rebuilt for each setting, except for simply changing the budget constraint
    modelSASS = None

    #Assure that the folder to save the results is created
    #Avoiding unnecessary decimal places in the folder name
    if nIter == floor(nIter):
        folderPath = pathlib.Path('./' + folderSaveModel + '/' + str(int(nIter)))
    else:
        folderPath = pathlib.Path('./' + folderSaveModel + '/' + str(nIter))
    folderPath.mkdir(parents=True, exist_ok=True)

    for i in range(numRuns):
        for budgetPower in range(30):
            fileName = folderPath / (str(i + 1) + '_' + str(budgetPower) + '.json')
            #fileName = folderPath / 'model.mps'

            budget_value = pow(2, budgetPower)
            
            #Discover the next filename
            if fileName.exists():
                continue

            elif modelSASS is None:
                modelSASS = buildGurobiModel(budget=budget_value, nIterations=nIter)
            #Updating the budget, since the model was not rebuilt
            else:
                budget_constraint = modelSASS.getConstrByName("budget_constraint")
                budget_constraint.rhs = budget_value
                modelSASS.update()

            try:
                modelSASS.Params.outputFlag = 0
                #modelSASS.Params.Threads = 1
                #modelSASS.Params.Presolve = 2
                #modelSASS = modelSASS.presolve()
                print("Starting to solve the model " + str(i + 1) + '_' + str(budgetPower))
                modelSASS.optimize()
                modelSASS.write(str(fileName.resolve()))
                print("Model solved")
                modelSASS.reset(clearall=1)

                #sleep(10)

            except gp.GurobiError as e:
                print("ERROR: " + str(e))
                break
