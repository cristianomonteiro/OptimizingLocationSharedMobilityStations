import pandas as pd
import networkx as nx
from heapq import heapify, heapreplace#, heappop, heappush
import psycopg2 as pg
from math import floor

def loadMultiDiGraph():
    params = {'host':'localhost', 'port':'5432', 'database':'afterqualifying', 'user':'cristiano', 'password':'cristiano'}
    conn = pg.connect(**params)

    sqlQuery = '''	select	EDGE.IDVERTEXORIG_FK,
                            EDGE.IDVERTEXDEST_FK,
                            EDGE.IDEDGE,
                            EDGE.LENGTH,
                            EDGE.UTILITYVALUE,
                            EDGE.PARKINGEXPENSES
                    from	STREETSEGMENT as EDGE
                    --where   EDGE.UTILITYVALUE > 0 and
                    --        EDGE.PARKINGEXPENSES is not null '''
    dataFrameEdges = pd.read_sql_query(sqlQuery, conn)
    conn.close()

    G = nx.MultiDiGraph()
    for row in dataFrameEdges.itertuples():
        dictRow = row._asdict()
        
        G.add_edge(dictRow['idvertexorig_fk'], dictRow['idvertexdest_fk'], key=str(dictRow['idedge']),
                    u=dictRow['idvertexorig_fk'], v=dictRow['idvertexdest_fk'],
                    idedge=str(dictRow['idedge']), length=dictRow['length'],
                    utilityvalue=dictRow['utilityvalue'], parkingexpenses=dictRow['parkingexpenses'])

    print(G.number_of_edges(), G.number_of_nodes())

    return G

def reBuildGraph(G, edgesHeap, firstSplit):
    for item in edgesHeap:
        (heapValue, u, v, idedge, lengthOriginal, utilityValue, parkingexpenses, numSplit) = item
        #The number of segments the edge must be split into is 1 less the value stored in the heap
        numSplit = numSplit - 1
        if numSplit >= firstSplit:
            lengthSplitted = lengthOriginal/numSplit
            vertexStart = u

            G.remove_edge(u, v, key=idedge)
            for i in range(numSplit - 1):
                vertexEnd = idedge + '_' + str(i + 1)
                G.add_edge(vertexStart, vertexEnd, key=vertexEnd, u=vertexStart, v=vertexEnd, idedge=vertexEnd,
                           length=lengthSplitted, utilityvalue=utilityValue, parkingexpenses=parkingexpenses)
                vertexStart = vertexEnd
            keyLast = idedge + '_' + str(numSplit)
            G.add_edge(vertexStart, v, key=keyLast, u=vertexStart, v=v, idedge=keyLast,
                       length=lengthSplitted, utilityvalue=utilityValue, parkingexpenses=parkingexpenses)

    return G

def loadMultiGraphEdgesSplit(nIterations=9, maxDistance=None):
    #It must be a MultiDiGraph because besides it has multiple edges between the same nodes, networkx does not assure the order of edges.
    #Using a directed graph, the start node of an edge will always be the start node, avoiding errors in the reBuildGraph function.
    G = loadMultiDiGraph()

    if nIterations > 0:
        firstSplit = 2
        #The value must be negative because the data structure is a min heap
        edgesHeap = [(-1*data['length'], data['u'], data['v'], data['idedge'], data['length'],
                      data['utilityvalue'], data['parkingexpenses'], firstSplit) for u, v, data in G.edges(data=True)]
        heapify(edgesHeap)
    
        for i in range(round(floor(len(edgesHeap) * nIterations))):
            #The value must be multiplied by -1 because the data structure is a min heap
            if maxDistance != None and -1 * edgesHeap[0][0] <= maxDistance:
                break
            
            #(heapValue, u, v, idedge, lengthOriginal, utilityValue, parkingExpenses, numSplit) = heappop(edgesHeap)
            (heapValue, u, v, idedge, lengthOriginal, utilityValue, parkingExpenses, numSplit) = edgesHeap[0]
            #The value must be negative because the data structure is a min heap
            heapValue = -1 * lengthOriginal/numSplit
            #The numSplit is prepared for the next time the edge may be splitted (numsplit + 1)
            heapreplace(edgesHeap, (heapValue, u, v, idedge, lengthOriginal, utilityValue, parkingExpenses, numSplit + 1))

        G = reBuildGraph(G, edgesHeap, firstSplit)
        
    print(G.number_of_edges(), G.number_of_nodes())

    return G.to_undirected()