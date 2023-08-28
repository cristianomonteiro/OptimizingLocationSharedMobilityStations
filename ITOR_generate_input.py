import networkx as nx
import psycopg2 as pg
import bz2
import pickle
import numpy as np
from math import isnan

from loadSplitEdges import loadMultiGraphEdgesSplit

class Edge:
    def __init__(self, G, u, v, idEdge, utilityValue, distanceFromSe, distanceCutOff):
        self.idEdge = idEdge
        self.utilityValue = utilityValue
        self.distanceFromSe = distanceFromSe

        self.edgesNearby = set(self.reachableEdges(G, u, distanceCutOff))
        self.edgesNearby.update(self.reachableEdges(G, v, distanceCutOff))

    #Return all id of edges adjacent to the current vertex
    def reachableEdges(self, G, u, cutoff):
        vertices = nx.single_source_dijkstra_path_length(G, u, cutoff=cutoff, weight='length').keys()

        edges = []
        for vertex in vertices:
            edges.extend([item[2]['idedge'] for item in G.edges(vertex, data=True) if item[2]['utilityvalue'] != 0])

        return edges

class Solution:
    def __init__(self, objectiveValue, allocatedEdges, forbiddenEdges, farestDistFromSe):
        self.objectiveValue = objectiveValue
        self.allocatedEdges = allocatedEdges
        self.forbiddenEdges = forbiddenEdges
        self.farestDistFromSe = farestDistFromSe

import networkx as nx

def convert_multigraph_to_int_keys(multigraph):
    # Create a mapping dictionary for nodes
    node_mapping = {node: i for i, node in enumerate(multigraph.nodes())}

    # Create a new graph with integer keys
    int_graph = nx.MultiGraph()
    int_graph.add_nodes_from(range(len(multigraph.nodes())))
    
    # Add edges to the new graph with integer keys
    key_edge_counter = 0
    for edge in multigraph.edges(keys=True, data=True):
        # The current key is already stored in the data dictionary with key 'idedge'
        src, tgt, key, data = edge
        data['idedgeint'] = key_edge_counter
        int_edge = (node_mapping[src], node_mapping[tgt], key_edge_counter)
        int_graph.add_edge(*int_edge, **data)

        key_edge_counter += 1

    return int_graph, node_mapping

def get_reached_edges_within_distance(graph, source_vertices, distance_threshold):
    reached_edges = set()
    
    for source_vertex in source_vertices:
        shortest_distances = nx.single_source_dijkstra_path_length(G=graph, source=source_vertex, weight='length', cutoff=distance_threshold)
        
        for end_vertex in shortest_distances.keys():
            edges = graph.edges(end_vertex, data=True)
            edges_key = [data['idedgeint'] for u, v, data in edges if is_useful(data['utilityvalue'])]
            reached_edges.update(edges_key)
    
    return reached_edges

# Write a file with all edges nearby given a distance cutoff
def write_file_reached_edges(multigraph_int_keys, precisionInput, distanceCutOff):
    with open('edges_nearby_' + str(precisionInput) + '_' + str(distanceCutOff) + '.txt', 'w') as file:
        break_line = ''
        string_write = ''
        string_max_size = 999999
        # Get the reached edges within the distance threshold
        for u, v, data in multigraph_int_keys.edges(data=True):
            vertices = [u, v]
            edge_key = data['idedgeint']

            if is_useful(data['utilityvalue']):
                reached_edges = get_reached_edges_within_distance(multigraph_int_keys, vertices, distanceCutOff)

                string_write += break_line + str(edge_key) + ' ' + ' '.join([str(edge) for edge in reached_edges])
                if len(string_write) >= string_max_size:
                    file.write(string_write)
                    string_write = ''
                
                # Breaks the line after the first line written
                break_line = '\n'

        # Write the remaining not written by the if above
        file.write(string_write)

def write_file(file_name, data_to_write):
    with open(file_name, 'w') as file:
        string_write = ''
        string_max_size = 999999
        # Writing first line, without the break line
        line_str = [str(data) for data in data_to_write[0]]
        file.write(' '.join(line_str))

        # Writing the other lines
        for line in data_to_write:
            line_str = [str(data) for data in line]
            string_write += '\n' + ' '.join(line_str)

            if len(string_write) >= string_max_size:
                file.write(string_write)
                string_write = ''

        # Write the remaining not written by the if above
        file.write(string_write)

def is_useful(value):
    if not isnan(value) and value > 0:
        return True
    else:
        return False

def build_data_to_write(multigraph_int_keys, considers_cost):
    data_to_write = []
    for u, v, data in multigraph_int_keys.edges(data=True):
        edge_key = data['idedgeint']
        edge_utility = data['utilityvalue']
        edge_cost = data['parkingexpenses']

        if is_useful(edge_utility):
            data_to_write.append((u, v, edge_key, edge_utility, edge_cost))

    cost_filename_suffix = ''
    if considers_cost:
        data_to_write.sort(key=lambda x: x[3]/x[4], reverse=True)
        cost_filename_suffix = '_costs'
    else:
        data_to_write.sort(key=lambda x: x[3], reverse=True)
    
    return data_to_write, cost_filename_suffix

def write_file_greedy_max_utility_order(multigraph_int_keys, precisionInput, distanceCutOff, considers_cost=False):
    data_to_write, cost_filename_suffix = build_data_to_write(multigraph_int_keys=multigraph_int_keys,
                                                              considers_cost=considers_cost)

    write_file(file_name='max_utility_order_' + str(precisionInput) + '_' + str(distanceCutOff) + cost_filename_suffix + '.txt', data_to_write=data_to_write)

def write_file_greedy_dijkstra(multigraph_int_keys, precisionInput, distance_used, considers_cost=False):
    data_to_write, cost_filename_suffix = build_data_to_write(multigraph_int_keys=multigraph_int_keys,
                                                              considers_cost=considers_cost)

    vertices_best_edge = [data_to_write[0][0], data_to_write[0][1]]
    shortest_distances_v1 = nx.single_source_dijkstra_path_length(G=multigraph_int_keys, source=vertices_best_edge[0], weight='length')
    shortest_distances_v2 = nx.single_source_dijkstra_path_length(G=multigraph_int_keys, source=vertices_best_edge[1], weight='length')

    # Sorting the data_to_write by shortest distance between u and v
    for i, (u, v, edge_key, edge_utility, edge_cost) in enumerate(data_to_write):
        shortest_distances = [shortest_distances_v1[u], shortest_distances_v1[v], shortest_distances_v2[u], shortest_distances_v2[v]]
        shortest_distances.sort()

        data_to_write[i] = (u, v, edge_key, edge_utility, edge_cost, shortest_distances[0])

    data_to_write.sort(key=lambda x: x[-1])
    write_file(file_name='dijkstra_order_' + str(precisionInput) + '_' + str(distance_used) + cost_filename_suffix + '.txt', data_to_write=data_to_write)

def generateInput(precisionInput=1, distanceCutOff=200):
    G = loadMultiGraphEdgesSplit(nIterations=precisionInput)

    multigraph_int_keys, node_mapping = convert_multigraph_to_int_keys(G)
    G = None

    # Save the multigraph to a text file with original and integer keys
    #nx.write_edgelist(multigraph_int_keys, "multigraph.txt", data=True, delimiter='\t', encoding='utf-8')

    write_file_reached_edges(multigraph_int_keys=multigraph_int_keys,
                             precisionInput=precisionInput,
                             distanceCutOff=distanceCutOff)
    
    write_file_greedy_max_utility_order(multigraph_int_keys=multigraph_int_keys,
                                        precisionInput=precisionInput,
                                        distanceCutOff=distanceCutOff)
    write_file_greedy_max_utility_order(multigraph_int_keys=multigraph_int_keys,
                                        precisionInput=precisionInput,
                                        distanceCutOff=distanceCutOff,
                                        considers_cost=True)
    
    write_file_greedy_dijkstra(multigraph_int_keys=multigraph_int_keys,
                               precisionInput=precisionInput,
                               distance_used=distanceCutOff)
    write_file_greedy_dijkstra(multigraph_int_keys=multigraph_int_keys,
                               precisionInput=precisionInput,
                               distance_used=distanceCutOff,
                               considers_cost=True)
    
    return

    pracaDaSe = 1407132173 #1837923352 #60641211      #26129121 is in Guarulhos       #1407132173 is in Sao Caetano do Sul
    distances = nx.single_source_dijkstra_path_length(G, pracaDaSe, weight='length')
    distances = sorted(distances.items(), key=lambda item: item[1])
    otherComponents = sorted(nx.connected_components(G), key=len, reverse=True)[1:]

    distanceToOtherComponent = float('inf')
    for component in otherComponents:
        for vertex in component:
            distances.append((vertex, distanceToOtherComponent))

    count = 0
    nextPrint = 1
    edgesSet = set()
    edges = []
    maxEdgeLength = 0
    for key, valueDistance in distances:
        count += 1
        if count == nextPrint:
            print(count)
            nextPrint *= 2

        for u, v, data in G.edges(key, data=True):
            if data['utilityvalue'] != 0 and data['idedge'] not in edgesSet:
                edgesSet.add(data['idedge'])
                edges.append(Edge(G, u, v, data['idedge'], data['utilityvalue'], valueDistance, distanceCutOff))

                if data['length'] > maxEdgeLength:
                    maxEdgeLength = data['length']

    return edges, distanceCutOff + maxEdgeLength

for precision in np.arange(0.5, 9, 1): 
    generateInput(precisionInput=precision)

    #data = generateInput(precisionInput=precision)

    #fileName = 'SASS_input_' + str(precision) + '.bz2'
    #fileName = 'SASS_Sao_Caetano_Sul_input_' + str(precision) + '.bz2'
    #filehandler = bz2.BZ2File(fileName, 'wb') 
    #pickle.dump(data, filehandler)
    #filehandler.close()
