import numpy as np
import heapq
from typing import Union

class Graph:
    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """ Unlike project 2, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or the path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None
        self.big_number = 1000

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """ Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. 
        Note that because we assume our input graph is undirected, `self.adj_mat` is symmetric. 
        Row i and column j represents the edge weight between vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        DONE: 
            This function does not return anything. Instead, store the adjacency matrix 
        representation of the minimum spanning tree of `self.adj_mat` in `self.mst`.
        We highly encourage the use of priority queues in your implementation. See the heapq
        module, particularly the `heapify`, `heappop`, and `heappush` functions.
        """
	
        # initialize priority queue with start node 0
        # entries in the priority queue will be a tuple of the form (node, predecessor in mst)
        n_nodes = self.adj_mat.shape[0]
        h = []
        heapq.heappush(h, (0, (0,0))) 
        all_nodes = set(list(range(n_nodes)))
        mst_set = set()
        # initialize blank mst adjacency matrix
        self.mst = np.zeros((n_nodes, n_nodes))

        # while not all nodes have been added to the mst:
        while len(mst_set) != n_nodes:
            # pop the outstanding node closest to the mst nodes
            u = heapq.heappop(h)	
            mst_set.add(u[1][0])
            self.mst[u[1][0],u[1][1]] = self.adj_mat[u[1][0],u[1][1]]
            self.mst[u[1][1],u[1][0]] = self.adj_mat[u[1][1],u[1][0]]	
            h = []
            # update all the distances of the remaining nodes
            for node in list(all_nodes.difference(mst_set)):
                distances_to_mst = self.adj_mat[node,].copy()
                distances_to_mst[distances_to_mst == 0] = self.big_number
                distances_to_mst[list(all_nodes.difference(mst_set))] = self.big_number
                predecessor = np.argmin(distances_to_mst)
                heapq.heappush(h, (distances_to_mst[predecessor],(node, predecessor)))




