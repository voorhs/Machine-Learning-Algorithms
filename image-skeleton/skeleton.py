import matplotlib as mpl
import numpy as np
import cv2
from skan.csr import Skeleton
import skimage
from skan.csr import summarize
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import Counter
from segmentation import binarize, find_circles, add_notches


def skeletonize(img_dict):
    res = {}
    for key, img in img_dict.items():
        res[key] = skimage.morphology.skeletonize(img)
    return res

def draw_paths(skeleton: Skeleton):
    # pick fancy colors
    colors = mpl.colormaps['prism'](np.linspace(0, 1, skeleton.n_paths))
    colors = np.insert(colors[:, :-1], 0, values=[1,1,1], axis=0)

    # give color to each path
    img = colors[skeleton.path_label_image()]

    # compute length of each path
    path_lengths = skeleton.path_lengths()

    for i in range(skeleton.n_paths):
        # find coords of path center
        coords = skeleton.path_coordinates(i).mean(axis=0).astype(int)[::-1]
        
        # compute degrees of edge pixels
        start, end = skeleton.path(i)[[0, -1]]
        start_degree = skeleton.graph[start].count_nonzero()-1
        end_degree = skeleton.graph[end].count_nonzero()-1
        
        # normal path
        color = [0,1,0]
        if path_lengths[i] < 50:
            if start_degree * end_degree == 0:
                # unneccessary path
                color = [1,0,0]
            else:
                # ball-generated mess
                color = [0,0,1]

        # put label of path
        img = cv2.putText(
            img,
            text=str(i+1),
            org=coords,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA
        )

    return img


class Node:
    def __init__(self, *coords):
        self.coords = coords
        self.edge_inds = []
    
    def add_edge_ind(self, ind):
        self.edge_inds.append(ind)
    
    def reset_edge_inds(self):
        self.edge_inds = []
    
    def __repr__(self) -> str:
        c1, c2 = self.coords
        if hasattr(self, 'degree'):
            return f'Node(coords=({c1}, {c2}), edge_inds={self.edge_inds}, degree={self.degree})'
        return f'Node(coords=({c1}, {c2}), edge_inds={self.edge_inds})'

    def nodes_dict(skeleton):
        summary = summarize(skeleton)

        # choose source nodes
        src = ['node-id-src', 'coord-src-0', 'coord-src-1']
        names = ['ind', 'c1', 'c2']
        summary1 = summary[src].rename({key: val for key, val in zip(src, names)}, axis=1)

        # choose destination nodes
        dst = ['node-id-dst', 'coord-dst-0', 'coord-dst-1']
        summary2 = summary[dst].rename({key: val for key, val in zip(dst, names)}, axis=1)

        # concat and remove duplicates
        summary = pd.concat([summary1, summary2], ignore_index=True)
        summary = summary.drop_duplicates(subset='ind', ignore_index=True)

        # map node ind to Node object
        nodes = {row['ind']: Node(row['c1'], row['c2']) for _, row in summary.iterrows()}
        
        return nodes


class Edge:
    def __init__(self, type, path):
        self.type = type
        self.path = path
    
    def update_nodes(self, ind, node_dict):
        start = self.path[0]
        end = self.path[-1]

        node_dict[start].add_edge_ind(ind)
        node_dict[end].add_edge_ind(ind)
    
    def __repr__(self) -> str:
        return f'Edge(type={self.type})'

    def edge_list(skeleton, nodes):
        # compute length of each path
        path_lengths = skeleton.path_lengths()
        res = []

        for i in range(skeleton.n_paths):    
            # list of indices of path's pixels
            path = skeleton.path(i)
            start, end = path[[0, -1]]

            # compute degree of edge pixels
            start_degree = skeleton.graph[start].count_nonzero()-1
            end_degree = skeleton.graph[end].count_nonzero()-1
            
            # define type of path
            type = 'normal'
            if path_lengths[i] < 50:
                if start_degree * end_degree == 0:
                    type = 'trash'
                else:
                    type = 'medium'
            
            # update nodes' adjacent paths
            edge = Edge(type, path)
            edge.update_nodes(i, nodes)

            res.append(edge)
        
        return res


def solve(skeleton):
    nodes = Node.nodes_dict(skeleton)
    edges = Edge.edge_list(skeleton, nodes)

    # first pass: count degrees from all normal paths
    for node in nodes.values():
        # add attr
        node.degree = 0
        for edge_ind in node.edge_inds:
            node.degree += (edges[edge_ind].type == 'normal')

    # to store result
    res_dict = defaultdict(int)

    # for each node
    for i, node in enumerate(nodes.values()):
        # check if it is a mess of mediums
        node.degree = dfs_over_mediums(node, nodes, edges, i)
        
        # count only if it wasn't seen before (due to dfs)
        res_dict[node.degree] += not (node.subgraph_num < i)

    # convert dict to list
    res = []
    for i in range(max(res_dict.keys())):
        res.append(res_dict[i+1])
    
    return res

def dfs_over_mediums(node: Node, nodes, edges, subgraph_num):
    res = node.degree
    if hasattr(node, 'subgraph_num'):
        return res
    
    # add attr
    node.subgraph_num = subgraph_num

    # propagate to each direction
    for edge_ind in node.edge_inds:
        edge = edges[edge_ind]
        
        # search over 'medium' edges only
        if edge.type != 'medium':
            continue

        # mark as visited
        edge.type = 'medium_processed'

        # edge nodes
        start = nodes[edge.path[0]]
        end = nodes[edge.path[-1]]

        # choose direction to propagate to
        if node is start:
            res += dfs_over_mediums(end, nodes, edges, subgraph_num)
        else:
            res += dfs_over_mediums(start, nodes, edges, subgraph_num)
        
    return res

def pipeline(img_dict, k, print_results=False, plot_results=False):
    # find circles since they are verteces
    detected_circles_dict = find_circles(img_dict)

    # binarize
    binarized = binarize(img_dict)

    # build skeleton
    skeletonized = skeletonize(binarized)

    # to store results
    ans = defaultdict(list)

    # solve k times
    for i in range(k):
        # add notches with random angles
        skeletonized_with_notches = add_notches(skeletonized, detected_circles_dict)

        # for each image in collection
        for j, (key, sk) in enumerate(skeletonized_with_notches.items()):
            # build skeleton and solve problem
            skeleton = Skeleton(sk)
            ans[key].append(solve(skeleton))
    
    if print_results:
        for key, lst in ans.items():
            print(f'{key}:', *lst, sep='\t')

    # vote
    res = {}
    for key, lst in ans.items():
        # adjust lengths
        max_len = max(len(a) for a in lst)
        for a in lst:
            a.extend([0] * (max_len - len(a)))
        
        # for each degree
        tmp = []
        for i in range(max_len):
            tmp.append(Counter([a[i] for a in lst]).most_common(1)[0][0])
        res[key] = tmp
    
    if plot_results:
        n = len(skeletonized.values())
        fig, ax = plt.subplots(1, n, figsize=(2*n, 2))
        for j, (key, img) in enumerate(img_dict.items()):
            ax[j].imshow(img)
            ax[j].set_title(f'{key}: {res[key]}')
            ax[j].axis('off')
        
        plt.savefig('test.svg', bbox_inches='tight')
        plt.show()
    
    return res