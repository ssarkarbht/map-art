#import stuff
import torch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('figure', dpi=300)
from glob import glob
import sys
sys.path.insert(0,"/home/sourav/MapArt/edgeDet/HED/pytorch-hed/")
from edgeModule import *
sys.path.insert(0, "/home/sourav/MapArt/modules/")
from image_processing import *
from scipy.interpolate import RectBivariateSpline as RBS
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from matplotlib.colors import LogNorm
from shapely.geometry import LineString
import geopandas as gpd
import osmnx as ox

def Qfilter(img, Qfactor=500):
    ly,lx,nchan = img.shape
    if lx>=Qfactor and ly>=Qfactor: return True
    return False

#define the transformation functions
def rotation(cords,theta):
    x,y = cords
    xp = x*np.cos(theta)-y*np.sin(theta)
    yp = x*np.sin(theta)+y*np.cos(theta)
    return xp,yp

def translation(cords,shiftx, shifty):
    x,y = cords
    return x-shiftx, y-shifty

def scaling(cords,scale,xrange,yrange,xmin,ymin):
    x,y = cords
    xs = (x-xmin)/(scale*xrange)
    ys = (y-ymin)/(scale*yrange)
    return xs,ys

def scaleback(cords, scale, xrange, yrange, xmin, ymin):
    xp,yp = cords
    x = xp*scale*xrange + xmin
    y = yp*scale*yrange + ymin
    return x,y

class FitTransform:
    ''' This class initializes the map data from the geo package.
    Transforms the map coordinates and evaluates the likelihood
    function for the fit.
    '''
    def __init__(self, gpack, qfactor=500):
        print ("Will take while, go get a coffee!")
        #load the GeoPackagemap data
        self.nodes, self.edges, self.graph = load_map(gpack)
        #convert the nodes into arrays
        self.xmap = np.array(self.nodes['x'])
        self.ymap = np.array(self.nodes['y'])
        #store the absolute scaling variables
        self.xmin = min(self.xmap)
        self.xmax = max(self.xmap)
        self.xrange = self.xmax - self.xmin

        self.ymin = min(self.ymap)
        self.ymax = max(self.ymap)
        self.yrange = self.ymax - self.ymin

        print ("Done loading the map data.")
        
    def forward_transform(self, scale, shiftx, shifty, alpha):
        return rotation(translation(scaling((self.xmap, self.ymap),
            scale, self.xrange, self.yrange, self.xmin, self.ymin),
            shiftx, shifty), alpha)

    def back_transform(self, x,y, scale, shiftx, shifty, alpha):
        return rotation(translation(scaleback((x,y), scale,
            self.xrange, self.yrange, self.xmin, self.ymin),
            -1*shiftx, -1*shifty), -1*alpha)
    
    def fn_eval(self, x,y,fn):
        xidx = np.where(np.logical_or(x>1, x<0))[0]
        yidx = np.where(np.logical_or(y>1, y<0))[0]
        z = fn.ev(x,y)
        z[xidx] = 0.0
        z[yidx] = 0.0
        return z

def get_route(ids, mymap):
    #make the graph undirected for finding shortest distance
    G = mymap.graph.to_undirected()
    #get the osm ids for selected nodes
    subnodes = mymap.nodes.iloc[ids]
    sub_idlist = subnodes.index.to_list()
    #create a mas of ids to select on the intermediate nodes in the path
    osm_arr = np.array(mymap.nodes.index)
    mask_ids = []
    edge_ids = []
    for i in range(len(sub_idlist)-1):
        source = sub_idlist[i]
        target = sub_idlist[i+1]
        try:
            shortest_path = nx.shortest_path(G,
                                        source = source,
                                        target = target,
                                        weight = 'length')
        except:
            print ('no path found, moving on')
            continue
        # Step 3: Extract the selected edges' attributes forming the shortest path from the NetworkX graph

        for n in shortest_path:
            idx = np.where(osm_arr==n)[0]
            assert len(idx)==1, "More than one OSM id is ambiguious"
            mask_ids.append(idx[0])

        path_edges = [(shortest_path[j], shortest_path[j + 1]) for j in range(len(shortest_path) - 1)]
        edge_ids += path_edges

    return mask_ids, edge_ids

def get_osmid(mymap, node_mask, edge_ids, filename):
    node_ids = np.where(node_mask)[0]
    subnodes = mymap.nodes.iloc[node_ids]
    node_osm = subnodes.index.to_list()
    node_arr = np.array(node_osm)

    edge_arr = np.array(edge_ids)

    np.savez_compressed(filename, node_osm=node_arr, edge_osm=edge_arr)
    return None
