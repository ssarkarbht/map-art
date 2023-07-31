#!/bin/python

#Import stuff
import numpy as np
import cv2 as cv
import sys
sys.path.insert(0,"/home/sourav/MapArt/edgeDet/HED/pytorch-hed/")
from edgeModule import *
from scipy.interpolate import RectBivariateSpline as RBS
from PIL import Image, ImageOps

from sklearn.neighbors import kneighbors_graph
import networkx as nx
from matplotlib.colors import LogNorm
from networkx.algorithms.components import connected_components

from shapely.geometry import LineString
import geopandas as gpd
import osmnx as ox
import networkx as nx
import potrace
from functools import reduce


class PreProcessing:
    def __init__(self, imgpath, model):
        self.img0 = cv.imread(imgpath)
        img1 = edgeConvert('bsds500', imgpath, 'test.png')
        self.img1 = np.array(img1)#[:,:,0]
        lx, ly = self.img1.shape
        x = np.arange(lx)
        y = np.arange(ly)
        self.fn_generator = RBS(x,y,self.img1, kx=1, ky=1)

    def sample_points(self):
        pass

def point_cloud_generator(imgpath, thr=100, nsample=1000, full_shape=False):
    ''' The function takes the input image path str
    and returns a point cloud data structure
    '''
    #convert the image into edge dtected greyscale image array
    imgEdge = np.array(edgeConvert('bsds500', imgpath))
    #get the image resolution for building image pixel coordinate
    ly, lx = imgEdge.shape
    #flatten the image
    imgEdge = imgEdge.reshape(-1)
    #rebuild the coordinates
    x = np.arange(lx)
    y = ly-np.arange(ly) #flip the the y coordinate (from image to plot)
    X,Y = np.meshgrid(x,y)
    X = X.reshape(-1)
    Y = Y.reshape(-1)

    #apply the threshold binary filter
    fidx = np.where(imgEdge>=thr)[0]
    #sample pixels from the filtered image
    if nsample is not None:
        ridx = np.random.choice(fidx, size=nsample)
        return X[ridx], Y[ridx], imgEdge[ridx]

    if full_shape:
        nidx = np.where(imgEdge<thr)[0]
        imgEdge[nidx] = 0
        return X, Y, imgEdge

    return X[fidx], Y[fidx], imgEdge[fidx]

def interpolator(imgpath, thr=100, scaleimg=True):

    #convert the image into edge dtected greyscale image array
    imgEdge = np.array(ImageOps.flip(edgeConvert('bsds500', imgpath)))
    #get the image resolution for building image pixel coordinate
    ly, lx = imgEdge.shape
    #create a threshold filter
    imgEdge = imgEdge.reshape(-1)
    filt_idx = np.where(imgEdge<thr)[0]
    imgEdge[filt_idx] = 0
    imgEdge = imgEdge.reshape(ly,lx)
    #rebuild the coordinates
    x = np.arange(lx)
    y = np.arange(ly)
    if scaleimg:
        x = x/float(max(lx,ly))
        y = y/float(max(lx,ly))
        imgEdge = imgEdge/255.
    X,Y = np.meshgrid(x,y)
    fn = RBS(x,y,imgEdge.T,kx=1,ky=1)
    return fn,lx,ly,imgEdge

def load_map(mapfile):
    gdf_nodes = gpd.read_file(mapfile, layer='nodes').set_index('osmid')
    gdf_edges = gpd.read_file(mapfile, layer='edges').set_index(['u', 'v', 'key'])
    assert gdf_nodes.index.is_unique and gdf_edges.index.is_unique, "Duplicate IDs found!"

    # convert the node/edge GeoDataFrames to a MultiDiGraph
    graph_attrs = {'crs': 'epsg:4326', 'simplified': True}
    G2 = ox.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs)
    return gdf_nodes, gdf_edges, G2


#====================== Stable classes =================
def Qfilter(img):
    #Apply image quality filter based on resolution
    ly, lx, nc = img.shape
    if lx>500 and ly>500: return True
    return False

class ImageProcessing:
    ''' This class takes an input image file and performs several
    processing required for tranlating to map data
    '''
    def __init__(self, imgpath):
        #get the original image processed through edge detection
        self.imgEdge = np.array(ImageOps.flip(edgeConvert('bsds500', imgpath)))
        #get the image resolution and normalization factor
        self.resy, self.resx = self.imgEdge.shape
        self.scale_norm = float(max(self.resx, self.resy))
        # get the 1D pixel cooridinates
        self.xpix = np.arange(self.resx)
        self.ypix = np.arange(self.resy)

    def threshold_filter(self, thr=100):
        #apply threshold filter on grey scale image
        self.imgEdge[self.imgEdge<thr] = 0
        return None

    def scale_image(self):
        #scale the image keeping the aspect ratio based on the resolution
        self.imgEdge = self.imgEdge/255.
        self.xpix = self.xpix / self.scale_norm
        self.ypix = self.ypix / self.scale_norm

        self.xcord, self.ycord = np.meshgrid(self.xpix, self.ypix)
        return self.xcord, self.ycord, self.imgEdge.T

    def binary_filter(self):
        imcopy = self.imgEdge.copy()
        imcopy[imcopy>0] = 1
        return imcopy

    def make_interpolator(self):
        self.interpolator = RBS(self.xpix, self.ypix, self.imgEdge.T, kx=1, ky=1)
        return None

    def scale_xy(self, x,y):
        return x/self.scale_norm, y/self.scale_norm

    def graph_filter(self, x,y,z, knn=5, plot=False):
        #select only the non-zero pixels
        filtidx = np.where(z!=0)[0]
        #filtidx = np.arange(len(x))
        #make a pixel and respective intensity arrays
        nodes = np.vstack((x[filtidx], y[filtidx], z[filtidx]))
        nodes = nodes.T
        #find the k-nearest neighbours
        A = kneighbors_graph(nodes, knn, mode='connectivity', include_self=False)
        # create node list with attributes for graph
        nodeAttr = []
        for i in filtidx:
            adict = {'x':x[i], 'y':y[i], 'z':z[i]}
            nodeAttr.append(adict)
        #node dictionary
        nodeList = dict(zip(np.arange(len(filtidx)), nodeAttr))
        #graph with the neighbourhood connections
        G = nx.from_scipy_sparse_matrix(A, create_using=nx.MultiGraph)
        #fill the node attributes
        nx.set_node_attributes(G, nodeList)
        #separate the disconnected components
        components = list(connected_components(G))
        #select the largest component
        largest_component = max(components, key=len)
        #create final filtered graph
        subgraph = G.subgraph(largest_component)

        if plot:
            subpos = { node: (data['x'], data['y']) for node, data in subgraph.nodes.data()}
            plt.figure()
            nx.draw_networkx(subgraph, pos=subpos, with_labels=False,
                node_size=0.5)
            plt.show()

        return filtidx[list(subgraph.nodes())]

    def run_knn_filter(self, knn=10):
        X,Y = self.xcord, self.ycord
        xflat = X.reshape(-1)
        yflat = Y.reshape(-1)
        idxs = self.graph_filter(xflat, yflat, self.imgEdge.reshape(-1), knn=knn)
        mask = np.zeros(len(xflat))
        mask[idxs] = 1
        mask = mask.reshape(self.imgEdge.shape)
        self.imgEdge[mask==0] = 0
        return None

#functions
def bezier(t, start_point, c1,c2,end_point):
    '''Compute Bezier curve for given values of t
    '''
    s1 = (1 - t)**3
    s2 = 3 * (1 - t) ** 2 * t
    s3 = 3 * (1 - t) * t ** 2
    s4 = t ** 3

    x = s1 * start_point[0] + s2 * c1[0] + s3 * c2[0] + s4 * end_point[0]
    y = s1 * start_point[1] + s2 * c1[1] + s3 * c2[1] + s4 * end_point[1]
    return x,y

def bezier_der(t, start_point, c1,c2,end_point):
    '''Compute derivative of Bezier curve for given
    values of t
    '''
    s1 = 3*(1-t)**2
    s2 = 6*(1-t)*t
    s3 = 3*t**2
    dx = s1 * (c1[0] - start_point[0]) + s2 * (c2[0] - c1[0]) + s3 * (end_point[0] - c2[0])
    dy = s1 * (c1[1] - start_point[1]) + s2 * (c2[1] - c1[1]) + s3 * (end_point[1] - c2[1])

    norm = np.sqrt((dx**2+dy**2))
    nx = dx/norm
    ny = dy/norm
    return nx,ny

def compute_linewidth(img):
    '''Compute average pixel width of the lines'''
    b8img = img.astype(np.uint8)
    lines = cv.HoughLinesP(b8img, 1, np.pi / 180, threshold=50, minLineLength=5, maxLineGap=2)

    # Calculate the average thickness
    total_thickness = 0
    num_lines = len(lines)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_thickness = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_thickness += line_thickness

    average_thickness = total_thickness / num_lines if num_lines > 0 else 0
    return average_thickness

def get_tinterval(p0, bsegment, pix=5):
    '''compute the t parameter interval of Bezier curve
    for given fixed pixel distance'''
    total_length = 0
    num_intervals = 1000
    tarr = np.linspace(0,1,num_intervals)
    x,y = bezier(tarr, p0, bsegment.c1, bsegment.c2, bsegment.end_point)
    dx = np.diff(x)
    dy = np.diff(y)
    total_length = np.sum(np.sqrt(dx**2 + dy**2))
    #print (total_length)
    num_points = int(np.ceil(total_length/pix))
    #print ("t interval is : ", 1./num_points)
    return np.linspace(0,1,num_points)
def compute_xsign(x,y,dx,dy,width, proc):
    points = np.linspace(0,width/2,50)

    neg_x = x - dy * points
    neg_y = y + dx * points

    tx, ty = proc.scale_xy(neg_x, neg_y)
    neg_sum = np.sum(proc.interpolator.ev(tx, ty))

    pos_x = x + dy * points
    pos_y = y - dx * points
    tx, ty = proc.scale_xy(pos_x, pos_y)
    pos_sum = np.sum(proc.interpolator.ev(tx, ty))
    if neg_sum > pos_sum:
        #print("DIR : -1")
        return -1
    elif neg_sum < pos_sum:
        #print("DIR : 1")
        return 1
    else:
        print ("WARNING: Couldn't decide direction")
        print ("Negative: ", neg_sum)
        print ("Positive: ", pos_sum)
        return None

def compute_corners(x,y,dx,dy,xsign,wdt,hgt, padding=0):
    x += -xsign * dy * padding
    y += xsign * dx * padding

    xv = x + xsign * dy * hgt
    yv = y + -xsign * dx * hgt

    xh = x + dx * wdt
    yh = y + dy * wdt

    xd = xh + xsign * dy * hgt
    yd = yh + -xsign * dx * hgt

    return x,y,xh,yh,xv,yv,xd,yd

class ImageTracing:
    '''This class performs the tracing of a line-drawing
    image and outputs ordered sequencing of the traced
    line.
    '''
    def __init__(self, img):
        self.img = img #binary image
    def img_thinner(self, ksize=(3,3)):
        img1 = self.img.copy()
        thin = np.zeros(self.img.shape)
        kernel = cv.getStructuringElement(cv.MORPH_CROSS,ksize)
        while (cv.countNonZero(img1)!=0):
            # Erosion
            erode = cv.erode(img1,kernel)
            # Opening on eroded image
            opening = cv.morphologyEx(erode,cv.MORPH_OPEN,kernel)
            # Subtract these two
            subset = erode - opening
            # Union of all previous sets
            thin = cv.bitwise_or(subset,thin)
            # Set the eroded image for next iteration
            img1 = erode.copy()
        return thin

    def img_thicker(self, thinimg, ksize=(3,3), iters=5):
        kernel = np.ones(ksize, np.uint8)
        dilimg = cv.dilate(thinimg, kernel, iterations=iters)
        return dilimg

    def apply_transform(self, ksize=(3,3), iters=5):
        self.timg = self.img_thicker(self.img_thinner(ksize), ksize, iters)
        return None

    def apply_tracing(self):
        #Run pyportrace tracing algorithm
        bmap = potrace.Bitmap(self.timg)
        self.path = bmap.trace()
        print (f"The image tracing produced {len(self.path.curves)} curves.")
        return None

    def rectangle_sq(self, proc, pix=5, width=20, pad=5):
        curve_num = []
        p1x = []
        p1y = []
        p2x = []
        p2y = []
        p3x = []
        p3y = []
        p4x = []
        p4y = []
        for i, curve in enumerate(self.path.curves):
            start = curve.start_point
            for j, seg in enumerate(curve.segments):
                if seg.is_corner: continue
                #compute t array
                tarr = get_tinterval(start, seg, pix=pix)
                #compute curves on the points
                x,y = bezier(tarr, start, seg.c1, seg.c2, seg.end_point)
                dx, dy = bezier_der(tarr, start, seg.c1, seg.c2, seg.end_point)
                if j==0:
                    xsign = compute_xsign(x[0],y[0],dx[0],dy[0],width, proc)
                if xsign is None: continue
                nx,ny,xh,yh,xv,yv,xd,yd = compute_corners(x,y,dx,dy,
                                            xsign,2*pix,width)

                curve_num += list(np.repeat(i, len(tarr)))
                p1x += list(nx)
                p1y += list(ny)

                p2x += list(xh)
                p2y += list(yh)
                p3x += list(xd)
                p3y += list(yd)

                p4x += list(xv)
                p4y += list(yv)

                start = seg.end_point
        # merge everything into a single array
        blockarr = np.zeros(len(curve_num), dtype=[('curve_num', int),
                    ('p1x', float), ('p1y', float),
                    ('p2x', float), ('p2y', float),
                    ('p3x', float), ('p3y', float),
                    ('p4x', float), ('p4y', float)])
        blockarr['curve_num'] = curve_num
        blockarr['p1x'] = p1x
        blockarr['p1y'] = p1y
        blockarr['p2x'] = p2x
        blockarr['p2y'] = p2y
        blockarr['p3x'] = p3x
        blockarr['p3y'] = p3y
        blockarr['p4x'] = p4x
        blockarr['p4y'] = p4y
        return blockarr

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.len = 0

    def __len__(self):
        return self.len

    def append(self, data):
        self.len += 1
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node

    def display(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

def dot_product(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]

def is_point_in_rect(point, rect):
    for i in range(4):
        p1 = rect[i]
        p2 = rect[(i + 1) % 4]
        p4 = rect[(i + 3) % 4]

        edge_vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        normal_probe = np.array([p4[0] - p1[0], p4[1] - p1[1]])

        normal_vector1 = np.array([edge_vector[1], -edge_vector[0]])
        normal_vector2 = np.array([-edge_vector[1], edge_vector[0]])

        dot1 = dot_product(normal_probe, normal_vector1)
        dot2 = dot_product(normal_probe, normal_vector2)
        if dot1<0: normal_vector = normal_vector1
        elif dot2<0: normal_vector = normal_vector2
        else: assert False, "check bug"

        point_vector = np.array([point[0] - p1[0], point[1] - p1[1]])

        if dot_product(point_vector, normal_vector) > 0:
            return False
    return True

def points_in_rect(rect, points):
    inidxs = []
    big_minx = reduce(np.minimum,
                     (rect['p1x'],rect['p2x'], rect['p3x'], rect['p4x']))
    big_maxx = reduce(np.maximum,
                     (rect['p1x'],rect['p2x'], rect['p3x'], rect['p4x']))
    big_miny = reduce(np.minimum,
                     (rect['p1y'],rect['p2y'], rect['p3y'], rect['p4y']))
    big_maxy = reduce(np.maximum,
                     (rect['p1y'],rect['p2y'], rect['p3y'], rect['p4y']))

    #big_minx = np.minimum(rect['p1x'],rect['p2x'], rect['p3x'], rect['p4x'])
    #big_maxx = np.maximum(rect['p1x'],rect['p2x'], rect['p3x'], rect['p4x'])
    #big_miny = np.minimum(rect['p1y'],rect['p2y'], rect['p3y'], rect['p4y'])
    #big_maxy = np.maximum(rect['p1y'],rect['p2y'], rect['p3y'], rect['p4y'])

    xrange = np.where((points['x']>=big_minx)&(points['x']<=big_maxx))[0]
    yrange = np.where((points['y']>=big_miny)&(points['y']<=big_maxy))[0]

    filtidx = np.intersect1d(xrange, yrange, assume_unique=True)
    if len(filtidx)==0: return None

    for idx in filtidx:
        p = points[idx]
        recpoints = [(rect['p1x'], rect['p1y']),
                    (rect['p2x'], rect['p2y']),
                    (rect['p3x'], rect['p3y']),
                    (rect['p4x'], rect['p4y'])]

        inside = is_point_in_rect((p['x'], p['y']), recpoints)
        if inside: inidxs.append(idx)

    if len(inidxs)==0: return None
    return inidxs

def distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


class Sequencer:
    '''Class to order the given set of points so that
    the consecutive connections create the underlying
    line drawing'''
    def __init__(self, points, rects):
        self.points = points
        self.boxes = rects
        #mask to track which points are sequenced
        self.mask = np.repeat(False, len(points))
        #initialize empty sequence dictionary
        self.seq_dict = {}

    def is_sequenced(self, idx):
        '''Check if the point is already sequenced'''
        #print (self.mask)
        #print (idx)
        return self.mask[idx]

    def order_multiple(self, idlist, rect):
        '''In case there are multiple pouints in the box,
        order the points based on the distance from the
        first corner of the box.'''
        dlist = []
        #loop over the points
        for idx in idlist:
            p1 = self.points[idx]
            p2 = (rect['p1x'], rect['p1y'])
            dist = distance(p1,p2)
            dlist.append(dist)
        #get the sorted ids
        sort_order = np.argsort(np.array(dlist))
        #print (idlist)
        #print (sort_order)
        return list(np.array(idlist)[sort_order])

    def start_sequence(self, tolerance=20):
        '''The parent function that executes the
        sequencing.
        Args: Tolerance (int): number of consecutive
        empty boxes (no points) after which a new
        sequence is started.'''
        #initialize stuff
        #blank linked list
        llist = LinkedList()
        #running curve number
        current_curve = 0
        #running segment tracker
        seg_iter = 0
        #initialize the previous and current seq. points
        prev_node = None
        curr_node = None

        #number of consecutive boxes without any points
        miss_tol = 0

        #start the box loop
        for i, box in enumerate(self.boxes):
            #=========== section for checking new sequence
            #if the current box is part of the next
            #curve, start a new linked list
            if box['curve_num']!=current_curve:
                #store the last sequenced segment
                #and update the segment tracker
                #print (f"Moving to curve: {box['curve_num']}")
                if len(llist)>1:
                    self.seq_dict[seg_iter] = llist
                    seg_iter += 1
                #start a new sequence and update the curve number
                llist = LinkedList()
                current_curve = box['curve_num']
                miss_tol = 0

            if (box['curve_num']==current_curve and
               miss_tol>=tolerance):
                if len(llist)>1:
                    self.seq_dict[seg_iter] = llist
                    seg_iter += 1
                #start a new sequence with head being the
                #previous node
                llist = LinkedList()
                llist.append(prev_node)

            #============ Section for treating points inside the box
            #get the point idxs inside current box
            in_box = points_in_rect(box, self.points)

            #deal with the returned points
            if in_box is None:#empty box
                miss_tol += 1
                continue

            elif len(in_box)==1:#only one point
                if self.is_sequenced(in_box[0]):
                    miss_tol += 1
                    prev_node = in_box[0]
                    continue
                else:
                    llist.append(in_box[0])
                    self.mask[in_box[0]] = True
                    miss_tol = 0

            else:#multiple points in the box
                #order the points
                sortedid = self.order_multiple(in_box,
                                              box)
                #remove the points that are already sequenced
                update_box = []
                for idx in sortedid:
                    if self.is_sequenced(idx): continue
                    update_box.append(idx)
                #if no point survives,
                if len(update_box)==0:
                    miss_tol += 1
                    prev_node = sortedid[-1]

                else:
                    for nidx in update_box:
                        llist.append(nidx)
                        self.mask[nidx] = True
                    miss_tol = 0

