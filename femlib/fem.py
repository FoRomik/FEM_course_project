#!/usr/bin/env python

import os, time
import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.tri as mpl_tri
import matplotlib.animation as animation

def importInitMesh(matfile = None):
        # import mesh data from matlab data file
	data = sio.loadmat(matfile)
	points = data['p']
	edges = data['e']
	triangles = data['t']
	diam = data['diam'][0][0]
	
	# convert to python format
	points = np.transpose(points)
	triangles = np.transpose(triangles[0:3,:])
	edges = np.transpose(edges[0:2])

	points = points.astype('float')
	triangles = triangles.astype('int')
	edges = edges.astype('int')
	diam = diam.astype('float')

	# fix array index numbering for python
	for i in range(len(triangles)):
		triangles[i,0] -= 1
		triangles[i,1] -= 1
		triangles[i,2] -= 1
        for i in range(len(edges)):
                tmp = edges[i]
                edges[i,0] -= 1
                edges[i,1] -=1 
                edges[i] = edges[i][np.argsort(edges[i])]

	return (points, triangles, edges, diam)	


class Mesh:
	def __init__(self, K0):
		self.Nodes = K0[0]
		self.Elements = K0[1]
		self.BoundaryEdges = K0[2]
		self.Diam = K0[3]
		self.DirEdges = None
		self.NeumannEdges = None
		
		self.NumNodes = len(self.Nodes)
		self.NumElements = len(self.Elements)
		self.NumBoundaryEdges = len(self.BoundaryEdges)
		self.NumDirEdges = 0
		self.NumNeumannEdges = 0
		self.NumDirNodes = 0
		self.EdgeMatrix = np.zeros([self.NumNodes, self.NumNodes], np.int32) 	
		
		for element in self.Elements:
			n0 = element[0]; n1 = element[1]; n2 = element[2]
			for pair in [(n0,n1), (n1,n2), (n2, n0)]:
				i = np.min(pair)
				j = np.max(pair)
				self.EdgeMatrix[i,j] = 1

	
	def setNeumannBoundary(self, NeumannEdges = None):
	        # to be defined by the problem
	        return NeumannEdges
	
	
	def setDirBoundary(self, DirEdges = None):
	        # to be defined by the problem
	        return DirEdges
	
	
	def loadBoundaries(self):
	        self.DirEdges = self.setDirBoundary()
	        self.NeumannEdges = self.setNeumannBoundary()
	        if not self.DirEdges is None: self.NumDirEdges = len(self.DirEdges)
	        if not self.NeumannEdges is None: self.NumNeumannEdges = len(self.NeumannEdges)
	        # free overall boundary attributes
	        del self.BoundaryEdges
	        del self.NumBoundaryEdges
	        
	
	def partitionNodes(self):
	        if not self.NumDirEdges:
	                self.DirNodes = np.array([])
	                self.FreeNodes = np.array(range(self.NumNodes), np.int32)
	                self.NumDirNodes = 0
	                self.NumFreeNodes = self.NumNodes
	                return
	                
	        DirNodes = set(self.DirEdges.flatten())
	        FreeNodes = set(self.Elements.flatten()); FreeNodes.difference_update(DirNodes)
	        self.NumDirNodes = len(DirNodes)
	        self.NumFreeNodes = len(FreeNodes)
	        self.FreeNodes = np.array([], np.int32)
	        self.DirNodes = np.array([], np.int32)
	        while len(DirNodes) > 0:
	                self.DirNodes = np.append(self.DirNodes, DirNodes.pop())
                while len(FreeNodes) > 0:
                        self.FreeNodes = np.append(self.FreeNodes, FreeNodes.pop())
	
	
	def refineMesh(self):
		current_Elements = self.Elements
                current_DirEdges = self.DirEdges
                current_NeumannEdges = self.NeumannEdges
		current_EdgeMatrix = self.EdgeMatrix
		current_NumNodes = self.NumNodes
		current_NumElements = self.NumElements
		current_NumDirEdges = self.NumDirEdges
		current_NumNeumannEdges = self.NumNeumannEdges

		for T in current_Elements:
			# divide element into 4 triangles
			n0 = T[0]; n1 = T[1]; n2 = T[2]			
			p0 = self.Nodes[T[0], :]; p1 = self.Nodes[T[1], :]; p2 = self.Nodes[T[2], :]
			p01 = 0.5*(p0+p1); p12 = 0.5*(p1+p2); p20 = 0.5*(p2+p0)		
			
			# update nodes
			self.Nodes = np.append(self.Nodes, np.array([p01]), axis = 0)
			self.Nodes = np.append(self.Nodes, np.array([p12]), axis = 0)
			self.Nodes = np.append(self.Nodes, np.array([p20]), axis = 0)		
			self.NumNodes = len(self.Nodes)
			n01 = self.NumNodes-3; n12 = self.NumNodes-2; n20 = self.NumNodes-1

			# update edgematrix
			current_EdgeMatrix[T[0], T[1]] = 0
			current_EdgeMatrix[T[1], T[2]] = 0
			current_EdgeMatrix[T[2], T[0]] = 0
			self.EdgeMatrix = np.zeros([self.NumNodes, self.NumNodes], np.int32)
			self.EdgeMatrix[0:current_NumNodes, 0:current_NumNodes] = current_EdgeMatrix
			for e in [[n0,n01], [n01,n20], [n20,n0], [n01,n1], [n1,n12], [n12,n01], [n20,n12], [n12,n2], [n2,n20]]:
				self.EdgeMatrix[e[0], e[1]] = 1
			
			# update Dirichlet edges
			if self.NumDirEdges !=0:
			        bset = None
        			for b_index, b in enumerate(current_DirEdges):
        			        if b.__contains__(n0) and b.__contains__(n1): 
        			                bset = np.append(b, n01)
        			                break
        			        if b.__contains__(n0) and b.__contains__(n2): 
        			                bset = np.append(b, n20)
        			                break
        			        if b.__contains__(n1) and b.__contains__(n2): 
        			                bset = np.append(b, n12)
        			                break
        			
        			if not bset is None:
        			        self.DirEdges = np.delete(self.DirEdges, b_index, 0)
                			e0 = np.array([bset[0], bset[2]]); e0 = e0[np.argsort(e0)]
                			e1 = np.array([bset[1], bset[2]]); e1 = e1[np.argsort(e1)]
                		        self.DirEdges = np.append(self.DirEdges, np.array([e0]), axis = 0)        
                		        self.DirEdges = np.append(self.DirEdges, np.array([e1]), axis = 0)        
                			self.NumDirEdges += 1  
			        
			# update Neumann edges
			if self.NumNeumannEdges !=0:
			        bset = None
        			for b_index, b in enumerate(current_NeumannEdges):
        			        if b.__contains__(n0) and b.__contains__(n1): 
        			                bset = np.append(b, n01)
        			                break
        			        if b.__contains__(n0) and b.__contains__(n2): 
        			                bset = np.append(b, n20)
        			                break
        			        if b.__contains__(n1) and b.__contains__(n2): 
        			                bset = np.append(b, n12)
        			                break
        		
        			if not bset is None:
        			        self.NeumannEdges = np.delete(self.NeumannEdges, b_index, 0)
                			e0 = np.array([bset[0], bset[2]]); e0 = e0[np.argsort(e0)]
                			e1 = np.array([bset[1], bset[2]]); e1 = e1[np.argsort(e1)]
                		        self.NeumannEdges = np.append(self.NeumannEdges, np.array([e0]), axis = 0)        
                		        self.NeumannEdges = np.append(self.NeumannEdges, np.array([e1]), axis = 0)        
                			self.NumNeumannEdges += 1       
			
			# update elements
			T_index = np.where(self.Elements == T)[0][0]
			self.Elements = np.delete(self.Elements, T_index, 0)
			for subElem in [[n0,n01,n20], [n01,n1,n12], [n01,n12,n20], [n20,n12,n2]]:
				self.Elements = np.append(self.Elements, np.array([subElem]), axis = 0)
			self.NumElements += 3

			# update counts
			current_NumNodes = self.NumNodes
			current_EdgeMatrix = self.EdgeMatrix
			current_DirEdges = self.DirEdges
			current_NeumannEdges = self.NeumannEdges
			current_NumElements = self.NumElements
                        
		


class ShapeFn:
	def __init__(self, Mesh = None, Element = None, NeumannEdge = None):	
		self.Element = Element
		self.Mesh = Mesh
                self.NeumannEdge = NeumannEdge
                

	def getElemArea(self):
		vertices = self.Mesh.Nodes[self.Element]
		T = np.append(np.array([[1],[1],[1]]), vertices, axis = 1)
		area = np.linalg.det(T)
		return area

	
	def getSupport(self, Node):
		NodeIndex = np.where(self.Mesh.Nodes == Node)[0][0]		
		elems = np.where(self.Mesh.Elements == NodeIndex)[0]
		support = []
		for elem in elems:
			support.append(self.Mesh.Elements[elem])
		support = np.array(support)
		return support	


	def getLocalShapeFn(self, Node = None):
		elem_area = self.__getElemArea()
		vertices = self.Mesh.Nodes[self.Element]
		T = np.array([ [1., 		Node[0], 		    Node[1] ],
			       [1.,		vertices[1,0],		vertices[1,1] ],
			       [1.,		vertices[2,0],		vertices[2,1] ] ])
		
		N = np.linalg.det(T) / elem_area
		return N


	def getStiffMatElement_P1(self):
		vertices = self.Mesh.Nodes[self.Element]		
		g = np.array([ [	1., 		1., 				1., ], 
			       [vertices[0,0],		vertices[1,0],		vertices[2,0] ],
			       [vertices[0,1],		vertices[1,1],		vertices[2,1] ] ])

		I = np.array([ [0., 0.],
			       [1., 0.],
			       [0., 1.] ])

		elem_area = self.getElemArea()
		g = np.mat(g); I = np.mat(I); 
		G = np.linalg.inv(g) * I	
		ret = 0.5* elem_area * G * np.transpose(G)
		#ret = np.array(ret)
		return ret


	def getMassMatElement_P1(self):
		x = np.array([ [2,1,1],
			       [1,2,1],
			       [1,1,2] ])
		elem_area = self.getElemArea()
		ret = 2. * elem_area * x / 24.
		return ret


	def getSrcTermElement_P1(self, f_vals):
		elem_area = self.getElemArea()	
		ret =  (2.*elem_area/18.) * np.sum(np.array(f_vals)) * np.ones(3)
		return ret

			       
        def getNeumannVecEdge_P1(self, g):
                p0 = self.Mesh.Nodes[self.NeumannEdge[0]]
                p1 = self.Mesh.Nodes[self.NeumannEdge[1]]
                L = np.sqrt(np.sum((p0-p1)**2.))
                ret = 0.5 * L * (g(p0)+g(p1)) * np.ones(2)
                return ret



class Assemb:
	def __init__(self, Mesh = None):
		self.Mesh = Mesh
		self.globalStiffMat = None
		self.globalMassMat = None
		self.RHSVec = None
	

	def getAllElementAreas(self):
	        shapefunc = ShapeFn(Mesh = self.Mesh)
	        areas = np.zeros(len(self.Mesh.Elements))
	        for i, T in enumerate(self.Mesh.Elements):
	                shapefunc.Element = T
	                areas[i] = shapefunc.getElemArea()
	        return areas
	

	def AssembStiffMat(self):
		self.globalStiffMat = np.zeros([self.Mesh.NumNodes, self.Mesh.NumNodes])	
		for T in self.Mesh.Elements:
			elemShapeFn = ShapeFn(Mesh = self.Mesh, Element = T)
			elemStiffMat = elemShapeFn.getStiffMatElement_P1()
			for i, nodeI in enumerate(T):
				for j, nodeJ in enumerate(T):
					self.globalStiffMat[nodeI, nodeJ] += elemStiffMat[i, j]
					

	def AssembMassMat(self):
	        self.globalMassMat = np.zeros([self.Mesh.NumNodes, self.Mesh.NumNodes])	
	        for T in self.Mesh.Elements:
			elemShapeFn = ShapeFn(Mesh = self.Mesh, Element = T)
			elemMassMat = elemShapeFn.getMassMatElement_P1()
			for i, nodeI in enumerate(T):
				for j, nodeJ in enumerate(T):
					self.globalMassMat[nodeI, nodeJ] += elemMassMat[i, j]
	

	        	        
class Plot:
	def __init__(self, Mesh, ax = None):
		self.Mesh = Mesh		
		self.ax = ax
	

	def plotBoundaries(self):		
	        # plot the interior
                for i in range(self.Mesh.NumNodes):
                        for j in range(self.Mesh.NumNodes):
                                if not self.Mesh.EdgeMatrix[i,j]: continue
                                src_node = self.Mesh.Nodes[i]
                                tar_node = self.Mesh.Nodes[j]
                                clr = 'black'            
                                self.ax.plot([src_node[0], tar_node[0]], [src_node[1], tar_node[1]], 
					     color = clr, linewidth = 1, marker = 'o', markersize = 5)
                # plot the Neumann boundaries
                if self.Mesh.NumNeumannEdges:
                        clr = 'red'
                        for b in self.Mesh.NeumannEdges:
                                src_node = self.Mesh.Nodes[b[0]]
                                tar_node = self.Mesh.Nodes[b[1]]
                                self.ax.plot([src_node[0], tar_node[0]], [src_node[1], tar_node[1]], 
					     color = clr, linewidth = 1, marker = 'o', markersize = 5)
		
		 # plot the Dirichlet boundaries
                if self.Mesh.NumDirEdges:
                        clr = 'blue'
                        for b in self.Mesh.DirEdges:
                                src_node = self.Mesh.Nodes[b[0]]
                                tar_node = self.Mesh.Nodes[b[1]]
                                self.ax.plot([src_node[0], tar_node[0]], [src_node[1], tar_node[1]], 
					     color = clr, linewidth = 1, marker = 'o', markersize = 5)
                                                
                
	def plotShapeFunc(self, Node):
		# plot the mesh at the base
		for i in range(self.Mesh.NumNodes):
			for j in range(i+1, self.Mesh.NumNodes):
				if not self.Mesh.EdgeMatrix[i,j]: continue
				src_node = self.Mesh.Nodes[i]
				tar_node = self.Mesh.Nodes[j]
				self.ax.plot([src_node[0], tar_node[0]], [src_node[1], tar_node[1]], [0.,0.], 
					     linewidth = 1, marker = 'o', markersize = 5, color = 'black')	

		# plot the shape function
		self.ax.plot([Node[0]], [Node[1]], [0.], color = 'blue', marker = 'o', markersize = 10)		
		support = ShapeFn(Mesh = self.Mesh).getSupport(Node = Node)
		NodeIndex = np.where(self.Mesh.Nodes == Node)[0][0]	
		for T in support:
			# color the support
			vertices = self.Mesh.Nodes[T]
			vertices = np.append(vertices, np.array([[0], [0], [0]]), axis = 1)
			x = vertices[:,0]; y = vertices[:,1]; z = vertices[:,2]		
			support_element = Poly3DCollection([zip(x,y,z)])			
			support_element.set_color('red'); support_element.set_edgecolor('black')
			self.ax.add_collection3d(support_element)
	
			# plot the hat 					
			for p in T:
				if NodeIndex == p: continue
				src_node = Node
				tar_node = self.Mesh.Nodes[p]				
				self.ax.plot([src_node[0], tar_node[0]], [src_node[1], tar_node[1]], [1., 0.], 
		 		              linewidth = 1, marker = 'o', markersize = 5, color = 'red') 


	def patternPlot(self, u, Component = 0, showGrid = False):
		if not len(u.shape) == 1: u = u[:,Component].flatten()		
		t = mpl_tri.Triangulation(self.Mesh.Nodes[:,0], self.Mesh.Nodes[:,1], self.Mesh.Elements)
		pattern = self.ax.tripcolor(t, u, shading='interp1', cmap=plt.cm.jet)
		cbar = plt.colorbar(pattern, ax = self.ax)
	        cbar.set_clim(vmin = u.min(), vmax = u.max())
	        cbar.draw_all()
		if showGrid: self.plotBoundaries()
		
		
	def patternAnimate(self, dataFileList, Component = 0, delay = None):
	        t = mpl_tri.Triangulation(self.Mesh.Nodes[:,0], self.Mesh.Nodes[:,1], self.Mesh.Elements)
                
                u0 = np.loadtxt(dataFileList[0])
                if not len(u0.shape) == 1: u0 = u0[:,Component].flatten()
	        u0 = u0[:self.Mesh.NumNodes]
	        
	        for i, dataFile in enumerate(dataFileList):
	             if i == 0:
	                pattern = self.ax.tripcolor(t, u0, shading = 'interp1', cmap = plt.cm.jet)
	                cbar = plt.colorbar(pattern, ax = self.ax)
	                cbar.set_clim(vmin = u0.min(), vmax = u0.max())
	                cbar.draw_all()
	                self.ax.set_title('Time = %d'% i)
	             else:
	                u = np.loadtxt(dataFile)
	                if not len(u.shape) == 1: u = u[:,Component].flatten()
	                u = u[:self.Mesh.NumNodes]
	                #pattern.set_array(u)
	                pattern = self.ax.tripcolor(t, u, shading = 'interp1', cmap = plt.cm.jet)
	                cbar.set_clim(vmin = u.min(), vmax = u.max())
	                cbar.draw_all()
	                self.ax.set_title('Time = %d' % i)
	             
	             self.ax.hold(False)   
	             if not delay: plt.waitforbuttonpress()
	             else: plt.pause(delay)
	             
                
                
                
		     
