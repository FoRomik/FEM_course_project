#!/usr/bin/env python

import os
import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.tri as mpl_tri


def importInitMesh(matfile = None):
        # import mesh data from matlab data file
	data = sio.loadmat(matfile)
	points = data['p']
	triangles = data['t']
	
	# convert to python format
	points = np.transpose(points)
	triangles = np.transpose(triangles[0:3,:])

	points = points.astype('float')
	triangles = triangles.astype('int')

	# fix array index numbering for python
	for i in range(len(triangles)):
		triangles[i,0] -=1
		triangles[i,1] -=1
		triangles[i,2] -=1

	return (points, triangles)	


class Mesh:
	def __init__(self, K0):
		self.Nodes = K0[0]
		self.Elements = K0[1]
		self.NumNodes = len(self.Nodes)
		self.NumElements = len(self.Elements)
		self.EdgeMatrix = np.zeros([self.NumNodes, self.NumNodes], np.int32) 	
		
		for element in self.Elements:
			n0 = element[0]; n1 = element[1]; n2 = element[2]
			for pair in [(n0,n1), (n1,n2), (n2, n0)]:
				i = np.min(pair)
				j = np.max(pair)
				self.EdgeMatrix[i,j] = 1


	def refineMesh(self):
		current_Elements = self.Elements
		current_EdgeMatrix = self.EdgeMatrix
		current_NumNodes = self.NumNodes
		current_NumElements = self.NumElements

		for T in current_Elements:
			# divide element into 4 triangles			
			p0 = self.Nodes[T[0], :]; p1 = self.Nodes[T[1], :]; p2 = self.Nodes[T[2], :]
			p01 = 0.5*(p0+p1); p12 = 0.5*(p1+p2); p20 = 0.5*(p2+p0)		
			
			# update nodes
			self.Nodes = np.append(self.Nodes, np.array([p01]), axis = 0)
			self.Nodes = np.append(self.Nodes, np.array([p12]), axis = 0)
			self.Nodes = np.append(self.Nodes, np.array([p20]), axis = 0)		
			self.NumNodes = len(self.Nodes)

			# update edges
			current_EdgeMatrix[T[0], T[1]] = 0
			current_EdgeMatrix[T[1], T[2]] = 0
			current_EdgeMatrix[T[2], T[0]] = 0
			self.EdgeMatrix = np.zeros([self.NumNodes, self.NumNodes], np.int32)
			self.EdgeMatrix[0:current_NumNodes, 0:current_NumNodes] = current_EdgeMatrix

			n0 = T[0]; n1 = T[1]; n2 = T[2]
			n01 = self.NumNodes-3; n12 = self.NumNodes-2; n20 = self.NumNodes-1
			for e in [[n0,n01], [n01,n20], [n20,n0], [n01,n1], [n1,n12], [n12,n01], [n20,n12], [n12,n2], [n2,n20]]:
				self.EdgeMatrix[e[0], e[1]] = 1
			
			# update elements
			T_index = np.where(self.Elements == T)[0][0]
			self.Elements = np.delete(self.Elements, T_index, 0)
			for subElem in [[n0,n01,n20], [n01,n1,n12], [n01,n12,n20], [n20,n12,n2]]:
				self.Elements = np.append(self.Elements, np.array([subElem]), axis = 0)
			self.NumElements += (4-1)

			# update counts
			current_NumNodes = self.NumNodes
			current_EdgeMatrix = self.EdgeMatrix
			current_NumElements = self.NumElements

		


class ShapeFn:
	def __init__(self, Mesh = None, Element = None):	
		self.Element = Element
		self.Mesh = Mesh

	def __getElemArea(self):
		vertices = self.Mesh.Nodes[self.Element]
		T = np.append(vertices, np.array([[1],[1],[1]]), axis = 1)
		area = np.linalg.det(T)
		return area
	
	def getSupport(self, Node):
		NodeIndex = np.where(self.Mesh.Nodes == Node)[0][0]		
		elems = np.where(self.Mesh.Elements == NodeIndex)[0]
		support = []
		for elem in elems:
			support.append(Mesh.Elements[elem])
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

	def StiffMatElement_P1(self):
		vertices = self.Mesh.Nodes[self.Element]		
		g = np.array([ [	1., 		1., 				1., ], 
			       [vertices[0,0],		vertices[1,0],		vertices[2,0] ],
			       [vertices[0,1],		vertices[1,1],		vertices[2,1] ] ])

		I = np.array([ [0., 0.],
			       [1., 0.],
			       [0., 1.] ])

		elem_area = self.__getElemArea()
		g = np.mat(g); I = np.mat(I); 
		G = np.linalg.inv(g) * I	
		ret = 0.5* elem_area * G * np.transpose(G)
		#ret = np.array(ret)
		return ret

	def MassMatElement_P1(self):
		x = np.array([ [2,1,1],
			       [1,2,1],
			       [1,1,2] ])
		elem_area = self.__getElemArea()
		ret = 0.5 * elem_area * x / 24.
		return ret

	def fMatElement_P1(self, f):
		elem_area = self.__getElemArea()
		f_val = []
		[f_val.append(f(node)) for node in self.Mesh.Nodes[self.Element]]		
		ret =  (2*elem_area/18.) * np.sum(np.array(f_val)) * np.ones(3)
		return ret
		
        def fMatElement_nonlin(self, f, u_Node):
                # here f is a function of u
                elem_area = self.__getElemArea()
                f_val = []
                for i in range(3): f_val.append(f(u_Node[self.Element[i]]))
                ret =  (2*elem_area/18.) * np.sum(np.array(f_val)) * np.ones(3)
		return ret
                


class Assemb:
	def __init__(self, Mesh = None, f = None):
		self.Mesh = Mesh
		self.f = f
	
	def AssembMat_naive(self):
		self.globalStiffMat = np.zeros([self.Mesh.NumNodes, self.Mesh.NumNodes])
		self.globalMassMat = np.zeros([self.Mesh.NumNodes, self.Mesh.NumNodes])		
		self.globalfMat = np.zeros([self.Mesh.NumNodes,1])	
		for T in self.Mesh.Elements:
			elemShapeFn = ShapeFn(Mesh = self.Mesh, Element = T)
			elemStiffMat = elemShapeFn.StiffMatElement_P1()
			elemMassMat = elemShapeFn.MassMatElement_P1()
			elemfMat = elemShapeFn.fMatElement_P1(f = self.f)
			
			for i, nodeI in enumerate(T):
				self.globalfMat[nodeI] += elemfMat[i]
				for j, nodeJ in enumerate(T):
					self.globalStiffMat[nodeI, nodeJ] += elemStiffMat[i, j]
					self.globalMassMat[nodeI, nodeJ] += elemMassMat[i, j]



	def AssembMat_fast(self):
		#TODO: sparsify and vectorize
		pass		






class Plot:
	def __init__(self, Mesh, ax = None):
		self.Mesh = Mesh		
		self.ax = ax
	
	def plotMesh(self):			
		for i in range(self.Mesh.NumNodes):
			for j in range(self.Mesh.NumNodes):
				if not self.Mesh.EdgeMatrix[i,j]: continue
				src_node = self.Mesh.Nodes[i]
				tar_node = self.Mesh.Nodes[j]
				self.ax.plot([src_node[0], tar_node[0]], [src_node[1], tar_node[1]], 
					     linewidth = 1, marker = 'o', markersize = 5, color = 'black')

	def plotShapeFunc(self, Node):
		# plot the mesh at the base
		for i in range(self.Mesh.NumNodes):
			for j in range(self.Mesh.NumNodes):
				if not self.Mesh.EdgeMatrix[i,j]: continue
				src_node = self.Mesh.Nodes[i]
				tar_node = self.Mesh.Nodes[j]
				self.ax.plot([src_node[0], tar_node[0]], [src_node[1], tar_node[1]], [0.,0.], 
					     linewidth = 1, marker = 'o', markersize = 5, color = 'black')	

		# plot the shape function
		self.ax.plot([Node[0]], [Node[1]], [0.], color = 'blue', marker = 'o', markersize = 10)		
		support = ShapeFn().getSupport(Mesh = self.Mesh, Node = Node)
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


	def patternPlot(self, u_Node, showGrid = False):
		if u_Node.shape[0] == self.Mesh.NumNodes: u_Node = u_Node.flatten()		
		t = mpl_tri.Triangulation(self.Mesh.Nodes[:,0], self.Mesh.Nodes[:,1], self.Mesh.Elements)
		self.ax.tripcolor(t, u_Node, shading='interp1', cmap=plt.cm.rainbow)
		if showGrid: self.plotMesh()
