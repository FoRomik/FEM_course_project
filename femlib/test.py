#!/usr/bin/env python

import os

import numpy as np
import scipy as scp
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fem import *


testdata_dir = '/home/tanmoy/projects/FEM_course_project/code/testdata'


def testMesh():
        K0 = importInitMesh(matfile = os.path.join(testdata_dir, 'initmesh.mat'))
        m = Mesh(K0)
        
        m.NeumannEdges = m.BoundaryEdges 
        m.NeumannEdges = np.delete(m.NeumannEdges, 0, 0)
        m.NeumannEdges = np.delete(m.NeumannEdges, 14, 0)
        m.NumNeumannEdges = len(m.NeumannEdges)
       
        m.DirEdges = np.array([[0,8], [0,15]])
        m.NumDirEdges = len(m.DirEdges)
        
	m.partitionNodes()
	print m.DirNodes
	print m.FreeNodes
	
	ax1 = plt.subplot(121); ax2 = plt.subplot(122)
	p = Plot(Mesh = m)
	p.ax = ax1; p.plotBoundaries()
	
	m.refineMesh()
	m.partitionNodes()
	print m.DirNodes
	print m.FreeNodes
	p.ax = ax2; p.plotBoundaries()


def testShapeFnPlot():	
        K0 = importInitMesh(matfile = os.path.join(testdata_dir, 'initmesh.mat'))
        m = Mesh(K0); m.NeumannEdges = m.BoundaryEdges ;  m.NumNeumannEdges = len(m.NeumannEdges)
	fig = plt.figure()
	ax3d = Axes3D(fig)
	Node = m.Nodes[22]
	p = Plot(Mesh = m, ax = ax3d)
	p.plotShapeFunc(Node)  


def testPatternPlot():
	K0 = importInitMesh(matfile = os.path.join(testdata_dir, 'initmesh.mat'))
        m = Mesh(K0); m.NeumannEdges = m.BoundaryEdges ;  m.NumNeumannEdges = len(m.NeumannEdges)
	m.refineMesh();m.refineMesh()
	ax = plt.subplot(111)
	u = np.zeros(m.NumNodes)
	for i in range(m.NumNodes):
		u[i] = np.cos(m.Nodes[i,0] + m.Nodes[i,1])
	p = Plot(Mesh = m, ax = ax)
	p.patternPlot(u_Node = u)


def testElementMat():
	K0 = importInitMesh(matfile = os.path.join(testdata_dir, 'initmesh.mat'))
        m = Mesh(K0); m.NeumannEdges = m.BoundaryEdges ;  m.NumNeumannEdges = len(m.NeumannEdges)
	T = m.Elements[22]	
	s = ShapeFn(Mesh = m, Element = T)
	stiffmat = s.StiffMatElement_P1()
	massmat = s.MassMatElement_P1()		
	phi = []
	[phi.append(s.getLocalShapeFn(node)) for node in m.Nodes[T]]	

	print 'Element =', T
	print 'Nodes = ', m.Nodes[T]
	print 'phi = ', phi
	print 'stiffmat = ', stiffmat
	print 'massmat = ', massmat


def testAssembly():
	K0 = importInitMesh(matfile = os.path.join(testdata_dir, 'initmesh.mat'))
        m = Mesh(K0)
	
	m.NeumannEdges = m.BoundaryEdges 
        m.NeumannEdges = np.delete(m.NeumannEdges, 0, 0)
        m.NeumannEdges = np.delete(m.NeumannEdges, 14, 0)
        m.NumNeumannEdges = len(m.NeumannEdges)
       
        m.DirEdges = np.array([[0,8], [0,15]])
        m.NumDirEdges = len(m.DirEdges)
        
        m.partitionNodes()
        
        print '\n\n================CHECK MESH NODE SETUP ===================\n\n'
        print "NumDirNodes = ", m.NumDirNodes
	print "NumFreenodes = ", m.NumFreeNodes
	print "DirNodes = ", m.DirNodes
	print "FreeNodes = ", m.FreeNodes
	
	a = Assemb(Mesh = m)
	a.AssembStiffMat()
	a.AssembMassMat()
	
	print '\n\n================CHECK MATRICES ASSEMBLED OVER MESH===================\n\n'
	print "GlobalStiffMat =", a.globalStiffMat
	print "GlobalMassMat =", a.globalMassMat
	print "StiffMatsize = ", a.globalStiffMat.shape
	print "MassMatSize = ", a.globalMassMat.shape
       
        gdir = [lambda p: 1]
        gneumann = [lambda p: 0]
        pde = PDE(Mesh = m, StiffMat = a.globalStiffMat, MassMat = a.globalMassMat, 
                  gDir = gdir, gNeumann = gneumann)
        
        def srcFunc(p, u): return [10 * p[0] * p[1]]
        pde.srcFunc = srcFunc          
        allsrc = pde.getAllSrc()
        srcterm = pde.getSrcTerm()
        neumannbc = pde.getNeumannBC()
        
        print '\n\n================CHECK ASSEMBLED RHS VECTOR===================\n\n'
        print "AllSrc = ", allsrc
        print "Src Term = ", srcterm
        print "NeumannBC = ", neumannbc
        print "SrcTermSize = ", srcterm.shape
        print "NeumannBCSize = ", neumannbc.shape
        
        def getLHS(K,M): return K+M
               
        pde.getLHS = getLHS
        dirbc = pde.getDirBC()
        A,b = pde.load()
        
        print '\n\n================CHECK ASSEMBLED LHS MATRIX===================\n\n'
        print "DirBC = ", dirbc
        print "A = ", A
        print "b =", b
        print "DirBCSize = ", dirbc.shape
        print "ASize = ", A.shape
        print "bsize =", b.shape
        
        
        
def testPoisson():
        # init mesh
        K0 = importInitMesh(matfile = os.path.join(testdata_dir, 'testinitmesh.mat'))
        m = Mesh(K0)
        
        # set boundary conditions
        m.DirEdges = m.BoundaryEdges
        m.NumDirEdges = len(m.DirEdges)
        g = [lambda p: 0.]
        
        # get boundary partition
        m.partitionNodes()
        
        # assemble stiffness and mass matrices
        a = Assemb(Mesh = m)
	a.AssembStiffMat()
	a.AssembMassMat()
        K = a.globalStiffMat
        M = a.globalMassMat
        
        # set source function
        def fsrc(p,u): return [-4.]
        
        # assemble LHS of final lin. alg problem
        def getLHS(K,M): return -K
         
        # define the PDE
        pde = PDE(Mesh = m, StiffMat = a.globalStiffMat, MassMat = a.globalMassMat, gDir = g)
        pde.srcFunc = fsrc
        pde.getLHS = getLHS
        
        # generate final lin alg problem
        print 'Assembling Lin. Alg. problem...'
        A,b = pde.load()
        
        # solve the lin alg problem naively
        x = np.linalg.solve(A,b)
        
        # construct the full solution 
        u = pde.makeSol(x)

        # analytical exact solution
        u_ex = np.zeros([m.NumNodes,1])
        for i in range(m.NumNodes):
                p = m.Nodes[i]
                u_ex[i,0] = 1 - p[0]**2. - p[1]**2.
        
        # plot the solution and compare with analytical sol
	ax1 = plt.subplot(121); ax1.set_title('Exact')
	ax2 = plt.subplot(122); ax2.set_title('Finite Element')
	p = Plot(Mesh = m)
	p.ax = ax1; p.patternPlot(u_ex)
	p.ax = ax2; p.patternPlot(u)
	
	L2error = np.linalg.norm(u.flatten(order = 'F') -u_ex.flatten(order = 'F'), ord = 2)
	L1error = np.linalg.norm(u.flatten(order = 'F') -u_ex.flatten(order = 'F'), ord = np.inf)
	print "L2 Norm of Error = ", L2error
	print "L1 Norm of Error = ", L1error
	return (L2error, L1error)
	

def testErrorScaling(showPlots = False):
	pass


if __name__ == '__main__':
	#testMesh()
	#testShapeFnPlot()
	#testPatternPlot()
	#testShapeFnPlot()		
	#testAssembly()
	testPoisson()
	#testErrorScaling(showPlots = False)

plt.show()
