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
        
	ax1 = plt.subplot(121); ax2 = plt.subplot(122)
	p = Plot(Mesh = m)
	p.ax = ax1; p.plotBoundaries()
	
	m.refineMesh()
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
        m = Mesh(K0)
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


def testNaiveAssembly():
	K0 = importInitMesh(matfile = os.path.join(testdata_dir, 'initmesh.mat'))
        m = Mesh(K0)
	f = lambda node : node[0] + node[1]
	a = Assemb(Mesh = m)
	a.AssembMat_naive()
	a.AssembRHSVec(f =f)
	print a.globalStiffMat
	print a.globalMassMat
	print a.globalfMat

        
def testPoisson():
        K0 = importInitMesh(matfile = os.path.join(testdata_dir, 'initmesh.mat'))
        m = Mesh(K0)
	f = lambda node: 1.0
	a = Assemb(Mesh = m)
	a.AssembMat_naive()
	a.AssembRHSVec(f = f)
	K = a.globalStiffMat
	F = a.globalfMat
	u = np.linalg.solve(K,F)

	ax = plt.subplot(111)
	p = Plot(Mesh = m, ax = ax)
	p.patternPlot(u_Node = u, showGrid = False)
	

def testErrorScaling(showPlots = False):
	matfile_fmt = os.path.join(testdata_dir, 'testmesh%d.mat')
	outfile_fmt = os.path.join(testdata_dir, 'testfem%d.mat')
	diams = np.array([0.1, 0.08, 0.04, 0.06, 0.02])
	err = np.zeros(len(diams))	
		
	 # test for the function u(x,y) = arctan(y/x) 
        # which has du/dr = 0 at the boundaries
        			
	f_exact = lambda p : p[0]**2.+p[1]**2. - 2*np.sqrt(p[0]**2.+p[1]**2.)		
	f = lambda p : 4 - 2./np.sqrt(p[0]**2.+p[1]**2.)

	for i, h in enumerate(diams):
		matfile = matfile_fmt % i
		
		# fem solution	
		K = importInitMesh(matfile)
		mesh = Mesh(K);
		a = Assemb(Mesh = mesh)
		a.AssembRHSVec(f=f)
		a.AssembMat_naive()
		K = a.globalStiffMat
		F = a.globalfMat
		u = np.linalg.solve(K,F)
                np.savetxt(outfile_fmt % i, u)
                
                print mesh.NumNodes, mesh.Elements.shape, u.shape

		# exact solution		
		u_exact = np.zeros(mesh.NumNodes)
		for j in range(mesh.NumNodes):
			u_exact[j] = f_exact(mesh.Nodes[i])
		
		if showPlots:
	        	ax1 = plt.subplot(121) ; ax2 = plt.subplot(122)
		        p = Plot(Mesh = mesh)
        		p.ax = ax1; p.patternPlot(u_exact); ax1.set_title('Exact')
        		p.ax = ax2; p.patternPlot(u); ax2.set_title('FEM')
        		plt.show()
	
		# error
		err[i] = np.linalg.norm(u_exact - u)
	        print '\nFEM: Mesh_diameter = %g, Error = %g\n' % (h, err[i])
	
	plt.scatter(1./diams, err, marker = 'o', color = 'red', label = r'$L^2$' + ' norm of error')
	plt.xscale('log'); plt.yscale('log')
	plt.xlabel(r'$1/h$', fontsize = 'large'); plt.ylabel(r'$|u-u_{exact}|$', fontsize = 'large')
	out = stats.linregress(np.log10(diams), np.log10(err))
	slope = out[0]
	
	print 'Slope = ', slope


if __name__ == '__main__':
	#testMesh()
	#testShapeFnPlot()
	#testPatternPlot()
	#testShapeFnPlot()		
	#testNaiveAssembly()
	#testPoisson()
	#testErrorScaling(showPlots = False)
	#testbkEuler()
plt.show()
