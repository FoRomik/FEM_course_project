#!/usr/bin/env python

import os

import numpy as np
import scipy as scp
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fem import *
from pde import *


testdata_dir = '/home/tanmoy/projects/FEM_course_project/code/testdata'


def testMesh():
        K0 = importInitMesh(matfile = os.path.join(testdata_dir, 'testmesh0.mat'))
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
        K0 = importInitMesh(matfile = os.path.join(testdata_dir, 'testmesh0.mat'))
        m = Mesh(K0); m.NeumannEdges = m.BoundaryEdges ;  m.NumNeumannEdges = len(m.NeumannEdges)
	fig = plt.figure()
	ax3d = Axes3D(fig)
	Node = m.Nodes[22]
	p = Plot(Mesh = m, ax = ax3d)
	p.plotShapeFunc(Node)  


def testPatternPlot():
	K0 = importInitMesh(matfile = os.path.join(testdata_dir, 'testmesh0.mat'))
        m = Mesh(K0); m.NeumannEdges = m.BoundaryEdges ;  m.NumNeumannEdges = len(m.NeumannEdges)
	m.refineMesh();m.refineMesh()
	ax = plt.subplot(111)
	u = np.zeros(m.NumNodes)
	for i in range(m.NumNodes):
		u[i] = np.cos(m.Nodes[i,0] + m.Nodes[i,1])
	p = Plot(Mesh = m, ax = ax)
	p.patternPlot(u_Node = u)


def testElementMat():
	K0 = importInitMesh(matfile = os.path.join(testdata_dir, 'testmesh0.mat'))
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
	K0 = importInitMesh(matfile = os.path.join(testdata_dir, 'testmesh0.mat'))
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
       
        pde = Elliptic(Mesh = m, StiffMat = a.globalStiffMat, MassMat = a.globalMassMat)
        
        # set source term and boundary conditions
        def fsrc(Mesh = m): return [lambda p: 10 * p[0] * p[1]]
        def gdir(Mesh = m): return [lambda p: 1]
        def gneumann(Mesh = m): return [lambda p: 0]
        pde.setSrcFunc = fsrc     
        pde.setDirFunc = gdir
        pde.setNeumannFunc = gneumann     
        
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
        pde.AssembLHS = getLHS
        dirbc = pde.getDirBC()
        A,b = pde.AssemPDE()
        
        print '\n\n================CHECK ASSEMBLED LHS MATRIX===================\n\n'
        print "DirBC = ", dirbc
        print "A = ", A
        print "b =", b[0]+b[1]-b[2]
        print "DirBCSize = ", dirbc.shape
        print "ASize = ", A.shape
        print "bsize =", b.shape
        
        
        
def testPoisson(meshmatfile = 'testmesh0.mat', showPlot = True):
        # init mesh
        K0 = importInitMesh(matfile = os.path.join(testdata_dir, meshmatfile))
        m = Mesh(K0)
        
        
        # linear uncoupled elliptic test problems
        #1) u1(x,y) = 1 -x - y
        #2) u2(x,y) = (1 - x^2 - y^2)/4
        #3) u3(x,y) = (1 - x^4 - y^4)/12
        
        f = [lambda p: (1 - p[0] - p[1]),  
             lambda p: (1. - p[0]**2. - p[1]**2.)/4.,
             lambda p: (1. - p[0]**3. - p[1]**3.)/6.,
             lambda p: (1. - p[0]**4. - p[1]**4.)/12.,
             lambda p: (1. - p[0]**5. - p[1]**5.)/20.]
             
        
        NComponents = len(f)
        
        # set boundary conditions
        m.DirEdges = m.BoundaryEdges
        m.NumDirEdges = len(m.DirEdges)
        def gdir(Mesh = m): return f
        
        # get boundary partition
        m.partitionNodes()
        
        # assemble stiffness and mass matrices
        a = Assemb(Mesh = m)
	a.AssembStiffMat()
	a.AssembMassMat()
        K = a.globalStiffMat
        M = a.globalMassMat
        
        # set source function
        def fsrc(Mesh = m): 
                return [lambda p: 0., 
                        lambda p: 1.,
                        lambda p: p[0] + p[1],
                        lambda p: p[0]**2. + p[1]**2.,
                        lambda p: p[0]**3. + p[1]**3.]
                               
        # assemble LHS of final lin. alg problem
        def getLHS(K,M):
                zero = np.zeros([K.shape[0], K.shape[1]])
                lhs = []
                for i in range(NComponents):
                        row = []
                        for j in range(NComponents):
                                if i == j: col = K
                                else: col = zero
                                row.append(col)
                        lhs.append(row)
                        
                lhs = np.array(np.bmat(lhs))
          
                return lhs
         
        # define the PDE
        pde = Elliptic(NComponents = NComponents, Mesh = m, StiffMat = a.globalStiffMat, MassMat = a.globalMassMat)
        pde.setDirFunc = gdir
        pde.setSrcFunc = fsrc
        pde.AssembLHS = getLHS
        
        # generate final lin alg problem
        print 'Assembling Lin. Alg. problem...'
        A,rhs = pde.AssemPDE()
        b = rhs[0] + rhs[1] - rhs[2]
        
        # solve the lin alg problem naively
        x = np.linalg.solve(A,b)
        
        # construct the full solution 
        u = pde.makeSol(x)

        u_ex = np.zeros([m.NumNodes, NComponents])
        for i in range(m.NumNodes):
                for n in range(NComponents):
                        p = m.Nodes[i]
                        u_ex[i,n] = f[n](p)
        
        # plot the solution and compare with analytical sol
        if showPlot:
                p = Plot(Mesh = m)
	        fig = plt.figure(figsize = (12,8), facecolor = 'w', edgecolor = 'w')
        	nrows = NComponents ; ncols = 3
        	for n in range(NComponents):
        	        ind_ex =  3*n+1
        	        ind_fem = 3*n+2
        	        ind_err = 3*n+3
        	        ax1 = fig.add_subplot(nrows, ncols, ind_ex)
        	        ax2 = fig.add_subplot(nrows, ncols, ind_fem)
        	        ax3 = fig.add_subplot(nrows, ncols, ind_err)
        	        if n == 0:
        	                ax1.set_title(r'$u_{exact}$', fontsize = 'xx-large', fontweight = 'bold')
        	                ax2.set_title(r'$u_{FEM}$', fontsize = 'xx-large', fontweight = 'bold')
        	                ax3.set_title(r'$u_{exact} - u_{FEM}$', fontsize = 'xx-large', fontweight = 'bold')
        	        else:
        	                ax1.set_title('')
        	                ax2.set_title('')
        	                ax3.set_title('')
        	
        	        p.ax = ax1; p.patternPlot(u_ex, n)
        	        p.ax = ax2; p.patternPlot(u, n)
        	        p.ax = ax3; p.patternPlot(u_ex - u, n)
	
	                
	# calculate error norms
	L2err = np.zeros(pde.NComponents); L1err = np.zeros(pde.NComponents)
	for n in range(pde.NComponents):
	        L2err[n] =np.linalg.norm(u[:,n].flatten(order = 'F') -u_ex[:,n].flatten(order = 'F'), ord = 2)
	        L1err[n] = np.linalg.norm(u[:,n].flatten(order = 'F') -u_ex[:,n].flatten(order = 'F'), ord = np.inf)
	               
	return (L2err, L1err)
	

def testErrorScaling():
	meshmatfile_fmt = 'testmesh%d.mat'
	hmax = np.linspace(0.1,0.25,21)
	N = len(hmax)
	err = np.zeros([N,2])
	for i in range(N):
	        h = hmax[i]
	        print 'Solving system for mesh size = ', h
	        meshmatfile = meshmatfile_fmt % i
	        (L2_err, L1_err) = testPoisson(meshmatfile, showPlot = False)
	        err[i,0] = L2_err[0]; err[i,1] = L1_err[0]
	
	logh = np.log10(1./hmax)
	logL2err = np.log10(err[:,0]); logL1err = np.log10(err[:,1])
	statoutL2 = stats.linregress(logh, logL2err); statoutL1 = stats.linregress(logh, logL1err)
	slopeL2 = statoutL2[0]; interceptL2 = statoutL2[1]
	slopeL1 = statoutL1[0]; interceptL1 = statoutL1[1]
	logL2err_fit = slopeL2*logh + interceptL2
	logL1err_fit = slopeL1*logh + interceptL1
	
	fig = plt.figure(figsize = (8,4), facecolor = 'w', edgecolor = 'w')
	ax1 = fig.add_subplot(121)
	ax1.scatter(logh, logL2err, marker = 'o', color = 'red')
	ax1.plot(logh, logL2err_fit, color = 'black')
	
	ax2 = fig.add_subplot(122)
	ax2.scatter(logh, logL1err, marker = 'o', color = 'red')
	ax2.plot(logh, logL1err_fit, color = 'black')
	
	ax1.set_xlabel(r'$log(\frac{1}{h})$', fontsize = 'large')
	ax1.set_ylabel(r'$log(\epsilon_{L_2})$', fontsize = 'large')
	ax2.set_xlabel(r'$log(h)$', fontsize = 'large')
	ax2.set_ylabel(r'$log(\epsilon_{L_1})$', fontsize = 'large')
	
	plt.subplots_adjust(wspace = 0.5, bottom = 0.2, left = 0.12)
	
	print "L2 error order = ", slopeL2
	print "L1 error order = ", slopeL1

if __name__ == '__main__':
	#testMesh()
	#testShapeFnPlot()
	#testPatternPlot()
	#testShapeFnPlot()		
	#testAssembly()
	testPoisson()
	#testErrorScaling()

plt.show()
