#!/usr/bin/env python

import numpy as np
import scipy as scp
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fem import *


K0 = importInitMesh()
m = Mesh(K0)

def testMesh():
	ax = plt.subplot(111)
	m.refineMesh()
	p = Plot(Mesh = m, ax = ax)	
	p.plotMesh()


def testShapeFnPlot():	
	fig = plt.figure()
	ax3d = Axes3D(fig)
	Node = m.Nodes[22]
	p = Plot(Mesh = m, ax = ax3d)
	p.plotShapeFunc(Node)  


def testPatternPlot():
	ax = plt.subplot(111)
	u = np.zeros(m.NumNodes)
	for i in range(m.NumNodes):
		u[i] = np.cos(m.Nodes[i,0] + m.Nodes[i,1])
	p = Plot(Mesh = m, ax = ax, u_Node = u)
	p.patternPlot()


def testElementMat():
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
	f = lambda node : node[0] + node[1]
	a = Assemb(Mesh = m, f = f)
	a.AssembMat_naive()
	print a.globalStiffMat
	print a.globalMassMat
	print a.globalfMat


def testbkEuler():
        N = 4
        K = np.random.random((N,N))
        M = np.random.random((N,N))
        
        F0 = np.zeros([4,1])
        U0 = np.ones([4,1])
        delta_t = 1.0
        numpy_sol = np.linalg.solve(M+delta_t*K, delta_t*F0 + np.dot(M, U0))
        
        import solver
        solver.K = K ; solver.M = M ; solver.F0 = F0 ; solver.U0 = U0 ; solver.Delta_t = delta_t
        solver.LAPACK_PATH = '/usr/lib/lapack'
        solver.fort_compile()
        fort_sol = solver.bkEuler()
        
        print numpy_sol
        print '\n\n---------------\n\n'
        print fort_sol[0]

def testPoisson():
	f = lambda node: 1.0
	a = Assemb(Mesh = m, f = f)
	a.AssembMat_naive()
	K = a.globalStiffMat
	F = a.globalfMat
	u = np.linalg.solve(K,F)

	ax = plt.subplot(111)
	p = Plot(Mesh = m, ax = ax)
	p.patternPlot(u_Node = u, showGrid = False)
	

def testErrorScaling():
	matfile_fmt = 'testmesh%d.dat'
	diams = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
	err = np.zeros(len(diams))	
		
	f_exact = lambda node: np.cos(node[0]**2. + node[1]**2. - 1) #trial function		
	f = lambda node: -4*( (node[0]**2+node[1]**2)*np.cos(node[0]**2. + node[1]**2. - 1) + np.sin(node[0]**2. + node[1]**2. - 1))

	for i, h in enumerate(diams):
		matfile = matfile_fmt % i
		
		# fem solution	
		print 'Solving FEM problem form mesh dia = ', h		
		K = importInitMesh(matfile)
		mesh = Mesh(K)
		a = Assemb(Mesh = mesh, f = f)
		a.AssembMat_naive()
		K = a.globalStiffMat
		F = a.globalfMat
		u = np.linalg.solve(K,F)

		# exact solution		
		u_exact = np.zeros(mesh.NumNodes)
		for j in range(mesh.NumNodes):
			u_exact[j] = f_exact(mesh.Nodes[i])
		
		#p = Plot(Mesh = mesh)
		#p.ax = ax1; p.patternPlot(u_exact); ax1.set_title('Exact')
		#p.ax = ax2; p.patternPlot(u); ax2.set_title('FEM')
		#plt.show()
	

		# error
		err[i] = np.linalg.norm(u_exact - u)
	
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
	#testErrorScaling()
	testbkEuler()
plt.show()
