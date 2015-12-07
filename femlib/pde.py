#!/usr/bin/env python

import os
import numpy as np
import fem

LAPACK_PATH = '/usr/lib/lapack'
fort_recompile = False
linsolv_src = '''
SUBROUTINE MUL(A, B, C, rowsA, colsA, rowsB, colsB)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: rowsA, rowsB, colsA, colsB            
        REAL(8), INTENT(IN), DIMENSION(0:rowsA-1, 0:colsA-1) :: A
        REAL(8), INTENT(IN), DIMENSION(0:rowsB, 0:colsB) :: B
        REAL(8), INTENT(OUT), DIMENSION(0:rowsA-1, 0:colsB) :: C
        
        C = MATMUL(A,B)
        
END SUBROUTINE
        
SUBROUTINE LAPACKSOLV(A, b, x, FLAG, Dim)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: Dim
        REAL(8), INTENT(IN), DIMENSION(0:Dim-1, 0:Dim-1) :: A
        REAL(8), INTENT(IN), DIMENSION(0:Dim-1, 1) :: b
        REAL(8), INTENT(OUT), DIMENSION(0:Dim-1,1) :: x
        INTEGER, INTENT(OUT) :: FLAG
        
        REAL(8), DIMENSION(0:Dim-1, 0:Dim-1) :: A_LAPACK
        REAL(8), DIMENSION(0:Dim-1, 1) :: pivot
        
        
        ! DGESV is a LAPACK lib function and its syntax demands
        ! that the matrix b comes out as the output. Hence this step
        x = b
        A_LAPACK = A
        CALL DGESV(Dim, 1, A_LAPACK, Dim, pivot, x, Dim, FLAG) 
END SUBROUTINE
'''
file('linsolv.f90', 'w').write(linsolv_src)
cmdstring = 'f2py -L%s -llapack -c -m linsolv linsolv.f90 --fcompiler=gfortran' % LAPACK_PATH
if not os.path.isfile('linsolv.so') or fort_recompile: os.system(cmdstring)
if os.path.isfile('linsolv.f90'): os.remove('linsolv.f90')
                

class PDE:
        def __init__(self, NComponents = 1, Mesh = None, StiffMat = None, MassMat = None, u = None):
                self.NComponents = NComponents
                self.Mesh = Mesh
                self.StiffMat = StiffMat
                self.MassMat = MassMat
                if u is None: self.u = np.zeros([self.Mesh.NumNodes, self.NComponents])
        
        
        def getDirNodeArray(self, x):
                Ind = self.Mesh.FreeNodes
                Dir = self.Mesh.DirNodes
                if x.shape[1] > self.NComponents: x = x[Ind][:,Dir]
                return x
        
        
        def getFreeNodeArray(self, x):
                Ind = self.Mesh.FreeNodes
                if x.shape[1] > self.NComponents: x = x[Ind][:,Ind]
                else: x = x[Ind]                
                return x
        
        
        def setDirFunc(self, Mesh = None):
                # returns a list of lambda functions of a co-ordinate (x,y) 
                # Can vary on the mesh since the mesh is passed
                # as an argument
                pass

        
        def setNeumannFunc(self, Mesh = None):
                # returns a list of lambda functions of a co-ordinate (x,y) 
                # can vary on the mesh since the mesh is passed
                # as an argument
                pass

        
        def setSrcFunc(self, Mesh = None, u = None):
                # returns a list of lambda functions of a co-ordinate (x,y)
                # Can vary on both the mesh and solution since the mesh and solution are passed
                # as arguments. So, nonlinear problems can be handled too.
                pass

        
        def getAllSrc(self):
                ret = np.zeros([self.Mesh.NumNodes, self.NComponents]) 
                for i, node in enumerate(self.Mesh.Nodes):
                        f = self.setSrcFunc()
                        for n in range(self.NComponents):
                                ret[i,n] = f[n](node) 
                return ret        

            
        def getSrcTerm(self):
	        ret = np.zeros([self.Mesh.NumNodes,self.NComponents])
	        f_vals = self.getAllSrc()
                for T in self.Mesh.Elements:
                        for n in range(self.NComponents):
                                elemShapeFn = fem.ShapeFn(Mesh = self.Mesh, Element = T)
                                elemSourceTerm = elemShapeFn.getSrcTermElement_P1(f_vals[T][:,n])
                                for i, nodeI in enumerate(T): ret[nodeI,n] += elemSourceTerm[i]

		return ret


        def getNeumannBC(self):
	        if not self.Mesh.NumNeumannEdges: 
	                self.isNeumannAssembled = True
	                return np.zeros([self.Mesh.NumNodes, self.NComponents])
                
                g = self.setNeumannFunc()
	        ret = np.zeros([self.Mesh.NumNodes, self.NComponents])
	        for e in self.Mesh.NeumannEdges:
	                for n in range(self.NComponents):
	                        elemShapeFn = fem.ShapeFn(Mesh = self.Mesh, NeumannEdge = e)
	                        edgeNeumannVec = elemShapeFn.getNeumannVecEdge_P1(g = g[n])
	                        for i, nodeI in enumerate(e): ret[nodeI,n] += edgeNeumannVec[i]
                
                return ret
      
      
        def getDirBC(self):
                if not self.Mesh.NumDirEdges: 
                        self.isDirAssembled = True
                        return np.zeros([self.Mesh.NumDirNodes,self.NComponents])
                
                g = self.setDirFunc()
                u_Dir = np.zeros([self.Mesh.NumDirNodes, self.NComponents])
                for i, node in enumerate(self.Mesh.DirNodes):
                        for n in range(self.NComponents): 
                                p = self.Mesh.Nodes[node]
                                u_Dir[i,n] = g[n](p)
                 
                return u_Dir
                
                 
        def AssembLHS(self, StiffMat = None, MassMat = None):
                pass
        
        
        def AssembBlockStiffMat(self, StiffMat = None):
                pass       


        def AssembBlockMassMat(self, MassMat = None):
                pass


        def AssembPDE(self):
                pass
                
                
        def Solve(self, A, b):
                # lapack solver
                import linsolv
                x, flag = linsolv.lapacksolv(a = A, b = b)
                return x
                
        
        def wrapSol(self, x):
                x = x.flatten(order = 'F')
                x = x.reshape((len(x),1))
                return x
                        
        def unwrapSol(self, x):
                x = x.reshape(self.Mesh.NumFreeNodes, self.NComponents, order = 'F')
                sol = np.zeros([self.Mesh.NumNodes, self.NComponents])
                for n in range(self.NComponents):
                        for i, node in enumerate(self.Mesh.FreeNodes):
                                sol[node, n] = x[i,n]
                        if self.Mesh.NumDirNodes:
                                gDir = self.setDirFunc(Mesh = self.Mesh)[n]
                                for node in self.Mesh.DirNodes:
                                        p = self.Mesh.Nodes[node]
                                        sol[node,n] = gDir(p)
                
                return sol


                                                       
class Elliptic(PDE):                 
        def AssembPDE(self):
                # assemble LHS
                K_Free = self.getFreeNodeArray(self.StiffMat)
                M_Free = self.getFreeNodeArray(self.MassMat)
                K_Free = self.AssembBlockStiffMat(K_Free)
                M_Free = self.AssembBlockMassMat(M_Free)
                
                LHSMat = self.AssembLHS(K_Free, M_Free)


                # assemble Dirichlet terms
                if not self.Mesh.NumDirNodes:
                        return np.zeros([self.Mesh.NumFreeNodes, 1])
                else:        
                        K_Dir = self.getDirNodeArray(self.StiffMat)
                        M_Dir = self.getDirNodeArray(self.MassMat)
                        K_Dir = self.AssembBlockStiffMat(K_Dir)
                        M_Dir = self.AssembBlockStiffMat(M_Dir)
                        A_Dir = self.AssembLHS(K_Dir, M_Dir)
                        u_Dir = self.getDirBC().flatten(order = 'F')
                        u_Dir = u_Dir.reshape(len(u_Dir), 1)
                        b_Dir = np.dot(A_Dir, u_Dir)               


                # assemble Neumann terms
                b_Neumann = self.getFreeNodeArray(self.getNeumannBC()).flatten(order = 'F')
                b_Neumann = b_Neumann.reshape(len(b_Neumann),1)
                
                
                # assemble source terms (may be nonlinear)
                b_Src = self.getFreeNodeArray(self.getSrcTerm()).flatten(order = 'F')
                b_Src = b_Src.reshape(len(b_Src), 1)  
                
                # assemble RHSVec (simply return the three vectors 
                # and let the calling script do the assembly. This ensures
                # generality for nonlinear cases)
                RHSVec = (b_Src, b_Neumann, b_Dir)  
                  
                return LHSMat, RHSVec
         
         
class Parabolic(PDE):
        # turns out this is exactly same as that of the elliptic case
        def AssembPDE(self):
                # assemble LHS
                K_Free = self.getFreeNodeArray(self.StiffMat)
                M_Free = self.getFreeNodeArray(self.MassMat)
                K_Free = self.AssembBlockStiffMat(K_Free)
                M_Free = self.AssembBlockMassMat(M_Free)
                LHSMat = self.AssembLHS(K_Free, M_Free)
                
                # assemble Dirichlet terms
                if not self.Mesh.NumDirNodes:
                        b_Dir = np.zeros([self.Mesh.NumFreeNodes, 1])
                else:        
                        K_Dir = self.getDirNodeArray(self.StiffMat)
                        M_Dir = self.getDirNodeArray(self.MassMat)
                        K_Dir = self.AssembBlockStiffMat(K_Dir)
                        M_Dir = self.AssembBlockStiffMat(M_Dir)
                        A_Dir = self.AssembLHS(K_Dir, M_Dir)
                        u_Dir = self.getDirBC().flatten(order = 'F')
                        u_Dir = u_Dir.reshape(len(u_Dir), 1)
                        b_Dir = np.dot(A_Dir, u_Dir)               


                # assemble Neumann terms
                b_Neumann = self.getFreeNodeArray(self.getNeumannBC()).flatten(order = 'F')
                b_Neumann = b_Neumann.reshape(len(b_Neumann),1)
                
                
                # assemble source terms (may be nonlinear)
                b_Src = self.getFreeNodeArray(self.getSrcTerm()).flatten(order = 'F')
                b_Src = b_Src.reshape(len(b_Src), 1)  
                
                # assemble RHSVec (simply return the three vectors 
                # and let the calling script do the assembly. This ensures
                # generality for nonlinear cases)
                RHSVec = (b_Src, b_Neumann, b_Dir)  
               
                return LHSMat, RHSVec
                
                
        def checkStability(self):
                 pass 
