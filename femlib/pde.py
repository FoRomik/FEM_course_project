#!/usr/bin/env python

import numpy as np
import fem

class PDE:
        def __init__(self, NComponents = 1, Mesh = None, StiffMat = None, MassMat = None):
                self.NComponents = NComponents
                self.Mesh = Mesh
                self.StiffMat = StiffMat
                self.MassMat = MassMat
        
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

        
        def AssembLHS(self, StiffMat = None, MassMat = None):
                pass


        def AssemPDE(self):
                pass
                
                
        def makeSol(self, x):
                x = x.reshape(self.Mesh.NumFreeNodes, self.NComponents, order = 'F')
                sol = np.zeros([self.Mesh.NumNodes, self.NComponents])
                for n in range(self.NComponents):
                        for i, node in enumerate(self.Mesh.FreeNodes):
                                sol[node, n] = x[i,n]
                        for node in self.Mesh.DirNodes:
                                p = self.Mesh.Nodes[node]
                                gDir = self.setDirFunc(Mesh = self.Mesh)[n]
                                sol[node,n] = gDir(p)
                
                return sol


                                                       
class Elliptic(PDE):
        def __init__(self, NComponents = 1, Mesh = None, StiffMat = None, MassMat = None, u = None):
                self.NComponents = NComponents
                self.Mesh = Mesh
                self.StiffMat = StiffMat
                self.MassMat = MassMat
                if u is None: self.u = np.zeros([self.Mesh.NumNodes, self.NComponents])
                                 
               
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
                                elemSourceTerm = elemShapeFn.getSourceTermElement_P1(f_vals[T][:,n])
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
        
        
        def AssemPDE(self):
                # assemble LHS
                K_Free = self.getFreeNodeArray(self.StiffMat)
                M_Free = self.getFreeNodeArray(self.MassMat)
                LHSMat = self.AssembLHS(K_Free,M_Free)
               
                # assemble Neumann terms
                b_Neumann = self.getFreeNodeArray(self.getNeumannBC()).flatten(order = 'F')
                b_Neumann = b_Neumann.reshape(len(b_Neumann),1)
                
                # assemble Dirichlet terms
                K_Dir = self.getDirNodeArray(self.StiffMat)
                M_Dir = self.getDirNodeArray(self.MassMat)
                A_Dir = self.AssembLHS(K_Dir, M_Dir)
                u_Dir = self.getDirBC().flatten(order = 'F')
                u_Dir = u_Dir.reshape(len(u_Dir), 1)
                b_Dir = np.dot(A_Dir, u_Dir)
                
                # assemble source terms (may be nonlinear)
                b_Src = self.getFreeNodeArray(self.getSrcTerm()).flatten(order = 'F')
                b_Src = b_Src.reshape(len(b_Src), 1)  
                
                # assemble RHSVec (simply return the three vectors 
                # and let the calling script do the assembly. This ensures
                # generality for nonlinear cases)
                RHSVec = (b_Src, b_Neumann, b_Dir)  
                  
                return LHSMat, RHSVec
         
         
        
