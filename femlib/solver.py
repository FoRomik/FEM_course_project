#!/usr/bin/env/python

import os
import numpy as np

fort_src = '''
SUBROUTINE CALCF2(U, f_val, I, N)
 ! f_val = f(u) or g(u), based on U vals and index I 
 IMPLICIT NONE
 INTEGER, INTENT(IN) :: I, N
 REAL(8), INTENT(IN), DIMENSION(0:N-1,1) :: U
 REAL(8), INTENT(INOUT) :: F_VAL
 
 REAL(8) :: u1, u2
 u1 = 0
 u2 = 0
 
 ! Please specify f and g
 
 IF (I<N/2) THEN 
  u1 = U(I,1)
  u2 = U(I+N/2,1)
  ! F_VAL = f(u1,u2)
 ELSE 
  u2 = U(I,1)
  u1 = U(I-N/2,1)
  ! F_VAL = g(u1,u2)
 ENDIF
 
END SUBROUTINE



SUBROUTINE ASSEMBF(F, U, N, Elements, Areas, NElements)
 IMPLICIT NONE
 INTEGER, INTENT(IN) :: N, NElements
 REAL(8), INTENT(IN), DIMENSION(0:N-1,1) :: U
 REAL(8), INTENT(IN), DIMENSION(0:NElements-1) :: Areas
 INTEGER, INTENT(IN), DIMENSION(0:NElements-1, 0:2) :: Elements
 
 EXTERNAL :: CALCF2
 REAL(8), INTENT(INOUT), DIMENSION(0:N-1,1) :: F
 
 INTEGER :: I, T
 REAL(8) :: F_ELEMENT, F_VAL
 
 ! zero the Fs
 F_ELEMENT = 0.0
 DO I = 0, N-1
        F(I,1) = 0.0
 END DO
 
 ! assemble F over the elements
 DO T = 0, NElements-1
  DO I = 0, 2
  CALL CALCF2(U, F_VAL, I, N)
  F_ELEMENT = F_ELEMENT + (2*Areas(T)/18) * F_VAL
  ENDDO
  F(Elements(T,0),1) = F(Elements(T,0),1) + F_ELEMENT
  F(Elements(T,1),1) = F(Elements(T,1),1) + F_ELEMENT
  F(Elements(T,2),1) = F(Elements(T,2),1) + F_ELEMENT
 ENDDO
END SUBROUTINE



SUBROUTINE BKEULER(M, K, F0, U0, Delta_t, N, F, U , FLAG, Elements, Areas, NElements)
 IMPLICIT NONE
	
 ! i/o variables	
 INTEGER, INTENT(IN) :: N, NElements
 REAL(8), INTENT(IN) :: Delta_t
 REAL(8), INTENT(IN), DIMENSION(0:N-1, 1) :: F0, U0
 REAL(8), INTENT(IN), DIMENSION(0:N-1, 0:N-1) :: K, M
 INTEGER, INTENT(IN), DIMENSION(0:NElements-1, 0:2) :: Elements
 REAL(8), INTENT(IN), DIMENSION(0:NElements-1) :: Areas
 EXTERNAL :: CALCF	
 REAL(8), INTENT(OUT), DIMENSION(0:N-1, 1) :: F, U
 INTEGER, INTENT(OUT) :: FLAG	

 ! parameters used in this problem 
 REAL(8) :: TOL, MAXITER
 
 ! on the fly variables	
 INTEGER :: ITER, LAPACK_FLAG
 REAL(8) :: CURR_TOL, NORM0, NORM
 REAL(8), DIMENSION(0:N-1, 0:N-1) :: A, A_LAPACK
 REAL(8), DIMENSION(0:N-1, 1) :: bf, b0, U_old, F_old, pivot
 
 TOL = 1E-3
 MAXITER = 10

 FLAG = 1
 CURR_TOL = 1.0
 ITER = 0
 A = M + Delta_t*K
 U_old = U0
 F_old = F0
 b0 = MATMUL(M, U_old)

 DO WHILE (CURR_TOL > TOL)
  NORM0 = NORM2(F_old)		
  bf = Delta_t * F0
  
  ! DGESV is a LAPACK lib function and its syntax demands
  ! that the matrix b comes out as the output. Hence this step
  U = b0 + bf
  A_LAPACK = A
  CALL DGESV(N, 1, A_LAPACK, N, pivot, U, N, LAPACK_FLAG) 
  
  CALL ASSEMBF(F, U, N, Elements, Areas, NElements) 					
  NORM = NORM2(F)
  CURR_TOL = ABS(NORM - NORM0)
		
  ! update	
  F_old = F
  U_old = U
  ITER = ITER + 1
  IF (ITER >= MAXITER) THEN 
   FLAG = 0
   EXIT
  ENDIF	
 
 END DO

END SUBROUTINE
'''


# specify these before calling bkEuler
M = None
K = None
F0 = None
U0 = None
Delta_t = None
recompile = True
LAPACK_PATH = None

#compile fort code
def fort_compile():
        if not os.path.isfile('bkeuler.so') or recompile:
                fortcode = file('bkeuler.f90', 'w'). write(fort_src)
                cmdstring = 'f2py -L%s -llapack -c -m bkeuler bkeuler.f90 --fcompiler=gfortran' % LAPACK_PATH
                os.system(cmdstring)
                for this_file in ['bkeuler.f90', 'bkeuler.o']:
	                if os.path.isfile(this_file): os.remove(this_file)


def bkEuler():
	import bkeuler
	F,U, Flag = bkeuler.bkeuler(m = M, k = K, f0 = F0, u0 = U0, delta_t = Delta_t)
        if not Flag:
                print "Solution did not converge in max # of iterations"
        
        return U, F


def naiveSolve(A,b):
        x = np.linalg.solve(A,b)
        return x
