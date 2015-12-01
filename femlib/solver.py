#!/usr/bin/env/python

import os

fort_src = '''SUBROUTINE CALCF(F, U, N)
 IMPLICIT NONE
 INTEGER, INTENT(IN) :: N
 REAL(8), INTENT(IN), DIMENSION(0:N-1) :: U
 REAL(8), INTENT(INOUT), DIMENSION(0:N-1) :: F
 
 INTEGER :: I
 DO I = 0, N-1
  F(i) = 0.d0
 ENDDO
END SUBROUTINE


SUBROUTINE BKEULER(M, K, F0, U0, Delta_t, N, F, U , FLAG)
 IMPLICIT NONE
	
 ! i/o variables	
 INTEGER, INTENT(IN) :: N
 REAL(8), INTENT(IN) :: Delta_t
 REAL(8), INTENT(IN), DIMENSION(0:N-1, 1) :: F0, U0
 REAL(8), INTENT(IN), DIMENSION(0:N-1, 0:N-1) :: K, M
 EXTERNAL :: CALCF	
 REAL(8), INTENT(OUT), DIMENSION(0:N-1, 1) :: F, U
 INTEGER, INTENT(OUT) :: FLAG	

 ! parameters used in this problem 
 REAL(8) :: TOL, MAXITER
 
 ! on the fly variables	
 INTEGER :: ITER, LAPACK_FLAG
 REAL(8) :: CURR_TOL, NORM0, NORM
 REAL(8), DIMENSION(0:N-1, 0:N-1) :: A
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
  
  ! SGESV is a LAPACK lib function and its syntax demands
  ! that the matrix b comes out as the output. Hence this step
  U = b0 + bf
  CALL SGESV(N, 1, A, N, pivot, U, N, LAPACK_FLAG) 
  
  CALL CALCF(F, U, N) 					
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
recompile = False
LAPACK_PATH = None

#compile fort code
def fort_compile():
        if not os.path.isfile('bkeuler.so') or recompile:
                fortcode = file('bkeuler.f90', 'w'). write(fort_src)
                cmdstring = 'f2py -L%s -llapack -c -m bkeuler bkeuler.f90 --fcompiler=gfortran' % LAPACK_PATH
                print cmdstring; raw_input()
                os.system(cmdstring)
                for this_file in ['bkeuler.f90', 'bkeuler.o']:
	                if os.path.isfile(this_file): os.remove(this_file)


def bkEuler():
	import bkeuler
	F,U, Flag = bkeuler.bkeuler(m = M, k = K, f0 = F0, u0 = U0, delta_t = Delta_t)
        if not Flag:
                print "Solution did not converge in max # of iterations"
        
        return U, F

