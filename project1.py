#/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import time
from timeit import default_timer as timer



def ExactSolution(x):
	"""
	Exact solution of -u''(x)=f(x), x in(0,1), u(0)=u(1)=0
	f(x) = 100exp(-10x)
	"""		
	
	exp = np.exp
	return 1 - (1 - exp(-10))*x - exp(-10*x)
	

def source_term(x):
	return 100*np.exp(-10*x)



def Create_Matrix(N):
	"""
	Create a tridiagonal N x N matrix
	with integer 2 along the main diagonal
	and integer -1 along upper and lower diagonal.
	"""
	
	A = np.zeros((N, N))
	
	for i in range(N):
		A[i][i] = 2
		A[i][i-1] = -1	
		A[i-1][i] = -1

	return A



def Solve_Linear_System(source_term, N, endpoint, case):
	"""
	Solve linear system Ax=v by Gaussian elimination
	for two cases, first by assuming non identical elements
	along each diagonal and second case by assuming identical
	elements along each diagonal.
	"""

	
	# Obtain tridiagonal matrix
	A = Create_Matrix(N=N)

	h = 1./(N+1)		# Step size
	
	x = np.linspace(0+h, endpoint, N, endpoint=False)
	

	# Initial values
	a = np.zeros(N)
	b = np.zeros(N)
	c = np.zeros(N)
	f = np.zeros(N)		# Right-hand-side


	u = np.zeros(N)		# Solution
	

	# New values to be computed
	b_new = np.zeros(N)			# New main diagonal
	f_new = np.zeros(N)			# New right-hand side

	

	# Fill initial arrays
	for i in range(N):
		a[i] = -1
		b[i] = 2
		c[i] = -1
		f[i] = h*h*source_term(x[i])



	# Perform Gaussian elimination
	
	if case == 'general':
		#print("Solving linear system in general case")
		#time.sleep(1)
		
		b_new[0] = b[0]
		f_new[0] = f[0]

		for i in range(1, N):
					
			b_new[i] = b[i] - (c[i-1]*a[i-1])/b_new[i-1]
			f_new[i] = f[i] - (f_new[i-1]*a[i-1])/b_new[i-1]

		
	elif case == 'special':
		#print("Solving linear system in special case")
		#time.sleep(1)

			
		b_new[0] = b[0]
		f_new[0] = f[0]

		
		for i in range(1, N):
			b_new[i] = b[0] - c[0]*a[0]/(b_new[i-1])
			f_new[i] = f[i] - f_new[i-1]*a[0]/b_new[i-1]


	else:
		print("Missing or incorrect argument. Please spesify case='general' or case='special'")
		sys.exit()


	# Perform backward substitution

	# Last value in array	
	u[N-1] = f_new[N-1]/b_new[N-1]
	
	for i in reversed(range(1, N)):
		u[i-1] = (f_new[i-1] - u[i]*c[i-1])/(b_new[i-1])
		
	return u, x



def plot_and_compare(case, N, show=False, saveimage=False):

	# Numerical solution
	start = timer()
	u, x = Solve_Linear_System(source_term, N=N, endpoint=1, case=case)
	end = timer()

	total_time = end - start

	
	# Exact solution and fine mesh
	x_fine = np.linspace(0, 1, 1000)
	u_exact = ExactSolution(x_fine)


	plt.plot(x_fine, u_exact, x, u, '--')
	plt.legend(['Exact', 'Numerical'])
	plt.xlabel('x'); plt.ylabel('u')
	plt.title(f"{case} algorithm N={N}")		

	if saveimage:
		plt.savefig("plot_N={}.png".format(N))


	if show:
		plt.show()
	
	
	print(f"Total time elapsed for {case} algorithm with {N} mesh points: {total_time}")






	


def ComputeRelativeError(N):
	
	u, x = Solve_Linear_System(source_term, N=N, endpoint=1, case='special')
	v = ExactSolution(x)

	error = np.zeros(N)
	
	
	log = np.log10
	abs = np.abs

	for i in range(N):
		error[i] = log(abs((v[i] - u[i])/u[i]))

	#print(error)	

	
	plt.plot(error)
	plt.title('Error')
	plt.show()
	




if __name__=='__main__':
	
	os.system('clear')

	for N in 10, 100, 1000:

		u, x = Solve_Linear_System(source_term, N, endpoint=1.0, case='special')
		plot_and_compare(case='special', N=N, show=True, saveimage=False)

		ComputeRelativeError(N=N)
	











	
	






#plot_and_compare(case='special', N=10000, show=True, saveimage=False)































	
	


