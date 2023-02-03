import numpy as np
from sympy import Symbol, sqrt, lambdify
from scipy.optimize import fsolve

def numerical_solution_of_delta_constant(r_ds_temp,r_ds_uwind,delta_temp,delta_uwind):
	# Find covariance matrix
	y = [r_ds_temp,r_ds_uwind]
	cov_y = np.cov(y)
	print("Covariance matrix = ",cov_y)

	# 8) Extract and look at model error coefficients
	d1 = delta_temp
	d2 = delta_uwind
	s11 = cov_y[0,0]
	s12 = cov_y[0,1]
	s22 = cov_y[1,1]
	print("s11 = ",s11)
	print("s12 = ",s12)
	print("s22 = ",s22)
	print("d1 = ",d1)
	print("d2 = ",d2)

	print(' ')
	print(' ')
	print('********** TEST **********')
	print(d1/np.sqrt(2*s11))
	print(d2/np.sqrt(2*s22))


	print(' ')
	print(' ')
	print('********** TERE ARE 12 POSSIBLE SOLUTIONS **********')
	# Find the idiosyncratic error distribution scales (constants) numerically

	print(' ')
	print('********** #1, #2, #3, #4 **********')
	print(' ')

	x = Symbol('x')

	expr = sqrt(	s11 - (		(1.0/(2.0*x))*(	sqrt( 2*s11*(x**2) - d1**2 ) - d1)	)**2	)\
			*(1.0/(2.0*x))*(	sqrt( 2*s22*(x**2) - d2**2 ) - d2)\
			+(1.0/(2.0*x))*(	sqrt( 2*s11*(x**2) - d1**2 ) - d1)\
			*sqrt(	s22 - (		(1.0/(2.0*x))*(	sqrt( 2*s22*(x**2) + d2**2 ) - d2)	)**2	)\
			-s12
	#sol=solve(expr, x)
	#print("Solutions: ",sol)
	print('********** EXPR DEFINED **********')
	func_np = lambdify(x, expr, modules=['numpy'])
	print('********** LAMBDIFY DONE **********')
	solution = fsolve(func_np, 1.3)
	print('********** SOLUTION COMPUTED **********')
	C=solution[0]
	print(C)
	
	
	# Test conditions
	computed_beta2 = (1.0/(2.0*C))*(	sqrt( 2*s11*C**2 - d1**2 ) - d1)
	computed_beta3 = (1.0/(2.0*C))*(	sqrt( 2*s22*C**2 - d2**2 ) - d2)


	print("Solution: ",C)
	print("zero beta2 const: ",d1**2/np.sqrt(s11))
	print("zero beta3 const: ",d2**2/np.sqrt(s22))

	computed_beta2 = (1.0/(2.0*C))*(	sqrt( 2*s11*C**2 - d1**2 ) - d1)
	computed_beta3 = (1.0/(2.0*C))*(	sqrt( 2*s22*C**2 - d2**2 ) - d2)
	print("BETA2=",computed_beta2)
	print("BETA3=",computed_beta3)

	computed_beta1 = np.sqrt(float(s11-computed_beta2**2))
	computed_beta4 = np.sqrt(float(s22-computed_beta3**2))
	print("BETA1=",computed_beta1)
	print("BETA4=",computed_beta4)

	#TEST CONDITIONS:
	if float(C)>0:
		print("C>0: ",C)


	cond1 = d1/np.sqrt(2*s11)
	cond2 = d2/np.sqrt(2*s22)
	cond3 = computed_beta2**2
	cond4 = computed_beta3**2
	print("Condition 1, delta_1/sqrt(2*sigma_11):",cond1)
	print("Condition 1, delta_2/sqrt(2*sigma_22):",cond2)
	print("Condition 3, beta_12**2:",cond3)
	print("Condition 4, beta_21**2:",cond4)

	cond_on_s11 = d1**2/(2*C**2)
	cond_on_s22 = d2**2/(2*C**2)

	if s11 >= cond_on_s11:
		print("Condition on s11 ok: ",s11," >= ",cond_on_s11)

	if s22 >= cond_on_s22:
		print("Condition on s22 ok: ",s22," >= ",cond_on_s22)

	cond_on_d1 = C*np.abs(computed_beta1) + C*np.abs(computed_beta2)
	cond_on_d2 = C*np.abs(computed_beta3) + C*np.abs(computed_beta4)

	print(cond_on_d1," == ",d1)
	print(cond_on_d2," == ",d2)
	print("Relative error delta_1: ",np.abs(cond_on_d1-d1)/d1)
	print("Relative error delta_2: ",np.abs(cond_on_d2-d2)/d2)
	
	"""
	if np.testing.assert_almost_equal(d1, cond_on_d1, decimal=6) == None:
		print("Condition on d1 ok: ",round(d1,6)," == ",round(cond_on_d1,6))

	if np.testing.assert_almost_equal(d2, cond_on_d2, decimal=6) == None:
		print("Condition on d1 ok: ",round(d2,6)," == ",round(cond_on_d2,6))
	"""

	cond_on_beta2 = float(s11 - computed_beta2**2)
	cond_on_beta3 = float(s22 - computed_beta3**2)

	if cond_on_beta2 >= 0:
		print("Condition on beta2 ok: ",cond_on_beta2," >= ",0)

	if cond_on_beta3 >= 0:
		print("Condition on beta3 ok: ",cond_on_beta3," >= ",0)

	return 0

numerical_solution_of_delta_constant(r_ds_temp,r_ds_uwind,delta_temp,delta_uwind)