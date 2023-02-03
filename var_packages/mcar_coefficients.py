import numpy as np


phi1 = np.loadtxt('var_coefficients_phi1.txt',dtype='f',delimiter=',')
phi1 = np.array(phi1)

phi2 = np.loadtxt('var_coefficients_phi2.txt',dtype='f',delimiter=',')
phi2 = np.array(phi2)

phi3 = np.loadtxt('var_coefficients_phi3.txt',dtype='f',delimiter=',')
phi3 = np.array(phi3)

phi4 = np.loadtxt('var_coefficients_phi4.txt',dtype='f',delimiter=',')
phi4 = np.array(phi4)
"""
print('VAR model coefficients phi_1 = ', phi1)
print(' ')
print('VAR model coefficients phi_2 = ', phi2)
print(' ')
print('VAR model coefficients phi_3 = ', phi3)
print(' ')
print('VAR model coefficients phi_4 = ', phi4)
print(' ')
"""
def A1(phi1):
	a1 = -phi1
	return a1

def A2(phi1,phi2):
	a2 = -3*phi1-phi2
	return a2

def A3(phi1,phi2,phi3):
	a3 = -3*phi1-2*phi2-phi3
	return a3

def A4(phi1,phi2,phi3,phi4):
	a4 = -phi1 - phi2 - phi3 - phi4
	return a4

matrix_A1 = [[A1(phi1[0][0])-1,A1(phi1[0][1])],
			 [A1(phi1[1][0]),A1(phi1[1][1])-1]]

matrix_A2 = [[A2(phi1[0][0],phi2[0][0])+1,A2(phi1[0][1],phi2[0][1])],
			 [A2(phi1[1][0],phi2[1][0]),A2(phi1[1][1],phi2[1][1])+1]]

matrix_A3 = [[A3(phi1[0][0],phi2[0][0],phi3[0][0])-1,A3(phi1[0][1],phi2[0][1],phi3[0][1])],
			 [A3(phi1[1][0],phi2[1][0],phi3[1][0]),A3(phi1[1][1],phi2[1][1],phi3[1][1])-1]]			 

matrix_A4 = [[A4(phi1[0][0],phi2[0][0],phi3[0][0],phi4[0][0])+1,A4(phi1[0][1],phi2[0][1],phi3[0][1],phi4[0][1])],
			 [A4(phi1[1][0],phi2[1][0],phi3[1][0],phi4[1][0]),A4(phi1[1][1],phi2[1][1],phi3[1][1],phi4[1][1])+1]]
"""
print('MCAR model coefficients A1 = ', matrix_A1)
print(' ')
print('MCAR model coefficients A2 = ', matrix_A2)
print(' ')
print('MCAR model coefficients A3 = ', matrix_A3)
print(' ')
print('MCAR model coefficients A4 = ', matrix_A4)
print(' ')
"""
"""
matrix_A = [[np.diag(0,0),np.diag(1,1),np.diag(0,0),np.diag(0,0)],
			[np.diag(0,0),np.diag(0,0),np.diag(1,1),np.diag(0,0)],
			[np.diag(0,0),np.diag(0,0),np.diag(0,0),np.diag(1,1)],
			[matrix_A4,matrix_A3,matrix_A2,matrix_A1]]

p1 = np.hstack((np.diag([0,0]),np.diag([1,1]),np.diag([0,0]),np.diag([0,0])))
p2 = np.hstack((np.diag([0,0]),np.diag([0,0]),np.diag([1,1]),np.diag([0,0])))
p3 = np.hstack((np.diag([0,0]),np.diag([0,0]),np.diag([0,0]),np.diag([1,1])))
p4 = np.hstack((matrix_A4,matrix_A3,matrix_A2,matrix_A1))
"""
"""
matrix_A = np.block([[np.diag([0,0]),np.diag([1,1]),np.diag([0,0]),np.diag([0,0])], 
					[np.diag([0,0]),np.diag([0,0]),np.diag([1,1]),np.diag([0,0])], 
					[np.diag([0,0]),np.diag([0,0]),np.diag([0,0]),np.diag([1,1])], 
					[matrix_A4,matrix_A3,matrix_A2,matrix_A1]])
"""

#matrix_A = np.vstack((p1[:,0],p2[:,0],p3[:,0],p4[:,0]))

matrix_B = [[0,0,1,0,0,0,0,0],
			[0,0,0,1,0,0,0,0],
			[0,0,0,0,1,0,0,0],
			[0,0,0,0,0,1,0,0],
			[0,0,0,0,0,0,1,0],
			[0,0,0,0,0,0,0,1],
			[phi4[0][0],phi4[0][1],phi3[0][0],phi3[0][1],phi2[0][0],phi2[0][1],phi1[0][0],phi1[0][1]],
			[phi4[1][0],phi4[1][1],phi3[1][0],phi3[1][1],phi2[1][0],phi2[1][1],phi1[1][0],phi1[1][1]]]
matrix_B = np.matrix(matrix_B)
#print(matrix_B)
#print(matrix_B.shape)
w_b,v_b = np.linalg.eig(matrix_B)
#print(w_b)
#print(np.absolute(w_b))



matrix_A = [[0,0,1,0,0,0,0,0],
			[0,0,0,1,0,0,0,0],
			[0,0,0,0,1,0,0,0],
			[0,0,0,0,0,1,0,0],
			[0,0,0,0,0,0,1,0],
			[0,0,0,0,0,0,0,1],
			[matrix_A4[0][0],matrix_A4[0][1],matrix_A3[0][0],matrix_A3[0][1],matrix_A2[0][0],matrix_A2[0][1],matrix_A1[0][0],matrix_A1[0][1]],
			[matrix_A4[1][0],matrix_A4[1][1],matrix_A3[1][0],matrix_A3[1][1],matrix_A2[1][0],matrix_A2[1][1],matrix_A1[1][0],matrix_A1[1][1]]]
matrix_A = np.matrix(matrix_A)
#print(matrix_A)
#print(matrix_A.shape)
w_a,v_a = np.linalg.eig(matrix_A)
print(w_a)
#print(np.real(w_a))

#eps=-0.03
eps=-0.03
matrix_A2 = [[0,0,1,0,0,0,0,0],
			[0,0,0,1,0,0,0,0],
			[0,0,0,0,1,0,0,0],
			[0,0,0,0,0,1,0,0],
			[0,0,0,0,0,0,1,0],
			[0,0,0,0,0,0,0,1],
			[matrix_A4[0][0]+eps,matrix_A4[0][1]+eps,matrix_A3[0][0]+eps,matrix_A3[0][1]+eps,matrix_A2[0][0]+eps,matrix_A2[0][1]+eps,matrix_A1[0][0]+eps,matrix_A1[0][1]+eps],
			[matrix_A4[1][0]+eps,matrix_A4[1][1]+eps,matrix_A3[1][0]+eps,matrix_A3[1][1]+eps,matrix_A2[1][0]+eps,matrix_A2[1][1]+eps,matrix_A1[1][0]+eps,matrix_A1[1][1]+eps]]
matrix_A2 = np.matrix(matrix_A2)
#print(matrix_A2)
#print(matrix_A2.shape)
w_a2,v_a2 = np.linalg.eig(matrix_A2)
print(w_a2)
#print(np.real(w_a2))
np.set_printoptions(suppress=True)
print(w_a2)
print(np.real(w_a))
print(np.real(w_a2))





