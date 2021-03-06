import math
import numpy as np

#python lu decomposition
def lu(A):
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n, dtype=np.float32)
    for i in range(n):
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]
    return L, U

#the symmetric A
A = np.array([[18.0, 22.0, 54.0, 42.0],
                [22.0, 70.0, 86.0, 62.0],
                [54.0, 86.0, 174.0, 134.0],
                [42.0, 62.0, 134.0, 106.0]])

#calculate the determinant
print("the determinant of first minor is:", A[0, 0], "> 0")
print("the determinant of second minor is:", np.linalg.det(np.array([[18, 22],
                                                                     [22, 70]])), "> 0")
print("the determinant of third minor is:", np.linalg.det(np.array([[18, 22, 54],
                                                                    [22, 70, 86],
                                                                    [54, 86, 174]])), "> 0")
print("the determinant of A is:", np.linalg.det(A), "> 0")

L, U = lu(A)
#create the matrix sqrt D
sqrtD = U.copy()
numOfRows = sqrtD.shape[0]
for i in range(numOfRows):
    for j in range(numOfRows):
        if i != j:
            sqrtD[i, j] = 0
        else:
            sqrtD[i, i] = math.sqrt(sqrtD[i, i])

#nultiply L and sqrtD
R = np.matmul(L, sqrtD)
print()
print("the matrix R:")
print()
print(R)

