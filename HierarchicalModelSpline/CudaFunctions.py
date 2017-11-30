import numpy as np
from numba import cuda,float32,int8
import math

max2d  = (13,5)
e = 2.718281828

#IMPORTANT: The functions that have to return new arrays don't return anything
#and write their results in the last parameter (that contain the array created before).

@cuda.jit(device=True)
def zeros2d(array):
    for i in range(len(array)):
        for j in range(len(array[0])):
            array[i][j] = 0

@cuda.jit(device=True)
def add1d(A, B, result):
    for i in range(len(A)):
        result[i] = A[i] + B[i]
   
@cuda.jit(device=True)
def add2d(A, B, result):
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i,j] = A[i,j] + B[i,j]

@cuda.jit(device=True)
def sub1d(A, B, result):
    for i in range(len(A)):
        result[i] = A[i] - B[i]

@cuda.jit(device=True)    
def sub2d(A, B, result):
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i,j] = A[i,j] - B[i,j]

@cuda.jit(device=True)
def subValue(A, v, result):
    for i in range(len(A)):
        result[i] = A[i] - v

@cuda.jit(device=True)            
def multValue(A, v, result):
    for i in range(len(A)):
        result[i] = A[i] * v            

@cuda.jit(device=True)            
def divValue(A, v, result):
    for i in range(len(A)):
        result[i] = A[i] / v

@cuda.jit(device=True)
def divValue2d(A, v, result):
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i,j] = A[i,j] / v 

@cuda.jit(device=True)        
def powValue(A, v, result):
    for i in range(len(A)):
        result[i] = A[i] ** v        

@cuda.jit(device=True)        
def negVec(A, result):
    for i in range(len(A)):
        result[i] = -A[i]        

@cuda.jit(device=True)        
def expVec(A, result):
    for i in range(len(A)):
        result[i] = e ** A[i]

@cuda.jit(device=True)        
def subMatVec(A, B, result):
    rowsA = len(A)
    colA = len(A[0])  
    for i in range(rowsA):
        for j in range(colA):
            result[i][j] = A[i][j] - B[j]

@cuda.jit(device=True)
def dotMatVec(A, B, result):
    rowsA = len(A)
    colA = len(A[0])        
    for i in range(rowsA):
        acum = 0.0
        for k in range(colA):
            acum = acum + (A[i][k]*B[k])
        result[i] = acum

@cuda.jit(device=True)        
def dotMats(A, B, result):
    rowsA = len(A)
    colA = len(A[0])
    colB = len(B[0])
    for i in range(rowsA):
        for j in range(colB):
            acum = 0.0
            for k in range(colA):
                acum = acum + (A[i][k]*B[k][j])
            result[i][j] = acum

@cuda.jit(device=True)
def dotVecMat(A, B, result):
    colA = len(A)
    colB = len(B[0])
    for j in range(colB):
        acum = 0.0
        for k in range(colA):
            acum = acum + (A[k]*B[k][j])
        result[j] = acum      

@cuda.jit(device=True)        
def dotVec(A, B):
    acum = 0.0
    for i in range(len(A)):
        acum = acum + (A[i]*B[i])
    return acum

@cuda.jit(device=True)    
def getItems1d(A, ind, result):
    for i in range(len(ind)):
        result[i] = A[ind[i]]

@cuda.jit(device=True)        
def getItems2d(A, ind1, ind2, result):
    for i in range(len(ind1)):
        for j in range(len(ind2)):
            result[i,j] = A[ind1[i],ind2[j]]

@cuda.jit(device=True)        
def subItems1d(A, ind, v):
    for i in ind:
        A[i] = A[i] - v

@cuda.jit(device=True)        
def subItems2d(A, ind1, ind2, v):
    for i in ind1:
        for j in ind2:
            A[i,j] = A[i,j] - v

@cuda.jit(device=True)
def idxFull(length,idx):
    for i in range(length):
        idx[i] = i

@cuda.jit(device=True)        
def copy1d(A, result):
    for i in range(len(A)):
        result[i] = A[i]

@cuda.jit(device=True)        
def copy2d(A, result):
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i,j] = A[i,j]

@cuda.jit(device=True)
def diag(d, matD):
    for i in range(len(d)):
        matD[i][i] = d[i]

@cuda.jit(device=True)        
def linspace(start, stop, num, result):
    num = int(num)
    result[0] = start
    result[num-1] = stop
    step = (stop-start)/(num-1)

    for i in range(1,num-1):
        result[i] = start + (step*i)

@cuda.jit(device=True)       
def SplEv(t, knots, coefs, n):
    tableBS = cuda.local.array(shape=max2d, dtype=float32)
    tableBS = tableBS[0:len(knots),0:n+1]    
        
    for i in range(n):
        k = len(knots)-i-2
        for j in range(i+1):
            BasicBSpline(k, j, t, knots, tableBS)
            
    S = 0.0       
    for i in range(len(knots)-n-1):
        k = len(knots)-n-2-i
        for j in range(n+1):
            BasicBSpline(k, j, t, knots, tableBS)
        S = S + (coefs[k]*tableBS[k][n])
    return S

@cuda.jit(device=True)            
def BasicBSpline(k, n, t, knots, tableBS):
    if n == 0:
        if knots[k] <= t and t < knots[k+1]:
            tableBS[k,n] = 1
        else:
            tableBS[k,n] = 0
    else:
        sub1 = knots[k+n]-knots[k]
        if sub1 == 0.0:
            div1 = 0.0
        else:
            div1 = (t-knots[k])/sub1
          
        sub2 = knots[k+n+1]-knots[k+1]
        if sub2 == 0.0:
            div2 = 0.0
        else:
            div2 = (knots[k+n+1]-t)/sub2

        tableBS[k,n] = (div1*tableBS[k][n-1]) + (div2*tableBS[k+1][n-1])

@cuda.jit(device=True)        
def getSubMat(A, iNot, jNot, subMat):
    iSub = 0
    jSub = 0
    for i in range(len(A)):
        if i != iNot:
            jSub = 0
            for j in range(len(A[0])):
                if j != jNot:
                    subMat[iSub,jSub] = A[i,j]
                    jSub = jSub + 1
            iSub = iSub + 1
                     
@cuda.jit(device=True)                       
def Determinant2x2(A):
    return (A[0,0]*A[1,1])-(A[0,1]*A[1,0])

@cuda.jit(device=True)
def Determinant3x3(A):
    det = 0
    for i in range(len(A[0])):
        subMat = cuda.local.array(shape=max2d, dtype=float32)
        subMat = subMat[0:len(A)-1, 0:len(A[0])-1]  
        getSubMat(A, 0, i, subMat)
        minor = Determinant2x2(subMat)
        det = det + ((-1)**(i))*minor*A[0,i]
    return det

@cuda.jit(device=True)            
def Determinant4x4(A):
    det = 0
    for i in range(len(A[0])):
        subMat = cuda.local.array(shape=max2d, dtype=float32)
        subMat = subMat[0:len(A)-1, 0:len(A[0])-1]  
        getSubMat(A, 0, i, subMat)
        minor = Determinant3x3(subMat)
        det = det + ((-1)**(i))*minor*A[0,i]
    return det        

@cuda.jit(device=True)    
def Determinant5x5(A):
    det = 0
    for i in range(len(A[0])):
        subMat = cuda.local.array(shape=max2d, dtype=float32)
        subMat = subMat[0:len(A)-1, 0:len(A[0])-1]  
        getSubMat(A, 0, i, subMat)
        minor = Determinant4x4(subMat)
        det = det + ((-1)**(i))*minor*A[0,i]
    return det      

@cuda.jit(device=True)
def Inverse2x2(A, inv):
    inv[0,0]=  A[1,1]
    inv[0,1]= -A[0,1]
    inv[1,0]= -A[1,0]
    inv[1,1]=  A[0,0] 
    divValue2d(inv, Determinant2x2(A), inv) 

@cuda.jit(device=True)    
def Inverse3x3(A, inv):
    adjMat = cuda.local.array(shape=max2d, dtype=float32)
    adjMat = adjMat[0:len(A), 0:len(A[0])]  
    det = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            subMat = cuda.local.array(shape=max2d, dtype=float32)
            subMat = subMat[0:len(A)-1, 0:len(A[0])-1]  
            getSubMat(A, i, j, subMat)
            minor = Determinant2x2(subMat)
            adjMat[i,j] = ((-1)**(i+j))*minor
            if i == 0:
                det = det + adjMat[i,j]*A[i,j]
    divValue2d(adjMat.T,det,inv)

@cuda.jit(device=True)    
def Inverse4x4(A, inv):
    adjMat = cuda.local.array(shape=max2d, dtype=float32)
    adjMat = adjMat[0:len(A), 0:len(A[0])]  
    det = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            subMat = cuda.local.array(shape=max2d, dtype=float32)
            subMat = subMat[0:len(A)-1, 0:len(A[0])-1]  
            getSubMat(A, i, j, subMat)
            minor = Determinant3x3(subMat)
            adjMat[i,j] = ((-1)**(i+j))*minor
            if i == 0:
                det = det + adjMat[i,j]*A[i,j]
    divValue2d(adjMat.T,det,inv)    

@cuda.jit(device=True)    
def Inverse5x5(A, inv):
    adjMat = cuda.local.array(shape=max2d, dtype=float32)
    adjMat = adjMat[0:len(A), 0:len(A[0])]  
    det = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            subMat = cuda.local.array(shape=max2d, dtype=float32)
            subMat = subMat[0:len(A)-1, 0:len(A[0])-1]  
            getSubMat(A, i, j, subMat)
            minor = Determinant4x4(subMat)
            adjMat[i,j] = ((-1)**(i+j))*minor
            if i == 0:
                det = det + adjMat[i,j]*A[i,j]
    divValue2d(adjMat.T,det,inv)    
    
    
    
    
    
    
    
    
    

    