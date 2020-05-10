# compile this file with: "cythonize -a -i sparsebackpropagation.pyx"
# I have tested this method in Linux (Ubuntu). If you compile it in Windows you may need some work around.

cimport cython
cimport numpy as np
ctypedef np.float32_t DTYPE_t

def backpropagation_updates(np.ndarray[np.float32_t,ndim=2] a, np.ndarray[np.float32_t,ndim=2] delta, np.ndarray[int,ndim=1] rows, np.ndarray[int,ndim=1] cols,np.ndarray[np.float32_t,ndim=1] out):
    cdef:
        size_t i,j
        float s
    for i in range (out.shape[0]):
        s=0
        for j in range(a.shape[0]):
            s+=a[j,rows[i]]*delta[j, cols[i]]
        out[i]=s/a.shape[0]
    #return out

