cimport numpy as np

def backpropagation_updates_Cython(np.ndarray[np.float64_t,ndim=2] a, np.ndarray[np.float64_t,ndim=2] delta, np.ndarray[int,ndim=1] rows, np.ndarray[int,ndim=1] cols,np.ndarray[np.float64_t,ndim=1] out):
    cdef:
        size_t i,j
        double s
    for i in range (out.shape[0]):
        s=0
        for j in range(a.shape[0]):
            s+=a[j,rows[i]]*delta[j, cols[i]]
        out[i]=s/a.shape[0]
    #return out

