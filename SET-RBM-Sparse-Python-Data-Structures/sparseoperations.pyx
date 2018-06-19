# compile this file with: "cythonize -a -i sparseoperations.pyx"
# I have tested this method in Linux (Ubuntu). If you compile it in Windows you may need some work around.

cimport numpy as np

def contrastive_divergence_updates_Cython(double wDecay, double lr, np.ndarray[np.float64_t,ndim=2] DV, np.ndarray[np.float64_t,ndim=2] DH, np.ndarray[np.float64_t,ndim=2] MV, np.ndarray[np.float64_t,ndim=2] MH, np.ndarray[int,ndim=1] rows, np.ndarray[int,ndim=1] cols,np.ndarray[np.float64_t,ndim=1] out):
    cdef:
        size_t i,j
        double s1,s2
    for i in range (out.shape[0]):
        s1=0
        s2=0
        for j in range(DV.shape[0]):
            s1+=DV[j,rows[i]]*DH[j, cols[i]]
            s2+=MV[j,rows[i]]*MH[j, cols[i]]
        out[i]+=lr*(s1/DV.shape[0]-s2/DV.shape[0])-wDecay*out[i]
    #return out
