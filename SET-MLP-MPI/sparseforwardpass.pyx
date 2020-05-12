# compile this file with: "cythonize -a -i sparsebackpropagation.pyx"
# I have tested this method in Linux (Ubuntu). If you compile it in Windows you may need some work around.

cimport cython
cimport numpy as np
ctypedef np.float32_t DTYPE_t

def feed_forward(np.ndarray[np.float32_t,ndim=2] a, np.ndarray[np.float32_t,ndim=2] delta, np.ndarray[int,ndim=1] rows, np.ndarray[int,ndim=1] cols,np.ndarray[np.float32_t,ndim=1] out):
    cdef:
        size_t i,j
        float s
        dict a, z, masks

    # w(x) + b
    z = {}

    # activations: f(z)
    a = {1: x}  # First layer has no activations as input. The input x is the input.
    masks = {}

    for i in range(1, 5):
        z[i + 1] = a[i] @ w[i] + b[i]
        a[i + 1] = activations[i + 1].activation(z[i + 1])
        if True:
            if i < 5 - 1:
                a[i + 1], keep_mask = dropout(a[i + 1], 0.3)
                masks[i + 1] = keep_mask

    return z, a, masks

