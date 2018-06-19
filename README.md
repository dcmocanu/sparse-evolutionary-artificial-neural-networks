# sparse-evolutionary-artificial-neural-networks
* Proof of concept implementations of various sparse artificial neural network models trained with the Sparse Evolutionary Training (SET) procedure.  
* The following implementations are distributed in the hope that they may be useful, but without any warranties; Their use is entirely at the user's own risk.

###### Implementation 1 - SET-MLP with Keras and Tensorflow (SET-MLP-Keras-Weights-Mask)

* Proof of concept implementation of Sparse Evolutionary Training (SET) for Multi Layer Perceptron (MLP) on CIFAR10 using Keras and a mask over weights.  
* This implementation can be used to test SET in varying conditions, using the Keras framework versatility, e.g. various optimizers, activation layers, tensorflow.  
* Also it can be easily adapted for Convolutional Neural Networks or other models which have dense layers.  
* However, due the fact that the weights are stored in the standard Keras format (dense matrices), this implementation can not scale properly.  
* If you would like to build an SET-MLP with over 100000 neurons, please use Implementation 2.

###### Implementation 2 - SET-MLP using just sparse data structures from pure Python 3 (SET-MLP-Sparse-Python-Data-Structures)

* Proof of concept implementation of Sparse Evolutionary Training (SET) for Multi Layer Perceptron (MLP) on lung dataset using Python, SciPy sparse data structures, and (optionally) Cython.  
* This implementation can be used to create SET-MLP with hundred of thousands of neurons on a standard laptop. It was made starting from the vanilla fully connected MLP implementation of Ritchie Vink (https://www.ritchievink.com/) and we would like to acknowledge his work and thank him. Also, we would like to thank Thomas Hagebols for analyzing the performance of SciPy sparse matrix operations.   
* If you would like to try out SET-MLP with various activation functions, optimization methods and so on (in the detriment of scalability) please use Implementation 1.  

###### Implementation 3 - SET-RBM using just sparse data structures from pure Python 3 (SET-RBM-Sparse-Python-Data-Structures)

* Proof of concept implementation of Sparse Evolutionary Training (SET) for Restricted Boltzmann Machine (RBM) on COIL20 dataset using Python, SciPy sparse data structures, and (optionally) Cython.  
* This implementation can be used to create SET-RBM with hundred of thousands of neurons on a standard laptop.

###### References

For an easy understanding of these implementations please read the following articles. Also, if you use parts of this code in your work, please cite the next articles:

@article{Mocanu2018SET,
  author =        {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
  journal =       {Nature Communications},
  title =         {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
  year =          {2018},
  doi =           {10.1038/s41467-018-04316-3}
}

@article{Mocanu2016XBM,
author={Mocanu, Decebal Constantin and Mocanu, Elena and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
title={A topological insight into restricted Boltzmann machines},
journal={Machine Learning},
year={2016},
volume={104},
number={2},
pages={243--270},
doi={10.1007/s10994-016-5570-z},
url={https://doi.org/10.1007/s10994-016-5570-z}
}

@phdthesis{Mocanu2017PhDthesis,
title = {Network computations in artificial intelligence},
author = {D.C. Mocanu},
year = {2017},
isbn = {978-90-386-4305-2},
publisher = {Eindhoven University of Technology}
}