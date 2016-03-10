# Backpropagation for neural network
This is a demo script for steepest descent backprogpagation (SDBP) algorithm for learning weights of neural network (NN). Two approaches - successive replacement and simultaneous replacement - are implemented in the functions, sdbp_successive.m and sdbp_simul.m. The objective functions are mean square error (MSE), and, as the name tells, the steepest descent algorithm is used for optimization for weight update. Also, a momentum parameter is used for better convergence, so you can call it backpropagation with momentum (MOBP). </br></br>
Here, the neural network is trained for classification of digits. The data is 5x4 pixel image with binary values as shown in Figure 1 and Figure 2. For training, the ten digit images (0~9) in Figure 1 are used. For testing, two data sets are used: (1) the ten digits in Figure 1 (training data), (2) noisy ten digit data with corruption rate, p, like Figure 2 (in case of p = 0.1).

![alt tag](https://github.com/seunghwanyoo/backprogagation_nn/blob/master/digits.jpg)
</br>Figure 1. 10 digits used for training data (also for testing)

![alt tag](https://github.com/seunghwanyoo/backprogagation_nn/blob/master/digits_corrupted.jpg)
</br>Figure 2. Corrupted data with 10% rate, used for the second test set
</br></br>
You can modify the network architecture - the number of layers and nodes - by changing `net`. Also, you can set the step length, `alpha`, the momentum, `gamma`, corruption rate, `p`, and the activation function type, `tf`.


# Description of file
- bp.m: test script for digit recognition
- sdbp_successive.m: function for SDBP with successive replacement
- sdbp_simul.m: function for SDBP with simultaneous replacement

# Contact 
Seunghwan Yoo (seunghwanyoo2013@u.northwestern.edu)
