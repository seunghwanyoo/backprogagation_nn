# Backpropagation for neural network
This is a demo script for steepest descent backprogpagation (SDBP) algorithm for learning weights of neural network (NN). Two approaches - successive replacement and simultaneous replacement - are implemented in the functions, sdbp_successive.m and sdbp_simul.m. For each approaches, momentum is added for better convergence. </br></br>
Here, the NN is trained for classification of digits. The digit data is binary pixel values of size 5x4 as the image below. For training, the ten digits (0~9), each of which is composed of 20 pixels (5x4). For testing, two sets of data are used: (1) the ten digits (testing data), (2) noisy ten digit data with corruption rate, p.


# Description of file
- bp.m: test script for digit recognition
- sdbp_successive.m: function for SDBP with successive replacement
- sdbp_simul.m: function for SDBP with simultaneous replacement

# Contact 
Seunghwan Yoo (seunghwanyoo2013@u.northwestern.edu)
