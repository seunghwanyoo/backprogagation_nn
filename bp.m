% This is a demo script for steepest descent backprogpagation (SDBP)
% algorithm for learning weights of neural network (NN). Two approaches - 
% successive replacement and simultaneous replacement - are implemented in
% the functions, sdbp_successive.m and sdbp_simul.m. For each approaches, 
% momentum is added for better convergence.
% The NN is trained for classification of digits with binary pixel values.
% For training, the ten digits (0~9), each of which is composed of 20 
% pixels (5x4). For testing, two sets of data are used: (1) the ten 
% digits (testing data), (2) noisy ten digit data with corruption rate, p.
%
% Author: Seunghwan Yoo (seunghwanyoo2013@u.northwestern.edu)

close all; clear; 

% data for training (input:X, desired ouput:T)
load('data.mat');

% Initialization for network (#nodes including input & output)
net = [20,10,10,10];
for i = 1:length(net)-1
    W0{i} = randn(net(i+1),net(i));
    b0{i} = randn(net(i+1),1);
end

for i = 1:1
% Parameters
alpha = 0.1; % set high for logsig, low for poslin
gamma = 0.9;%.1;%.1*i
tf = 1;  % activation function, 1:logsig, 2:poslin(relu)
p = 0.1; % percentage of corruption
vis = 1;

% Successive replacement approach
[W1,b1] = sdbp_successive(X,T,net,W0,b0,alpha,gamma,tf);
stat1 = check_performance(W1,b1,X,T,p,tf,vis);

% Simultaneous replacement approach
[W2,b2] = sdbp_simul(X,T,net,W0,b0,alpha,gamma,tf);
stat2 = check_performance(W2,b2,X,T,p,tf,vis);
end
