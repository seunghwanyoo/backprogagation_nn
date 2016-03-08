function [W,b] = sdbp_simul(X,T,net,W0,b0,alpha,gamma,tf)
% steepest descent backpropagation method with simultaneous replacement
% Input
%   (X,T): trainig data of size N (N columns)
%   W0: initial weights
%   b0: inital bias
%   alpha: learning rate
%   gamma: momentum term
%   tf: transfer function/activation function
% Output
%   W (weights):     W{1:M} = W1:WM
%   b (bias term):   b{1:M} = b1:bM
% Variables.
%   M: number of layers
%   a (node output): a{1} = a0, a{2:M+1} = a1:aM
%   W (weights):     W{1:M} = W1:WM
%   b (bias term):   b{1:M} = b1:bM
%   n (net input):   n{1:M} = n1:nM
%   s (sensitivity): s{1:M} = s1:sM
%
% Author: Seunghwan Yoo (seunghwanyoo2013@u.northwestern.edu)

tic;
W = W0;
b = b0;
M = length(net)-1;
N = size(X,2);
s = cell(M,1);
Fj = cell(M,1);
dW0 = cell(M,1);
db0 = cell(M,1);
for i=1:M
    s{i} = zeros(net(i+1),1);
    Fj{i} = zeros(net(i+1));
    dW0{i} = zeros(net(i+1),net(i));
    db0{i} = zeros(net(i+1),1);
end
dW = cell(M,1);
db = cell(M,1);

tol = 10^-5;
max_epoch = 100000;
fprintf('\nSDBP simultaneous, training');
for ii = 1:max_epoch
    W_old = W;
    b_old = b;
    for i = 1:M
        dW{i} = zeros(net(i+1),net(i));
        db{i} = zeros(net(i+1),1);
    end
    if mod(ii,1000) == 0
        fprintf('.');
    end
    
    for k = 1:N
        x = X(:,k);
        t = T(:,k);
        
        % Forward propagation
        a{1} = x;
        for i = 1:M%-1
            n{i} = W{i}*a{i} + b{i};
            if tf == 1
                a{i+1} = logsig(n{i});
            elseif tf == 2
                a{i+1} = poslin(n{i}); %sigmf(n{i},[1 0]);
            end
        end
        %a{M+1} = W{M}*a{M} + b{M};

        % Backpropagation
        for i = 1:M
            for j = 1:net(i+1)
                if tf == 1
                    Fj{i}(j,j) = (1-a{i+1}(j))*a{i+1}(j); % for logsig
                elseif tf == 2
                    Fj{i}(j,j) = (a{i+1}(j) >= 0);
                end
            end
        end
        s{M} = -2*Fj{M}*(t-a{M+1});
        for i = M-1:-1:1
            s{i} = Fj{i}*W{i+1}'*s{i+1};
        end
        for i = 1:M
            dW{i} = dW{i} + s{i}*a{i}';
            db{i} = db{i} + s{i};
        end
    end
    
    % Weight updates (MOBP)
    for i = 1:M
%         W{i} = W{i} - alpha/N*dW{i};
%         b{i} = b{i} - alpha/N*db{i};
        dW0{i} = gamma*dW0{i} - (1-gamma)*alpha/N*dW{i};
        db0{i} = gamma*db0{i} - (1-gamma)*alpha/N*db{i};
        W{i} = W{i} + dW0{i};
        b{i} = b{i} + db0{i};
    end
    
    % Termination
    flag = 0;
    for i = 1:M
        if norm(W{i}-W_old{i})/norm(W_old{i}) > tol
            flag = 1;
            break;
        end
        if norm(b{i}-b_old{i})/norm(b_old{i}) > tol
            flag = 1;
            break;
        end
    end
    if flag == 0
        fprintf('\n Converged at %ith epochs!\n',ii);
        break;
    end
end
if flag == 1
    fprintf(' Stopped at the max epoch (%i)\n',max_epoch);
end
toc;