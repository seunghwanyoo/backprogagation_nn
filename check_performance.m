function stat = check_performance(W,b,X,T,p,tf,vis)
% check_performance
% Input:
%   W: trained weights
%   b: trained biases
%   X: ten digit data
%   T: ground truth of X
%   p: corruption rate
%   tf: transfer function
%   vis: visual option
% Output:
%   stat: results with two test sets
%
% Author: Seunghwan Yoo (seunghwanyoo2013@u.northwestern.edu)

fprintf(' (1) Performance check with training data \n');
cnt = 0;
[M,N] = size(X);
for i = 1:N
    x = X(:,i);
    t = T(:,i);
    ans = find(t==1);
    y = forwardnet(W,b,x,tf);
    [~,ind] = max(y);
    if ind == ans
        cnt = cnt + 1;
        fprintf(' %i-%i (O),',i,ind);
    else
        fprintf(' %i-%i (X),',i,ind);
    end
end
stat.test1 = cnt / N;
fprintf('\n => accuracy: %.2f\n',stat.test1);

fprintf(' (2) Performance check with noisy data (%.2f corruption)\n',p);
X1 = zeros(size(X));
cnt = 0;
rng(1);
for i = 1:N
    rp = randperm(M);
    icorrupt = rp(1:floor(p*M));
    x = X(:,i);
    x(icorrupt) = ~x(icorrupt);
    X1(:,i) = x;
    t = T(:,i);
    ans = find(t==1);
    y = forwardnet(W,b,x,tf);
    [~,ind] = max(y);
    if ind == ans
        cnt = cnt + 1;
        fprintf(' %i-%i (O),',i,ind);
    else
        fprintf(' %i-%i (X),',i,ind);
    end
end
stat.test2 = cnt / N;
fprintf('\n => accuracy: %.2f\n',stat.test2);

% show the input data if vis == 1
if vis
    showinputs(X);
    showinputs(X1);
end

function x = randomflip(x,p)
n = length(x);
x1 = rand(n,1);
for i = 1:n
    if x1(i) <= p
        x(i) = 1 - x(i);
    end
end

function showinputs(X)
[M,N] = size(X);
figure,
for i = 1:N
    subplot(2,5,i);
    x = reshape(X(:,i),5,4);
    imshow(x,'InitialMagnification',300);
end
