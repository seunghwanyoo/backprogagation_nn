function y = forwardnet(W,b,x,tf)
% y = forwardnet(W,b,x,tf)
% input
%    W: weights
%    b: bias
%    x: input
%    tf: type of transfer function, 0:purelin,1:logsig,2:poslin,3:tansig
% ouput
%    y: output

M = length(W);
for i = 1:M%-1
    n = W{i}*x + b{i};
    switch tf
        case 0
            x = purelin(n);
        case 1
            x = logsig(n); %x = sigmf(n,[1 0]);
        case 2
            x = poslin(n);
        case 3
            x = tansig(n);
    end
end
y = x;
%y = W{M}*x + b{M}; % linear transfer function for ouput