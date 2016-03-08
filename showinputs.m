function showinputs(X)

[M,N] = size(X);
figure,
for i = 1:N
    subplot(2,5,i);
    x = reshape(X(:,i),5,4);
    imshow(x,'InitialMagnification',300);
end