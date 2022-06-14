function [c, ceq] = quadcon(x,uE,nu)
N = round(length(x)/nu);
for i = 1:N
    c(i) = x((i-1)*nu+1:i*nu)'*x((i-1)*nu+1:i*nu) - uE;
end
ceq = [];
end