function out = concave_check_poly(H,c,V1)

for j = 1:length(H(1,1,:))
    [U,S,~] = svd(H(:,:,j));
    r = sum(diag(S)>10^-10 * S(1,1));
    U1 = U(:,1:r);
    S1 = S(1:r,1:r);
    for i = 1:length(V1(1,:))
        conc(i,j) = V1(:,i)' * H(:,:,j) * V1(:,i)  +  c(:,:,j)' * V1(:,i)  +  0.25 * c(:,:,j)' * U1 / S1 * U1' * c(:,:,j) < 0.5;
    end
end
out = min(min(conc));
    
end