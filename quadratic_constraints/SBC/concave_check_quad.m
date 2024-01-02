function out = concave_check_quad(H,c,uE,N,nf)

count = 1;
for i = 1:nf
    for j = i+1:nf+1
        % SVD of H
        [svdH(count).U,svdH(count).S,~] = svd(H(:,:,count));
        svdH(count).r = sum(diag(svdH(count).S)>10^-10 * svdH(count).S(1,1));
        svdH(count).U1 = svdH(count).U(:,1:svdH(count).r);
        svdH(count).S1 = svdH(count).S(1:svdH(count).r,1:svdH(count).r);

        % Check whether reference is within the concave domain
        rkc(count) = 0.25 * c(:,:,count)' * svdH(count).U1 * svdH(count).S1^(-1) * svdH(count).U1' * c(:,:,count) - 0.5;

        % Check whether all possible inputs are within concave domain
        b1(count,:) = -c(:,:,count)' * svdH(count).U1 * svdH(count).S1^(-1.5);
        b(count,:) = sign(b1(count,:)) .* b1(count,:);
        a(:,count) = diag(inv(svdH(count).S1));

        fun = @(l) sum((b(count,:)' ./ (2*(l - a(:,count)))).^2) - 0.5;
        l0(count) = fzero(fun,[min(a(:,count) - b(count,:)'.*sqrt(length(b(count,:))/2)) - 10^(-9)*a(1,count), min(a(:,count) - b(count,:)'./sqrt(2)) + 10^(-9)*a(1,count)]);
        q(:,count) = b(count,:)' ./ (2*(l0(count) - a(:,count)));
        g(:,count) = sign(b1(count,:)') .* q(:,count);
        u1(:,count) = svdH(count).U1 * svdH(count).S1^(-0.5) * g(:,count) - 0.5 * svdH(count).U1 * svdH(count).S1^(-1) * svdH(count).U1' * c(:,:,count);

        count = count+1;
    end
end
out = max(rkc,[],'all') < 0 && sqrt(uE * (N-1)) < min(vecnorm(u1));
end