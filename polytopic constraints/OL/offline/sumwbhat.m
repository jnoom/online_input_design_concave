% Function for calculating sum of weighted Bhattacharrya coefficients
% P: 1-D array of model probabilities
% H: 3-D array for pairwise Bhat. coef.
% c: 2-D array for pairwise Bhat. coef.
% h: 1-D array for pairwise Bhat. coef.
% u: 1-D array describing the system input

function out = sumwbhat(P,H,c,h,u)
    count = 1;
    out = 0;
    nf=length(P)-1;
    for j2 = 1:nf
        for j = j2+1:nf+1
            out = out + sqrt(P(j2)*P(j)) * exp(-u' * H(:,:,count) * u  -  c(:,:,count)' * u  -  h(count));
            count = count+1;
        end
    end
end