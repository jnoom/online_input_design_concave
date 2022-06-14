%% Run only once in a MATLAB session (change to the right path)
% pyenv('Version',"path_to_environment\python.exe")

%% 
clear all
Nexp = 1; % number of experiments
N = 3;  % horizon length

for mn = [0:4]
    clearvars -except mn N Nexp 
    for kk=1:Nexp
        clearvars -except kk f Nk N mn hij concave conc t1 Nexp P
        seednumber = str2double([num2str(N) num2str(mn) num2str(kk)]);
        rng(seednumber)
        %% Constraints
        uE = 2; % Maximum input energy
        eps = 0.98;    % Certainty threshold
        NF = 400;   % Maximum number of measurements
        nf = 4;     % Number of fault models
        ccpN = 1;  % Number of initial points of DCCP per time step

        %% Candidate models
        beta = 0.1; % damping
        omega = pi/2; % resonance frequency
        sys = c2d(ss(tf(1,[1 2*beta*omega omega^2])),1);

        A = repmat(sys.A,1,1,nf+1); 
        B = repmat([sys.B [0.5;0]],1,1,nf+1);
        C = repmat([sys.C;[0.1 0.5]],1,1,nf+1);
        E = repmat(eye(2),1,1,nf+1);
        F = repmat(eye(2),1,1,nf+1);

        A(:,:,2) = A(:,:,1) + [0.2 0; 0 0];
        A(:,:,3) = A(:,:,1) + [0.4 0; 0 0];
        A(:,:,4) = A(:,:,1) + [1.0 0; 0 0];
        A(:,:,5) = A(:,:,1) + [1.1 0; 0 0];

        %% Equal DC gain
        for i = 1:nf+1
            B(:,2,i) = B(:,2,i) / (C(1,:,i)/(eye(2) - A(:,:,i))*B(:,2,i));
        end

        %% Bode plots
        % figure;bode(ss(A(:,:,1),B(:,:,1),C(:,:,1),0,1),ss(A(:,:,2),B(:,:,2),C(:,:,2),0,1),ss(A(:,:,3),B(:,:,3),C(:,:,3),0,1),ss(A(:,:,4),B(:,:,4),C(:,:,4),0,1),ss(A(:,:,5),B(:,:,5),C(:,:,5),0,1))
        % legend('$i=0$','$i=1$','$i=2$','$i=3$','$i=4$','Interpreter','latex')
        % ylabel('Magnitude','Interpreter','latex');
        % xlabel('Frequency','Interpreter','latex');
        % title('')

        %% Noise
        R = 80*eye(2);              % Measurement noise > 0
        Q = 0.2*eye(2);              % Process noise >= 0

        %% Initial condition
        nx = length(A(:,1,1));
        ny = length(C(:,1,1));
        nu = length(B(1,:,1));
        nv = length(F(1,:,1));
        nw = length(E(1,:,1));
        x  = zeros(nx,NF);
        xh = zeros(nx,NF,nf+1);
        y  = zeros(ny,NF);
        yh = zeros(ny,NF,nf+1);
        u  = zeros(nu,NF+1);
        S  = zeros(ny,ny,NF,nf+1);
        K  = zeros(nx,ny,NF,nf+1);
        P  = zeros(NF+1,nf+1,Nexp);
        Sigma = zeros(nx,nx,NF,nf+1);
        Sigmat = zeros(N*ny,N*ny,nf+1);

        P(1,1,:) = 0.2;
        P(1,2:nf+1,:) = (1 - P(1,1,1)) / nf;
        u(:,1) = 0;

        for i=1:nf+1
            xh(:,1,i) = [0;1];
            Sigma(:,:,1,i) = 0.5*eye(nx);
        end
        x(:,1) = xh(:,1,1) + [sqrt(Sigma(1,1,1,1))*randn(1,1); sqrt(Sigma(2,2,1,1))*randn(1,1)];

        %% N-step expanded models
        At = zeros(N*nx,nx,nf+1);
        Bt = zeros(N*nx,N*nu,nf+1);
        Ct = zeros(N*ny,N*nx,nf+1);
        Et = zeros(N*nx,N*nw,nf+1);
        Ft = zeros(N*ny,N*nv,nf+1);
        for k = 1:nf+1
            for i = 1:N
                for j = 1:i
                    Bt((i-1)*nx+1:(i)*nx,(j-1)*nu+1:j*nu,k) = A(:,:,k)^(i-j) * B(:,:,k);
                    Et((i-1)*nx+1:(i)*nx,(j-1)*nw+1:j*nw,k) = A(:,:,k)^(i-j) * E(:,:,k);
                end
                At((i-1)*nx+1:i*nx,1:nx,k) = A(:,:,k)^(i);
            end
            Ft(:,:,k) = kron(eye(N),F(:,:,k));
            Ct(:,:,k) = kron(eye(N),C(:,:,k));
        end

        Qt = kron(eye(N),Q);
        Rt = kron(eye(N),R);

        %% Online
        tic
        for k = 1:NF
            if k>1
                x(:,k) = A(:,:,mn+1)*x(:,k-1) + B(:,:,mn+1)*u(:,k-1) + E(:,:,mn+1)*Q^0.5*randn(nw,1);  % Update system state
            end
            y(:,k)   = C(:,:,mn+1)*x(:,k) + F(:,:,mn+1)*R^0.5*randn(nv,1);  % Obtain measurement
            psum = 0;
            for i = 1:nf+1
                yh(:,k,i) = C(:,:,i) * xh(:,k,i);  % Predicted output
                S(:,:,k,i) = C(:,:,i) * Sigma(:,:,k,i) * C(:,:,i)'  +  F(:,:,i) * R * F(:,:,i)';  % Covariance corresponding to predicted ouput
                psum = psum + mvnpdf(y(:,k), C(:,:,i)*xh(:,k,i), real(S(:,:,k,i))) * P(k,i,kk);  % Denominator of Bayesian update rule
            end
            for i = 1:nf+1
                P(k+1,i,kk) = mvnpdf(y(:,k), C(:,:,i)*xh(:,k,i), real(S(:,:,k,i)))  *  P(k,i,kk)  /  psum;  % Bayes update rule
                if P(k+1,i,kk) > eps  % Terminate when accuracy eps is achieved
                    flag = 1;
                    break
                end

                K(:,:,k,i) = Sigma(:,:,k,i) * C(:,:,i)' / S(:,:,k,i);  % Kalman gain
                xh(:,k,i) = xh(:,k,i)  +  K(:,:,k,i) * (y(:,k) - C(:,:,i)*xh(:,k,i));  % Filtered state
                Sigma(:,:,k,i) = (eye(nx) - K(:,:,k,i)*C(:,:,i)) * Sigma(:,:,k,i);  % Covariance corresponding to filtered state 

                Sigmat(:,:,i) = Ct(:,:,i) * At(:,:,i) * Sigma(:,:,k,i) * At(:,:,i)' * Ct(:,:,i)'...
                    +  Ct(:,:,i) * Et(:,:,i) * Qt * Et(:,:,i)' * Ct(:,:,i)'...
                    +  Ft(:,:,i) * Rt * Ft(:,:,i)';  % Covariance corresponding to predicted output sequence
            end
            if flag == 1
                break
            end
            
            % Calculate Bhattacharyya coefficients
            Gamma = zeros(N*ny,N*nu);
            Gamma2= zeros(size(Ct(:,:,1)*At(:,:,1)*xh(:,1,1)));
            Omega = zeros(size(Sigmat));
            H     = zeros(N*nu,N*nu);
            c     = zeros(N*nu,1);
            h     = 0;
            count = 1;
            for i = 1:nf
                for j = i+1:nf+1
                    Gamma(:,:,count) = Ct(:,:,i) * Bt(:,:,i) - Ct(:,:,j) * Bt(:,:,j);
                    Gamma2(:,:,count)= Ct(:,:,i) * At(:,:,i) * xh(:,k,i) - Ct(:,:,j) * At(:,:,j) * xh(:,k,j);
                    Omega(:,:,count) = Sigmat(:,:,i) + Sigmat(:,:,j);
                    H(:,:,count)     = 0.25 * Gamma(:,:,count)' / Omega(:,:,count) * Gamma(:,:,count);
                    c(:,:,count)     = 0.5 * Gamma(:,:,count)' / Omega(:,:,count)  * Gamma2(:,:,count);
                    h(count)        = 0.25 * Gamma2(:,:,count)' / Omega(:,:,count) * Gamma2(:,:,count)...
                                      + 0.5 * log( det(0.5*Omega(:,:,count)) / ...
                                     sqrt(det(Sigmat(:,:,i))*det(Sigmat(:,:,j))) );

                    % SVD of H (Check for concavity)
                    [svdH(count).U,svdH(count).S,~] = svd(H(:,:,count));
                    svdH(count).r = sum(diag(svdH(count).S)>10^-10 * svdH(count).S(1,1));
                    svdH(count).U1 = svdH(count).U(:,1:svdH(count).r);
                    svdH(count).S1 = svdH(count).S(1:svdH(count).r,1:svdH(count).r);


                    % Check whether all possible inputs are within concave domain
                    b1(count,:) = -c(:,:,count)' * svdH(count).U1 * svdH(count).S1^(-1.5);
                    b(count,:) = sign(b1(count,:)) .* b1(count,:);
                    a(:,count) = diag(inv(svdH(count).S1));

                    fun = @(l) sum((b(count,:)' ./ (2*(l - a(:,count)))).^2) - 0.5;
                    l0(count) = fzero(fun,round(a(1,count))-0.000001*round(a(1,count)));
                    q(:,count) = b(count,:)' ./ (2*(l0(count) - a(:,count)));
                    v(:,count) = sign(b1(count,:)') .* q(:,count);
                    u1(:,count) = svdH(count).U1 * svdH(count).S1^(-0.5) * v(:,count) - 0.5 * svdH(count).U1 * svdH(count).S1^(-1) * svdH(count).U1' * c(:,:,count);

                    count = count+1;
                end
            end

            concave(kk,k) = sqrt(uE * N) < min(vecnorm(u1));

            % Call Python script for DCCP optimization
            u_opt = double(py.dccp_BC.minsumbhat(py.numpy.array(permute(H,[3,2,1])),py.numpy.array(reshape(permute(c,[3,2,1]),[count-1,N*nu])),py.numpy.array(h),py.numpy.array(P(k+1,:,kk)),uE,nu,ccpN,str2double([num2str(seednumber) num2str(k)])));
            u(:,k) = u_opt(1:nu);
            for i = 1:nf+1
                xh(:,k+1,i) = A(:,:,i) * xh(:,k,i)  +  B(:,:,i) * u(:,k);  % Predicted state
                Sigma(:,:,k+1,i) = A(:,:,i) * Sigma(:,:,k,i) * A(:,:,i)'  +  E(:,:,i) * Q * E(:,:,i)';  % Covariance corresponding to predicted state
            end    
        end
        t1(kk)=toc;
        conc(kk) = min(concave(kk,1:k-1));

        y(:,k+1:end) = NaN;
        u(:,k+1:end) = NaN;
        P(k+2:end,:,kk) = NaN;
        
        %% Figure (uncomment only if Nexp is small)
        figure;
        subplot(3,1,1);plot(1:NF,y(:,:));xlim([1 200]);ylim([-15 15]);ylabel('$y_k$','Interpreter','latex')
        subplot(3,1,2);plot(1:NF,u(:,1:end-1)');xlim([1 200]);ylim([-3 3]);ylabel('$u_k$','Interpreter','latex')
        legend('$u_{k,1}$','$u_{k,2}$','Interpreter','latex')
        subplot(3,1,3);plot(1:NF,P(1:end-1,:));xlim([1 200]);ylim([0 1]);ylabel('$P(M_i)$','Interpreter','latex');
        xlabel('$k$','Interpreter','latex')
        legend('$i=0$','$i=1$','$i=2$','$i=3$','$i=4$','Interpreter','latex');

        %% save variables
        if k == NF
            [~,i] = max(P(end,:,kk));
        end
        if i ~= mn+1
            f(kk) = 1;
        else
            f(kk) = 0;
        end
        Nk(kk) = k;
    end
    acc = 1 - sum(f)/kk;
    Nkm = mean(Nk);
    save(['clamd_quadratic_BC_' num2str(mn)])
end

