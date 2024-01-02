% This script generates results for the closed-loop approach "Bhattacharyya
% Coefficient" (BC) in a quadratic constraint set. It requires the Python 
% script "dccp_BC.py" in the Current Folder and produces a .mat-file for
% each true candidate model containing the generated simulation data. 

%% Run only once in a MATLAB session (change to the right path)
% pyenv('Version',"path_to_environment\python.exe")

%% 
clear all
Nexp = 1; % number of experiments
Npexp= 0; % number of previous experiments
N = 3;  % horizon length

for mn = [0:4]  % Actual model (0 if fault-free,1:4 if fault)
    clearvars -except mn N Nexp Npexp
    for kk=1:Nexp  % Experiment number
        clearvars -except kk f Nk N mn hij concave conc t1 Nexp Npexp P
        seednumber = str2double([num2str(N) num2str(mn) num2str(kk+Npexp)]);
        rng(seednumber)
        %% Constraints
        uE = 2;         % Maximum input energy
        eps = 0.98;     % Certainty threshold
        NF = 400;       % Maximum number of measurements
        nf = 4;         % Number of fault models
        ccpN = 1;       % Number of initial points of DCCP per time step

        %% Candidate models
        beta = 0.1; % damping
        omega = pi/2; % resonance frequency
        sys = c2d(ss(tf(1,[1 2*beta*omega omega^2])),1);

        A = repmat(sys.A,1,1,nf+1); 
        B = repmat([sys.B [0.5;0]],1,1,nf+1); 
        C = repmat([sys.C;[0.1 0.5]],1,1,nf+1);

        A(:,:,2) = A(:,:,1) + [0.2 0; 0 0];
        A(:,:,3) = A(:,:,1) + [0.4 0; 0 0];
        A(:,:,4) = A(:,:,1) + [1.0 0; 0 0];
        A(:,:,5) = A(:,:,1) + [1.1 0; 0 0];

        nx = length(A(:,1,1));
        ny = length(C(:,1,1));
        nu = length(B(1,:,1));
        nv = ny;
        nw = nx;

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
        S = zeros(2);

        %% N-step expanded models
        At = zeros((N)*nx,nx,nf+1);
        Bt = zeros((N)*nx,(N-1)*nu,nf+1);
        Ct = zeros((N)*ny,(N)*nx,nf+1);
        TA = zeros((N)*nx,(N)*nw,nf+1);
        for k = 1:nf+1
            for i = 1:N
                At((i-1)*nx+1:i*nx,1:nx,k) = A(:,:,k)^(i-1);
            end
            for i = 1:N-1
                for j = 1:i
                    Bt((i)*nx+1:(i+1)*nx,(j-1)*nu+1:j*nu,k) = A(:,:,k)^(i-j) * B(:,:,k);
                    TA((i)*nx+1:(i+1)*nx,(j-1)*nw+1:j*nw,k) = A(:,:,k)^(i-j);
                end
            end
            Ct(:,:,k) = kron(eye(N),C(:,:,k));
        end

        Qt = kron(eye(N),Q);
        Rt = kron(eye(N),R);
        St = [zeros(nw,(N-1)*nv), zeros(nw,nv)
            kron(eye(N-1),S), zeros((N-1)*nw,nv)];

        %% Initial condition
        x  = zeros(nx,NF);
        xh = zeros(nx,NF,nf+1);
        y  = zeros(ny,NF);
        yh = zeros(ny,NF,nf+1);
        u  = zeros(nu,NF+1);
        Sm  = zeros(ny,ny,NF,nf+1);
        K  = zeros(nx,ny,NF,nf+1);
        P(:,:,kk)  = zeros(NF+1,nf+1);
        Sigma = zeros(nx,nx,NF,nf+1);
        Sigmat = zeros(N*ny,N*ny,nf+1);

        P(1,1,kk) = 0.2;
        P(1,2:nf+1,kk) = (1 - P(1,1,kk)) / nf;
        u(:,1) = 0;

        for i=1:nf+1
            xh(:,1,i) = [0;1];
            Sigma(:,:,1,i) = 0.5*eye(nx);
        end
        x(:,1) = xh(:,1,1) + [sqrt(Sigma(1,1,1,1))*randn(1,1); sqrt(Sigma(2,2,1,1))*randn(1,1)];

        %% Online
        tic
        for k = 1:NF
            if k>1
                x(:,k) = A(:,:,mn+1)*x(:,k-1) + B(:,:,mn+1)*u(:,k-1) + Q^0.5*randn(nw,1);  % Update system state
            end
            y(:,k)   = C(:,:,mn+1)*x(:,k) + R^0.5*randn(nv,1);  % Obtain measurement
            psum = 0;
            for i = 1:nf+1
                yh(:,k,i) = C(:,:,i) * xh(:,k,i);  % Predicted output
                Sm(:,:,k,i) = C(:,:,i) * Sigma(:,:,k,i) * C(:,:,i)'  +  R;  % Covariance corresponding to predicted ouput
                psum = psum + mvnpdf(y(:,k), C(:,:,i)*xh(:,k,i), real(Sm(:,:,k,i))) * P(k,i,kk);  % Denominator of Bayesian update rule
            end
            for i = 1:nf+1
                P(k+1,i,kk) = mvnpdf(y(:,k), C(:,:,i)*xh(:,k,i), real(Sm(:,:,k,i)))  *  P(k,i,kk)  /  psum;  % Bayes update rule
                if P(k+1,i,kk) > eps  % Terminate when accuracy eps is achieved
                    flag = 1;
                    break
                end

                K(:,:,k,i) = (S + A(:,:,i)* Sigma(:,:,k,i) * C(:,:,i)') / Sm(:,:,k,i);  % Kalman gain
                xh(:,k+1,i) = A(:,:,i) * xh(:,k,i)  +  B(:,:,i) * u(:,k)...
                    +  K(:,:,k,i) * (y(:,k) - C(:,:,i)*xh(:,k,i));  % Predicted state
                Sigma(:,:,k+1,i) = A(:,:,i) * Sigma(:,:,k,i) * A(:,:,i)'  +  Q...
                    -  K(:,:,k,i) * (S' + C(:,:,i) * Sigma(:,:,k,i) * A(:,:,i)');  % Covariance corresponding to predicted state

                Sigmat(:,:,i) = Ct(:,:,i) * At(:,:,i) * Sigma(:,:,k+1,i) * At(:,:,i)' * Ct(:,:,i)'...
                    +  Ct(:,:,i) * TA(:,:,i) * Qt * TA(:,:,i)' * Ct(:,:,i)'...
                    +  Ct(:,:,i) * TA(:,:,i) * St  +  St' * TA(:,:,i)' * Ct(:,:,i)'  +  Rt;  % Covariance corresponding to predicted output sequence
            end
            if flag == 1
                break
            end

            % Calculate parameters for Bhattacharyya coefficients
            Gamma = zeros(N*ny,(N-1)*nu);
            Gamma2= zeros(size(Ct(:,:,1)*At(:,:,1)*xh(:,1,1)));
            Omega = zeros(size(Sigmat));
            H     = zeros((N-1)*nu,(N-1)*nu);
            c     = zeros((N-1)*nu,1);
            h     = 0;
            count = 1;
            for i = 1:nf
                for j = i+1:nf+1
                    Gamma(:,:,count) = Ct(:,:,i) * Bt(:,:,i) - Ct(:,:,j) * Bt(:,:,j);
                    Gamma2(:,:,count)= Ct(:,:,i) * At(:,:,i) * xh(:,k+1,i) - Ct(:,:,j) * At(:,:,j) * xh(:,k+1,j);
                    Omega(:,:,count) = Sigmat(:,:,i) + Sigmat(:,:,j);
                    H(:,:,count)     = 0.25 * Gamma(:,:,count)' / Omega(:,:,count) * Gamma(:,:,count);
                    c(:,:,count)     = 0.5 * Gamma(:,:,count)' / Omega(:,:,count)  * Gamma2(:,:,count);
                    h(count)         = 0.25 * Gamma2(:,:,count)' / Omega(:,:,count) * Gamma2(:,:,count)...
                                      + 0.5 * log( det(0.5*Omega(:,:,count)) / ...
                                     sqrt(det(Sigmat(:,:,i))*det(Sigmat(:,:,j))) );
                    count = count+1;
                end
            end

            concave(kk,k) = concave_check_quad(H,c,uE,N,nf);

            % Call Python script for DCCP optimization
            u_opt = double(py.dccp_BC.minsumbhat(py.numpy.array(permute(H,[3,2,1])),...
                py.numpy.array(reshape(permute(c,[3,2,1]),[count-1,(N-1)*nu])),...
                py.numpy.array(h),py.numpy.array(P(k+1,:,kk)),uE,nu,ccpN,...
                str2double([num2str(seednumber) num2str(k)])));
            u(:,k+1) = u_opt(1:nu);
        end
        t1(kk)=toc;
        conc(kk) = min(concave(kk,1:k-1));

        %% Figure
        y(:,k+1:end) = NaN;
        u(:,k+1:end) = NaN;
        P(k+2:end,:,kk) = NaN;

        % figure;
        % subplot(3,1,1);plot(1:NF,y(:,:));xlim([1 k]);ylim([-15 15]);ylabel('$y_k$','Interpreter','latex')
        % subplot(3,1,2);plot(1:NF,u(:,1:end-1)');xlim([1 k]);ylim([-3 3]);ylabel('$u_k$','Interpreter','latex')
        % subplot(3,1,3);plot(1:NF,P(1:end-1,:));xlim([1 k]);ylim([0 1]);ylabel('$P(M_i)$','Interpreter','latex');
        % xlabel('$k$','Interpreter','latex')

        if k == NF
            [~,i] = max(P(end,:));
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
    save(['clafd_quadratic_BC_' num2str(Npexp) '_' num2str(N) '_' num2str(mn)])

end

         