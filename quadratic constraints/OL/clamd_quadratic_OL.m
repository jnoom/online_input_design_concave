clear all
Nexp = 1;
N = 400;

for mn = [0:4]
    clearvars -except mn N Nexp
    for kk=1:Nexp
        clearvars -except kk f Nk N mn concave conc t1 Nexp P
        seednumber = str2double([num2str(N) num2str(mn) num2str(kk)]);
        rng(seednumber)
        %% Constraints
        uE = 2; % Maximum input energy
        eps = 0.98;  % Certainty threshold
        NF = 400;  % Maximum number of measurements
        nf = 4;  % Number of fault models

        %% Candidate models
        beta = 0.1;  % damping
        omega = pi/2;  % resonance frequency
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
        R = 80*eye(2);  % Measurement noise > 0
        Q = 0.2*eye(2);  % Process noise >= 0

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
        S  = zeros(ny,ny,NF,nf+1);
        K  = zeros(nx,ny,NF,nf+1);
        P  = zeros(NF+1,nf+1,Nexp);
        Sigma = zeros(nx,nx,NF,nf+1);
        Sigmat = zeros(N*ny,N*ny,nf+1);

        P(1,1,:) = 0.2;
        P(1,2:nf+1,:) = (1 - P(1,1,1)) / nf;

        for i=1:nf+1
            xh(:,1,i) = [0;1];
            Sigma(:,:,1,i) = 0.5*eye(nx);
        end
        x(:,1) = xh(:,1,1) + [sqrt(Sigma(1,1,1,1))*randn(1,1); sqrt(Sigma(2,2,1,1))*randn(1,1)];

        %% Offline computation of OL solution
        load('OL_solution','u');

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
            end
            if flag == 1
                break
            end            
            for i = 1:nf+1
                xh(:,k+1,i) = A(:,:,i) * xh(:,k,i)  +  B(:,:,i) * u(:,k);  % Predicted state
                Sigma(:,:,k+1,i) = A(:,:,i) * Sigma(:,:,k,i) * A(:,:,i)'  +  E(:,:,i) * Q * E(:,:,i)';  % Covariance corresponding to predicted state
            end    
        end
        t1(kk)=toc;
        
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
    save(['clamd_quadratic_OL_' num2str(mn)])
end

