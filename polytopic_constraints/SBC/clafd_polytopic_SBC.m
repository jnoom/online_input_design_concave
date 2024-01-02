% This script generates results for the closed-loop approach "Summed 
% Bhattacharyya Coefficient" (SBC) in a polytopic constraint set. It
% produces a .mat-file for each true candidate model containing the
% generated simulation data. 

%%
clear all
warning('off')
Nexp = 1; % number of experiments
Npexp= 0; % number of previous experiments
N = 5;    % horizon length

for mn = [0:4]  % Actual model (0 if fault-free,1:4 if fault)
    clearvars -except mn N Nexp Npexp
    for kk=1:Nexp
        clearvars -except kk f Nk N mn hij concave conc2 t1 Nexp Npexp P
        seednumber = str2double([num2str(N) num2str(mn) num2str(kk+Npexp)]);
        rng(seednumber)
        %% Constraints
        umax = 2; % Maximum input magnitude
        dumax = 1;  % Maximum difference of inputs
        eps = 0.98;    % Certainty threshold
        NF = 400;   % Maximum number of measurements
        nf = 4;     % Number of fault models

        %% Fault-free model
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
                    for j2 = 1:N-1
                        H(1:j2*nu,1:j2*nu,count,j2)     = 0.25 * Gamma(1:j2*nu,1:j2*nu,count)' / Omega(1:j2*nu,1:j2*nu,count) * Gamma(1:j2*nu,1:j2*nu,count);
                        c(1:j2*nu,:,count,j2)     = 0.5 * Gamma(1:j2*nu,1:j2*nu,count)' / Omega(1:j2*nu,1:j2*nu,count)  * Gamma2(1:j2*nu,:,count);
                        h(count,j2)        = 0.25 * Gamma2(1:j2*nu,:,count)' / Omega(1:j2*nu,1:j2*nu,count) * Gamma2(1:j2*nu,:,count)...
                                          + 0.5 * log( det(0.5*Omega(1:j2*nu,1:j2*nu,count)) / ...
                                         sqrt(det(Sigmat(1:j2*nu,1:j2*nu,i))*det(Sigmat(1:j2*nu,1:j2*nu,j))) );
                    end
                    count = count+1;
                end
            end

            clear V1
            if k < 2
                V1(1:nu,1:4) = [min(umax,dumax), max(-umax,-dumax), min(umax,dumax), max(-umax,-dumax);
                                min(umax,dumax), min(umax,dumax), max(-umax,-dumax), max(-umax,-dumax)];
            else
                V1(1:nu,1:4) = [min(umax,u(1,k)+dumax), max(-umax,u(1,k)-dumax), min(umax,u(1,k)+dumax), max(-umax,u(1,k)-dumax),
                                min(umax,u(2,k)+dumax), min(umax,u(2,k)+dumax), max(-umax,u(2,k)-dumax), max(-umax,u(2,k)-dumax)];
            end
            for j = 2:N-1
                l = length(V1(1,:));
                for i = 1:l
                    V1(1:j*nu,(i-1)+1)     = [V1(1:(j-1)*nu,i); min(umax,V1((j-2)*nu+1,i)+dumax); min(umax,V1((j-2)*nu+2,i)+dumax)];
                    V1(1:j*nu,(i-1)+l+1)   = [V1(1:(j-1)*nu,i); max(-umax,V1((j-2)*nu+1,i)-dumax); min(umax,V1((j-2)*nu+2,i)+dumax)];
                    V1(1:j*nu,(i-1)+2*l+1) = [V1(1:(j-1)*nu,i); min(umax,V1((j-2)*nu+1,i)+dumax); max(-umax,V1((j-2)*nu+2,i)-dumax)];
                    V1(1:j*nu,(i-1)+3*l+1) = [V1(1:(j-1)*nu,i); max(-umax,V1((j-2)*nu+1,i)-dumax); max(-umax,V1((j-2)*nu+2,i)-dumax)];
                end
            end

            % Check whether all inputs are within concave domain
            conc = 1;
            for j2 = 1:N-1
                conc = min(conc, concave_check_poly(H(:,:,:,j2),c(:,:,:,j2),V1));
            end

            concave(kk,k) = conc;

            for i = 1:length(V1(1,:))
                count = 1;
                obj(k,i) = 0;
                for j2 = 1:nf
                    for j = j2+1:nf+1
                        for j3 = 1:N-1
                            obj(k,i) = obj(k,i) + sqrt(P(k+1,j2,kk)*P(k+1,j,kk)) * exp(-V1(:,i)' * H(:,:,count,j3) * V1(:,i)  -  c(:,:,count,j3)' * V1(:,i)  -  h(count,j3));  % Bhat. coefficient
                        end
                        count = count+1;
                    end
                end
            end

            [~,ind] = min(obj(k,:));
            u(:,k+1) = V1(1:nu,ind);    
        end
        t1(kk)=toc;

        y(:,k+1:end) = NaN;
        u(:,k+1:end) = NaN;
        P(k+2:end,:,kk) = NaN;
        
        %% Figure
%         figure;
%         subplot(3,1,1);plot(1:NF,y(:,:));xlim([1 200]);ylabel('$y_k$','Interpreter','latex')
%         subplot(3,1,2);plot(1:NF,u(:,1:end-1)');xlim([1 200]);ylim([-3 3]);ylabel('$u_k$','Interpreter','latex')
%         legend('$u_{k,1}$','$u_{k,2}$','Interpreter','latex')
%         subplot(3,1,3);plot(1:NF,P(1:end-1,:));xlim([1 200]);ylim([0 1]);ylabel('$P(M_i)$','Interpreter','latex');
%         xlabel('$k$','Interpreter','latex')
%         legend('$i=0$','$i=1$','$i=2$','$i=3$','$i=4$','Interpreter','latex');

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
        conc2(kk) = min(concave(kk,1:k-1));
    end
    acc = 1 - sum(f)/kk;
    Nkm = mean(Nk);
    save(['clafd_polytopic_SBC_' num2str(N) '_' num2str(mn)])
end

warning('on')
