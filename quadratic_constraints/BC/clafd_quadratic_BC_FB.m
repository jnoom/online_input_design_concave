% This script generates results for the closed-loop approach "Bhattacharyya
% Coefficient" (BC) in a quadratic constraint set. It requires the Python 
% script "dccp_BC.py" in the Current Folder. 
% It produces a figure demonstrating the methodology on a
% feedback-controlled system.

%% Run only once in a MATLAB session (change to the right path)
% pyenv('Version',"path_to_environment\python.exe")

clear all

%% Simulation variables
NF = 550;   % Maximum number of measurements
NF1 = 150;  % Settling without CLAFD
nf = 4;     % Number of fault models
mn1 = 0;    % Model for control design
mn = 3;     % True model number
kk = 1;     % Experiment number

beta = 0.1;         % Damping of nominal model
omega = pi/2;       % Resonance frequency of nominal model
R = 0.01*eye(2);    % Measurement noise > 0
Q = 0.0001*eye(2);  % Process noise >= 0
S = zeros(2);       % Correlation

%% Algorithm variables
uE = 2.5*10^-3;  % Maximum input energy
N = 5;           % Horizon length
ccpN = 1;        % Number of initial points of DCCP per time step

r  = [zeros(2,20), kron(ones(1,NF-20+N),[3;5])];  % Reference signal

%% Random generator seed
seednumber = str2double([num2str(N) num2str(mn) num2str(kk)]);
rng(seednumber)

%% Fault-free model
sys = c2d(ss(tf(1,[1 2*beta*omega omega^2])),1);

A = repmat(sys.A,1,1,nf+1) + [2.0 0; 0 0];
B = repmat([sys.B [0.5;0]],1,1,nf+1);
C = repmat([sys.C;[0.1 0.5]],1,1,nf+1);

A(:,:,2) = A(:,:,1) + [0.01 0; 0 0];
A(:,:,3) = A(:,:,1) + [0.02 0; 0 0];
A(:,:,4) = A(:,:,1) + [0.03 0; 0 0];
A(:,:,5) = A(:,:,1) + [0.04 0; 0 0];

nx1 = length(A(:,1,1));
ny = length(C(:,1,1));
nu = length(B(1,:,1));
nv = ny;
nw1 = nx1;

%% Equal DC gain
for i = 1:nf+1
    B(:,2,i) = B(:,2,i) / (C(1,:,i)/(eye(2) - A(:,:,i))*B(:,2,i));
end

%% Kalman filter
Am = A(:,:,mn1+1); Bm = B(:,:,mn1+1); Cm = C(:,:,mn1+1);
[Pm,Km,L] = idare(Am',Cm',Q,R,S',eye(nx1));

%% Pole placement
Fm = place(Am,Bm,[0.94,0.95]);

%% Closed-loop dynamics
for i=1:nf+1
    Acl(:,:,i) = [A(:,:,i), -B(:,:,i)*Fm; 
        Km*C(:,:,i), Am-Bm*Fm-Km*Cm];
    Bcl(:,:,i) = [B(:,:,i); Bm];
    Ccl(:,:,i) = [C(:,:,i), zeros(ny,nx1)];
end

Ff = inv(Ccl(:,:,mn1+1)/(eye(2*nx1)-Acl(:,:,mn1+1)) * Bcl(:,:,mn1+1));  % Feedforward gain (inverse DC gain)

for i=1:nf+1
    BclF(:,:,i) = Bcl(:,:,i)*Ff;
end

Ecl = [eye(nx1), zeros(nx1,ny);
    zeros(ny,nx1), Km];

Phi = Ecl*[Q, S'; S, R]*Ecl';  % CL process noise 
Psi = Ecl*[S;R];               % CL noise correlation

for i =1:nf+1
    eigs(:,i) = abs(eig(Acl(:,:,i)));
end
max(eigs,[],'all')

%% Convert dimensions to CL system
nx = length(Acl(1,:,1));
nw = length(Q(1,:)) + length(R(1,:));

%% N-step expanded models for CL system
At = zeros((N)*nx,nx,nf+1);
Bt = zeros((N)*nx,(N-1)*nu,nf+1);
Ct = zeros((N)*ny,(N)*nx,nf+1);
TA = zeros((N)*nx,(N)*nw,nf+1);
for k = 1:nf+1
    for i = 1:N
        At((i-1)*nx+1:i*nx,1:nx,k) = Acl(:,:,k)^(i-1);
    end
    for i = 1:N-1
        for j = 1:i
            Bt((i)*nx+1:(i+1)*nx,(j-1)*nu+1:j*nu,k) = Acl(:,:,k)^(i-j) * BclF(:,:,k);
            TA((i)*nx+1:(i+1)*nx,(j-1)*nw+1:j*nw,k) = Acl(:,:,k)^(i-j);
        end
    end
    Ct(:,:,k) = kron(eye(N),Ccl(:,:,k));
end

Qt = kron(eye(N),Phi);
Rt = kron(eye(N),R);
St = [zeros(nw,(N-1)*nv), zeros(nw,nv)
    kron(eye(N-1),Psi), zeros((N-1)*nw,nv)];

%% Settling to reference
x3  = zeros(nx1,NF);
xh3 = zeros(nx1,NF,nf+1);
y3  = zeros(ny,NF);
yh3 = zeros(ny,NF,nf+1);
u3  = zeros(nu,NF+1);

u3(:,1) = 0; %[0;0];%V0(:,2);

for i=1:nf+1
    xh3(:,1,i) = [0;1];
    Sigma(:,:,1,i) = 0.5*eye(nx1);
end
x3(:,1) = xh3(:,1,1) + [sqrt(Sigma(1,1,1,1))*randn(1,1); sqrt(Sigma(2,2,1,1))*randn(1,1)];

for k=1:NF1
    u3(:,k) = -Fm*xh3(:,k,mn1+1) + Ff*r(:,k);
    v3(:,k) = R^0.5*randn(nv,1);
    if k>1
        w3(:,k-1) = Q^0.5*randn(nw1,1);
        x3(:,k) = A(:,:,mn+1)*x3(:,k-1) + B(:,:,mn+1)*u3(:,k-1) + w3(:,k-1);
    end
    y3(:,k) = C(:,:,mn+1)*x3(:,k) + v3(:,k); 
    for i = mn1+1
        xh3(:,k+1,i) = A(:,:,i) * xh3(:,k,i)  +  B(:,:,i) * u3(:,k)  +  Km * (y3(:,k) - C(:,:,i)*xh3(:,k,i));                        % Step 16: Kalman predictor
    end
end

%% Initial condition
x  = zeros(nx,NF);
xhp = zeros(nx,NF+1,nf+1);
y  = zeros(ny,NF);
yh = zeros(ny,NF,nf+1);
u  = zeros(nu,NF+1);
ut = zeros(nu,NF+1);
Sm  = zeros(ny,ny,NF,nf+1);
Kp  = zeros(nx,ny,NF,nf+1);
P(:,:,kk)  = zeros(NF+1,nf+1);
Sigmap = zeros(nx,nx,NF,nf+1);
Sigmat = zeros((N)*ny,(N)*ny,nf+1);

P(1,1,kk) = 0.2;
P(1,2:nf+1,kk) = (1 - P(1,1,kk)) / nf;

u(:,1) = r(:,k);
x(:,1) = [x3(:,k);xh3(:,k)];

for i=1:nf+1
    xhp(:,1,i) = [xh3(:,k);xh3(:,k)];
    Sigmap(:,:,1,i) = blkdiag(Pm,0.05*eye(2));
end


%% CLAFD
tic
for k = 1:NF-NF1
    wv(:,k) = blkdiag(Q,R)^0.5*randn(nw,1);
    if k>1
        x(:,k) = Acl(:,:,mn+1)*x(:,k-1) + BclF(:,:,mn+1)*u(:,k-1) + Ecl*wv(:,k-1);  % Update system state
    end
    y(:,k)   = Ccl(:,:,mn+1)*x(:,k) + wv(nw-nv+1:nw,k);  % Obtain measurement
    psum = 0;
    for i = 1:nf+1
        yh(:,k,i) = Ccl(:,:,i) * xhp(:,k,i);  % Predicted output
        Sm(:,:,k,i) = Ccl(:,:,i) * Sigmap(:,:,k,i) * Ccl(:,:,i)'  +  R;  % Covariance corresponding to predicted ouput
        psum = psum + mvnpdf(y(:,k), Ccl(:,:,i)*xhp(:,k,i), real(Sm(:,:,k,i))) * P(k,i,kk);  % Denominator of Bayesian update rule
    end
    for i = 1:nf+1
        P(k+1,i,kk) = mvnpdf(y(:,k), Ccl(:,:,i)*xhp(:,k,i), real(Sm(:,:,k,i)))  *  P(k,i,kk)  /  psum;  % Bayes update rule

        Kp(:,:,k,i) = (Psi + Acl(:,:,i)* Sigmap(:,:,k,i) * Ccl(:,:,i)') / Sm(:,:,k,i);  % Kalman gain
        xhp(:,k+1,i) = Acl(:,:,i) * xhp(:,k,i)  +  BclF(:,:,i) * u(:,k)  +  Kp(:,:,k,i) * (y(:,k) - Ccl(:,:,i)*xhp(:,k,i));  % Predicted state
        Sigmap(:,:,k+1,i) = Acl(:,:,i) * Sigmap(:,:,k,i) * Acl(:,:,i)'  +  Phi  -  Kp(:,:,k,i) * (Psi' + Ccl(:,:,i) * Sigmap(:,:,k,i) * Acl(:,:,i)');% Covariance corresponding to predicted state 

        Sigmat(:,:,i) = Ct(:,:,i) * At(:,:,i) * Sigmap(:,:,k+1,i) * At(:,:,i)' * Ct(:,:,i)'...
            +  Ct(:,:,i) * TA(:,:,i) * Qt * TA(:,:,i)' * Ct(:,:,i)'...
            +  Ct(:,:,i) * TA(:,:,i) * St  +  St' * TA(:,:,i)' * Ct(:,:,i)'  +  Rt;  % Covariance corresponding to predicted output sequence
    end
    
    % Calculate Bhattacharyya coefficients
    Gamma = zeros((N)*ny,(N-1)*nu);
    Gamma2= zeros(size(Ct(:,:,1)*At(:,:,1)*xhp(:,1,1)));
    Omega = zeros(size(Sigmat));
    H     = zeros((N-1)*nu,(N-1)*nu);
    c     = zeros((N-1)*nu,1);
    h     = 0;
    count = 1;
    rk = vec(r(:,k+NF1+2:k+NF1+N));
    for i = 1:nf
        for j = i+1:nf+1
            Gamma(:,:,count) = Ct(:,:,i) * Bt(:,:,i) - Ct(:,:,j) * Bt(:,:,j);
            Gamma2(:,:,count)= Ct(:,:,i) * At(:,:,i) * xhp(:,k+1,i) - Ct(:,:,j) * At(:,:,j) * xhp(:,k+1,j);
            Omega(:,:,count) = Sigmat(:,:,i) + Sigmat(:,:,j);
            for j2 = 2:N
                j3 = j2-1;
                H(:,:,count,j3)     = 0.25 * Gamma(1:j2*ny,:,count)' / Omega(1:j2*ny,1:j2*ny,count) * Gamma(1:j2*ny,:,count);
                c2(:,:,count,j3)    = 0.5 * Gamma(1:j2*ny,:,count)' / Omega(1:j2*ny,1:j2*ny,count)  * Gamma2(1:j2*ny,:,count);
                h2(count,j3)        = 0.25 * Gamma2(1:j2*ny,:,count)' / Omega(1:j2*ny,1:j2*ny,count) * Gamma2(1:j2*ny,:,count)...
                                  + 0.5 * log( det(0.5*Omega(1:j2*nu,1:j2*nu,count)) / ...
                                 sqrt(det(Sigmat(1:j2*ny,1:j2*ny,i))*det(Sigmat(1:j2*ny,1:j2*ny,j))) );
                             
                c(:,:,count,j3)     = c2(:,:,count,j3) + 2*H(:,:,count,j3)*rk;      % Account for reference
                h(count,j3)         = h2(count,j3) + c2(:,:,count,j3)'*rk + rk'*H(:,:,count,j3)*rk;     % Account for reference
            end
            count = count+1;
        end
    end
    for j2=1:N-1
        concave(kk,k,j2) = concave_check_quad(H(:,:,:,j2),c(:,:,:,j2),uE,j2+1,nf);
    end
        
    u_opt = double(py.dccp_BC.minsumbhat(py.numpy.array(permute(H(:,:,:,end),[3,2,1])),...
        py.numpy.array(reshape(permute(c(:,:,:,end),[3,2,1]),[count-1,(N-1)*nu])),...
        py.numpy.array(h(:,end)),py.numpy.array(P(k+1,:,kk)),uE,nu,ccpN,...
        str2double([num2str(seednumber) num2str(k)])));  % BC

%     u_opt = double(py.dccp_SBC.minsumbhat(py.numpy.array(permute(H,[4,3,2,1])),...
%         py.numpy.array(reshape(permute(c,[4,3,2,1]),[(N-1),count-1,(N-1)*nu])),...
%         py.numpy.array(permute(h,[2,1])),py.numpy.array(P(k+1,:,kk)),uE,...
%         nu,ccpN,str2double([num2str(seednumber) num2str(k)])));  % SBC
    
    u(:,k+1) = u_opt(1:nu)' + r(:,k+NF1+1); 
%     u(:,k+1) = r(:,k+NF1+1);  % Nominal input without CLAFD
end
t1(kk)=toc;

%% Figure
y(:,k+1:end) = NaN;
u(:,k+1:end) = NaN;
P(k+2:end,:,kk) = NaN;

figure;
subplot(3,1,1);plot(1:NF,y(:,:));hold on;plot(1:NF-NF1,r(:,NF1+1:NF)');xlim([1 k]);ylabel('$y_k$','Interpreter','latex')
subplot(3,1,2);plot(1:NF-NF1,u(:,1:NF-NF1)-r(:,NF1+1:NF));xlim([1 k]);ylabel('$u_k - r_k$','Interpreter','latex')
subplot(3,1,3);plot(1:NF,P(1:end-1,:));xlim([1 k]);ylim([0 1]);ylabel('$P_k(M^{[i]})$','Interpreter','latex');
xlabel('$k$','Interpreter','latex')

MSE = mean((y(:,1:NF-NF1)-r(:,NF1:NF-1)).^2,2)
mean(min(concave,[],3))



