%% Note: the OL solution may take several days to compute

warning('off')
clear all

N = 200;  % horizon length

%% Constraints
umax = 2; % Maximum input magnitude
dumax = 1;  % Maximum difference of inputs
eps = 0.98;    % Certainty threshold
NF = 400;   % Maximum number of measurements
nf = 4;     % Number of fault models
nc = 4;     % Number of constraints
ccpN = 20;  % Number of initializations

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
P  = zeros(NF+1,nf+1);
Sigma = zeros(nx,nx,NF,nf+1);
Sigmat = zeros(N*ny,N*ny,nf+1);

P(1,1) = 0.2;
P(1,2:nf+1) = (1 - P(1,1)) / nf;
u(:,1) = 0;

for i=1:nf+1
    xh(:,1,i) = [0;1];
    Sigma(:,:,1,i) = 0.5*eye(nx);
end

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

%% Offline computation of OL solution
for i = 1:nf+1
    Sigmat(:,:,i) = Ct(:,:,i) * At(:,:,i) * Sigma(:,:,1,i) * At(:,:,i)' * Ct(:,:,i)'...
        +  Ct(:,:,i) * Et(:,:,i) * Qt * Et(:,:,i)' * Ct(:,:,i)'...
        +  Ft(:,:,i) * Rt * Ft(:,:,i)';
end

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
        Gamma2(:,:,count)= Ct(:,:,i) * At(:,:,i) * xh(:,1,i) - Ct(:,:,j) * At(:,:,j) * xh(:,1,j);
        Omega(:,:,count) = Sigmat(:,:,i) + Sigmat(:,:,j);
        H(:,:,count)     = 0.25 * Gamma(:,:,count)' / Omega(:,:,count) * Gamma(:,:,count);
        c(:,:,count)     = 0.5 * Gamma(:,:,count)' / Omega(:,:,count)  * Gamma2(:,:,count);
        h(count)        = 0.25 * Gamma2(:,:,count)' / Omega(:,:,count) * Gamma2(:,:,count)...
                                + 0.5 * sum(log(diag(0.5*Omega(:,:,count)))) -...
                                0.5 * (sum(log((sqrt(diag(Sigmat(:,:,i)))))) + sum(log((sqrt(diag(Sigmat(:,:,j)))))));  % the remaining terms are omitted for preventing computational overload
        count = count+1;
    end
end

fun1 = @(l)sumwbhat(P(1,:),H,c,h,l);
opts = optimoptions(@fmincon,'Algorithm','sqp','MaxIterations',16000,'MaxFunctionEvaluations',4000*N*nu,'Display','off');
problem = createOptimProblem('fmincon','objective',...
fun1,'x0',zeros(N*nu,1),'lb',-umax*ones(N*nu,1),'ub',umax*ones(N*nu,1),'Aineq',...
[[eye(N*nu) - [zeros(nu,(N-1)*nu), zeros(nu); eye((N-1)*nu),zeros(((N-1)*nu),nu)]];...
[[zeros(nu,(N-1)*nu), zeros(nu); eye((N-1)*nu),zeros(((N-1)*nu),nu)] - eye(N*nu)]],...
'bineq',dumax*ones(2*N*nu,1),'options',opts);
ms = MultiStart('UseParallel',1);
u_opt = run(ms,problem,ccpN);    
for i = 1:floor(NF/N)
    u(:,(i-1)*N+1:i*N) = reshape(u_opt,nu,N);
end

save('OL_solution.mat','u')