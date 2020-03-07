function DEMO_MeanFieldsModules
% This demo performs the simulations that accompany the paper 'Modules or
% Mean-Fields' (Parr, Sajid, and Friston). First, we simulate a system of
% particles by numerically integrating stochastic differential equations.
% These equations are based upon a flow that achieves steady-state
% consistent with a specified Hamiltonian.
%
% We then numerically integrate the dynamics of their sufficient statistics
% using a Fokker-Planck Equation (under mean-field and Laplace
% approximations). Our aim is to show that the mean-field partition
% reproduces features of modular perspectives on brain function.

% Initialisation
%--------------------------------------------------------------------------
rng default
close all
G  = 2;            % Half the amplitude of random fluctuations [2]
dt = 0.01;         % Integration step
T  = 16;           % Final time
y  = 8*[1 1 -1 -1 1 -1 -1 1]' + 2*randn(8,1); % Data
x  = zeros(34,1);
td = 25;           % Timesteps between animation frames

if size(y,2) == 1
    y = repmat(y,1,round(T/dt)+1);
end

ANIMATE = 0;

% Colour-scheme for figures
s   = (0:0.05:1).^0.5;
map = [0.6*s;0.6*s;s]';

% Define Hamiltonian and find expressions for gradient and Hessian (under
% quadratic assumptions
%--------------------------------------------------------------------------
Hxy = @(xy) mfm_H(xy(1:size(x,1),1),xy(size(x,1)+1:end,1));
[dHdx0, dHdxx0] = mfm_dfdx(Hxy,zeros(size(x,1)+size(y,1),1));
dHdx = @(x,y) dHdx0(1:size(x,1),1) + dHdxx0(1:size(x,1),:)*[x;y];

f1 = figure(1);clf
f1.Name = 'Quadratic expansion';
f1.Color = 'w';
f1.Position = [500 50 500 550];
imagesc(-dHdxx0), axis square, axis tight
colormap gray
title('Precision (curvature of Hamiltonian)')

% Simulate SDE
%--------------------------------------------------------------------------
% Single instance
x0 = 2*randn(34,1);
f  = @(x,y) mfm_f_ness(G,dHdx,x,y);
[t, X] = mfm_sde(f,G,x0,T,dt,y);

% Simulate many instances
X2 = X;
mX = X/16;
for i = 1:16
    x0 = 2*randn(34,1);
    [~, C] = mfm_sde(f,G,x0,T,dt,y);
    X2 = [X2,C];
    mX = mX + C/16;
end

% Simulate density dynamics
%--------------------------------------------------------------------------
m0 = zeros(size(x0));
V0 = 8*eye(size(x0,1),size(x0,1));

[m,V] = mfm_ode_fpe(m0,V0,dHdx0,dHdxx0,G,y,T,dt);

if ANIMATE
    for i = 1:td:length(t)
        [B,D,r] = mfm_plot_dd(m(:,i),V(:,:,i),15);
        I{i} = B;
        J{i} = D;
    end
else
    [B,D,r] = mfm_plot_dd(m(:,end),V(:,:,end),15);
    I{1} = B;
    J{1} = D;
end

% Animate dynamics
%--------------------------------------------------------------------------

f2 = figure(2); clf
f2.Name = 'Dynamics';
f2.Color = 'w';
f2.Position = [500 50 500 550];

subplot(3,3,1:2)
plot(t,X)
xlim([0 T])
title('Dynamics (single realisation)')
subplot(3,3,4:5)
plot(t,mX)
xlim([0 T])
title('Average dynamics (many realisations)')
subplot(3,3,7:8)
plot(t,m)
xlim([0 T])
title('Density dynamics (mode)')

if ANIMATE
    for i = 1:td:length(t)
        subplot(3,3,3)
        plot(X(i,1:2:end-1),X(i,2:2:end),'.','MarkerSize',20,'Color',[0.6 0.6 1])
        axis equal, axis([-15 15 -15 15]), axis square, set(gca,'Color','k')
        hold on
        plot(y(1:2:end-1,i),y(2:2:end,i),'.','MarkerSize',20,'Color',[1 0.6 0.6]), hold off
        subplot(3,3,6)
        plot(X2(i,1:2:end-1),X2(i,2:2:end),'.','MarkerSize',5,'Color',[0.6 0.6 1])
        axis equal, axis([-15 15 -15 15]), axis square, set(gca,'Color','k')
        hold on
        plot(y(1:2:end-1,i),y(2:2:end,i),'.','MarkerSize',20,'Color',[1 0.6 0.6]), hold off
        subplot(3,3,9)
        axis equal, axis([-15 15 -15 15]), axis square
        imagesc(r,r,I{i}), axis xy, colormap(map)
        hold on
        plot(y(1:2:end-1,i),y(2:2:end,i),'.','MarkerSize',20,'Color',[1 0.6 0.6]), hold off
        drawnow
    end
    
    f3 = figure(3); clf
    f3.Name = 'Modules';
    f3.Color = 'w';
    f3.Position = [500 50 500 550];
    for i = 1:td:length(t)
        for j = 1:17
            if j == 1
                subplot(5,4,1)
            else
                subplot(5,4,j+3)
            end
            imagesc(r,r,J{i}{j}), axis xy, colormap(map)
            axis equal, axis([-15 15 -15 15]), axis square
        end
        drawnow
    end
    
else
    subplot(3,3,3)
    plot(X(end,1:2:end-1),X(end,2:2:end),'.','MarkerSize',20,'Color',[0.6 0.6 1])
    axis equal, axis([-15 15 -15 15]), axis square, set(gca,'Color','k')
    hold on
    plot(y(1:2:end-1,end),y(2:2:end,end),'.','MarkerSize',20,'Color',[1 0.6 0.6]), hold off
    subplot(3,3,6)
    plot(X2(end,1:2:end-1),X2(end,2:2:end),'.','MarkerSize',5,'Color',[0.6 0.6 1])
    axis equal, axis([-15 15 -15 15]), axis square, set(gca,'Color','k')
    hold on
    plot(y(1:2:end-1,end),y(2:2:end,end),'.','MarkerSize',20,'Color',[1 0.6 0.6]), hold off
    subplot(3,3,9)
    axis equal, axis([-15 15 -15 15]), axis square
    imagesc(r,r,I{1}), axis xy, colormap(map)
    hold on
    plot(y(1:2:end-1,end),y(2:2:end,end),'.','MarkerSize',20,'Color',[1 0.6 0.6]), hold off
    
    f3 = figure(3); clf
    f3.Name = 'Modules';
    f3.Color = 'w';
    f3.Position = [500 50 500 550];
    for j = 1:17
        if j == 1
            subplot(5,4,1)
        else
            subplot(5,4,j+3)
        end
        imagesc(r,r,J{1}{j}), axis xy, colormap(map)
        axis equal, axis([-15 15 -15 15]), axis square, hold on
        plot(m(2*(j-1)+1,:),m(2*(j-1)+2,:),'w')
    end
end
    


% Perturb dynamics (via y)
%--------------------------------------------------------------------------
y(:,4/dt+2:end) = []; % Truncate

y1 = y;
y1(1,40:100) = y(1,40:100) + 32*sin((1:61)*pi/61);
y1(2,40:100) = y(2,40:100) + 32*sin((1:61)*pi/61);

y2 = y;
y2(3,40:100) = y(3,40:100) + 32*sin((1:61)*pi/61);
y2(4,40:100) = y(4,40:100) + 32*sin((1:61)*pi/61);

y3 = y;
y3(1,40:100) = y(1,40:100) + 32*sin((1:61)*pi/61);
y3(2,40:100) = y(2,40:100) + 32*sin((1:61)*pi/61);
y3(3,40:100) = y(3,40:100) + 32*sin((1:61)*pi/61);
y3(4,40:100) = y(4,40:100) + 32*sin((1:61)*pi/61);

t = 0:dt:4;

m0 = m(:,end);
V0 = V(:,:,end);

[m1,V1] = mfm_ode_fpe(m0,V0,dHdx0,dHdxx0,G,y1,4,dt);
[m2,V2] = mfm_ode_fpe(m0,V0,dHdx0,dHdxx0,G,y2,4,dt);
[m3,V3] = mfm_ode_fpe(m0,V0,dHdx0,dHdxx0,G,y3,4,dt);

f4 = figure(4); clf
f4.Name = 'Modules upper right';
f4.Color = 'w';
f4.Position = [500 50 500 550];

subplot(6,3,16)
plot(t,y1)
subplot(6,3,17)
plot(t,y2)
subplot(6,3,18)
plot(t,y3)

for i = 1:5
    subplot(6,3,(i-1)*3 + 1)
    mfm_intervals(t,V1((1:2) + (i-1)*2,(1:2) + (i-1)*2,:),m1((1:2) + (i-1)*2,:))
end
for i = 1:5
    subplot(6,3,(i-1)*3 + 2)
    mfm_intervals(t,V2((1:2) + (i-1)*2,(1:2) + (i-1)*2,:),m2((1:2) + (i-1)*2,:))
end
for i = 1:5
    subplot(6,3,(i-1)*3 + 3)
    mfm_intervals(t,V3((1:2) + (i-1)*2,(1:2) + (i-1)*2,:),m3((1:2) + (i-1)*2,:))
end

f5 = figure(5); clf
f5.Name = 'Modules lower left';
f5.Color = 'w';
f5.Position = [500 50 500 550];

subplot(6,3,16)
plot(t,y1)
subplot(6,3,17)
plot(t,y2)
subplot(6,3,18)
plot(t,y3)

subplot(6,3,1)
mfm_intervals(t,V1(1:2,1:2,:),m1(1:2,:))
for i = 2:5
    subplot(6,3,(i-1)*3 + 1)
    mfm_intervals(t,V1((1:2) + (i-1)*2 + 8,(1:2) + (i-1)*2 + 8,:),m1((1:2) + (i-1)*2 + 8,:))
end
subplot(6,3,2)
mfm_intervals(t,V2(1:2,1:2,:),m2(1:2,:))
for i = 2:5
    subplot(6,3,(i-1)*3 + 2)
    mfm_intervals(t,V2((1:2) + (i-1)*2 + 8,(1:2) + (i-1)*2 + 8,:),m2((1:2) + (i-1)*2 + 8,:))
end
subplot(6,3,3)
mfm_intervals(t,V3(1:2,1:2,:),m3(1:2,:))
for i = 2:5
    subplot(6,3,(i-1)*3 + 3)
    mfm_intervals(t,V3((1:2) + (i-1)*2 + 8,(1:2) + (i-1)*2 + 8,:),m3((1:2) + (i-1)*2 + 8,:))
end

%==========================================================================

% Auxiliary functions
%--------------------------------------------------------------------------
function H = mfm_H(x,y)
% Hamiltonian (i.e., generative model)

R = [1 0.2; -0.2 1];

C = 4; % Variance parameter

H = x(1:2,1)'*x(1:2,1)/32;

for n = 2:4
    H = H + (x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[1;1]))'*(x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[1;1]))/((n - 1)*C);
end
n = 5;
H = H + (x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[1;1]))'*(x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[1;1]))/((n - 1)*C);
H = H + (y(1:2,1) - R*(x(2*(n-1)+(1:2),1)+[1;1]))'*(y(1:2,1) - R*(x(2*(n-1)+(1:2),1)+[1;1]))/(n*C);
n = 6;
H = H + (x(2*(n-1)+(1:2),1) - R*(x(1:2,1)+[-1;-1]))'*(x(2*(n-1)+(1:2),1) - R*(x(1:2,1)+[-1;-1]))/((n - 5)*C);
for n = 7:8
    H = H + (x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[-1;-1]))'*(x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[-1;-1]))/((n - 5)*C);
end
n = 9;
H = H + (x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[-1;-1]))'*(x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[-1;-1]))/((n - 5)*C);
H = H + (y(3:4,1) - R*(x(2*(n-1)+(1:2),1)+[-1;-1]))'*(y(3:4,1) - R*(x(2*(n-1)+(1:2),1)+[-1;-1]))/((n-4)*C);

R = R';

n = 10;
H = H + (x(2*(n-1)+(1:2),1) - R*(x(1:2,1)+[1;-1]))'*(x(2*(n-1)+(1:2),1) - R*(x(1:2,1)+[1;-1]))/((n - 9)*C);
for n = 11:12
    H = H + (x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[1;-1]))'*(x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[1;-1]))/((n - 9)*C);
end
n = 13;
H = H + (x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[1;-1]))'*(x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[1;-1]))/((n - 9)*C);
H = H + (y(5:6,1) - R*(x(2*(n-1)+(1:2),1)+[1;-1]))'*(y(5:6,1) - R*(x(2*(n-1)+(1:2),1)+[1;-1]))/((n - 8)*C);
n = 14;
H = H + (x(2*(n-1)+(1:2),1) - R*(x(1:2,1)+[-1;1]))'*(x(2*(n-1)+(1:2),1) - R*(x(1:2,1)+[-1;1]))/((n - 13)*C);
for n = 15:16
    H = H + (x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[-1;1]))'*(x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[-1;1]))/((n - 13)*C);
end
n = 17;
H = H + (x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[-1;1]))'*(x(2*(n-1)+(1:2),1) - R*(x(2*(n-1)+(-1:0),1)+[-1;1]))/((n - 13)*C);
H = H + (y(7:8,1) - R*(x(2*(n-1)+(1:2),1)+[-1;1]))'*(y(7:8,1) - R*(x(2*(n-1)+(1:2),1)+[-1;1]))/((n - 12)*C);
H = H*2;

function [t, x] = mfm_sde(f,G,x0,T,dt,y)
% This numerically integrates a stochastic differential equation with
% average rate of change given by f, and random fluctuations with amplitude
% 2*G.

if length(G) == 1
    G = G*eye(length(x0));
end

% Basic stochastic Runge-Kutter scheme
% (https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_(SDE))
x = x0;
for i = dt:dt:T
    dW = (mvnrnd(zeros(size(x,1),1),2*G*dt))';
    x(:,end+1) = x(:,end) + dt*f(x(:,end),y(:,round(i/dt))) + dW;
end
t = 0:dt:T;
x = x';

function f = mfm_f_ness(G,dHdx,x,y)
% This function provides the average rate of change of change of a system
% at non-equilibrium steady state.
G = G*eye(size(x,1));
Q = zeros(size(x,1),size(x,1));
for i = 1:2:size(x,1)-1
    Q(i+(0:1),i+(0:1)) = 2*[0 1; -1 0];
end

f = - (G + Q)*dHdx(x,y);

function [dmdt,dVdt] = mfm_f_fpe(m,V,dHdx,dHdxx,G,y)
% This function returns the rate of change of the mean and variance based
% upon Laplace and mean-field approximations. 

Q = zeros(size(m,1),size(m,1));
for i = 1:2:size(m,1)-1
    Q(i+(0:1),i+(0:1)) = 2*[0 1; -1 0];
end

dmdt = -(G*eye(size(m,1)) + Q)*(dHdx(1:size(m,1),1) + dHdxx(1:size(m,1),:)*[m;y]);
Q = 2*[0 1; -1 0];
for i = 1:size(m,1)/2
    dVdt(2*(i-1) + (1:2),2*(i-1) + (1:2))...
        = 2*G*eye(2)...
        - 2*G*dHdxx(2*(i-1) + (1:2),2*(i-1) + (1:2))*V(2*(i-1) + (1:2),2*(i-1) + (1:2))...
        - Q*dHdxx(2*(i-1) + (1:2),2*(i-1) + (1:2))*V(2*(i-1) + (1:2),2*(i-1) + (1:2))...
        - dHdxx(2*(i-1) + (1:2),2*(i-1) + (1:2))*V(2*(i-1) + (1:2),2*(i-1) + (1:2))*Q';
end

function [m,V] = mfm_ode_fpe(m0,V0,dHdx,dHdxx,G,y,T,dt)
% Use Runge-Kutter scheme to integrate Fokker-Planck Equation
f = @(m,V,y) mfm_f_fpe(m,V,dHdx,dHdxx,G,y);
m = m0;
V = V0;
for i = dt:dt:T
    [k1, j1] = f(m(:,end),V(:,:,end),y(:,round(i/dt)));
    k1 = k1*dt;
    j1 = j1*dt;
    [k2, j2] = f(m(:,end)+k1/2,V(:,:,end)+j1/2,y(:,round(i/dt)));
    k2 = k2*dt;
    j2 = j2*dt;
    [k3, j3] = f(m(:,end)+k2/2,V(:,:,end)+j2/2,y(:,round(i/dt)));
    k3 = k3*dt;
    j3 = j3*dt;
    [k4, j4] = f(m(:,end)+k3,V(:,:,end)+j3,y(:,round(i/dt)));
    k4 = k4*dt;
    j4 = j4*dt;
    m(:,end+1) = m(:,end) + (k1 + k4 + 2*(k2 + k3))/6;
    V(:,:,end+1) = V(:,:,end) + (j1 + j4 + 2*(j2 + j3))/6;
end

function [dfdx, dfdxx] = mfm_dfdx(f,x)
% Numerical derivatives
d     = exp(-4);
dfdx  = zeros(size(x));
dfdxx = zeros(length(x),length(x));
for i = 1:length(x)
    dx = zeros(length(x),1);
    dx(i) = d;
    dfdx(i) = 0.5*(f(x + dx) - f(x - dx))/d;
    for j = 1:length(x)
        dy = zeros(length(x),1);
        dy(j) = d;
        dfdxx(i,j) = 0.25*((f(x + dx + dy) - f(x - dx + dy))/d - (f(x + dx - dy) - f(x - dx - dy))/d)/d;
    end
end

function [C,D,r] = mfm_plot_dd(m,V,N)
% Returns frames of density dynamics for plotting. C includes all factors,
% while D gives individual factors.
r = -N:0.5:N;
[X,Y] = meshgrid(r,r);
C = zeros(size(X));
for n = 1:2:size(m,1)-1
   g = @(x,y) exp(- 0.5*(m(n:n+1) - [x;y])'*inv(V(n:n+1,n:n+1))*(m(n:n+1) - [x;y]))/(2*pi*sqrt(det(V(n:n+1,n:n+1))));
   B = arrayfun(g,X,Y);
   D{(n+1)/2} = B;
   B = B/max(max(B));
   C = arrayfun(@max,C,B);
end

function mfm_intervals(t,V,m)
C1 = [m(1,:)' - 1.64*squeeze(V(1,1,:)).^2,...
      2*1.64*squeeze(V(1,1,:)).^2];
a1 = area(t,C1,'FaceColor',[0.93,0.61,0.32]); hold on
C2 = [m(2,:)' - 1.64*squeeze(V(2,2,:)).^2,...
     2*1.64*squeeze(V(2,2,:)).^2];
a2 = area(t,C2,'FaceColor',[0.32,0.32,0.93]);
a1(1).FaceAlpha = 0;
a2(1).FaceAlpha = 0;
a1(1).EdgeAlpha = 0;
a2(1).EdgeAlpha = 0;
a1(1).ShowBaseLine = 'off';
a2(1).ShowBaseLine = 'off';
a1(2).FaceAlpha = 0.1;
a2(2).FaceAlpha = 0.1;
a1(2).EdgeAlpha = 0;
a2(2).EdgeAlpha = 0;
a1(2).ShowBaseLine = 'off';
a2(2).ShowBaseLine = 'off';

plot(t,m(1,:),'Color',[0.93,0.61,0.32])
plot(t,m(2,:),'Color',[0.32,0.32,0.93])
