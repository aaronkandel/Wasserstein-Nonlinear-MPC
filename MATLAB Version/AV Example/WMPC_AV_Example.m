%% Wasserstein Distributionally Robust MPC
% Aaron Kandel
% 08/11/2023

%{
This code solves the constrained obstacle avoidance problem for an
autonomous vehicle with a simple vision system.

The method used is Wasserstein DRO-MPC.
%}


clear
close all
clc

%% Generate Training Data:

% Define Params:
episodes = 1; % Number of training trajectories
tmax = 500; % [s] max time
dt = 0.2;  % [s] timestep
tint = 0:dt:tmax; % [s] time vector
umin = [-0.75;-0.65]; 
umax = -umin;
maxV = 10;  % [m/s] maximum velocity


p.dt = dt;
p.L = 0.25; 
hor = 8;

 %% Generate Terrain (Optional):
%  close all
% M = zeros(1000,1000); % initial matrix
% N = 50; % number of bumps
% sigma = 100;% std (width) of Gauss 
% maxAmplitude = 0.075; % maximum height
% [x,y] = meshgrid(1:size(M,1),1:size(M,2));
% for k=1:N
%     % random location of bumps
%     xc = randi(size(M,1));
%     yc = randi(size(M,2));
%     % Gauss function
%     exponent = ((x-xc).^2 + (y-yc).^2)./(2*sigma^2);
%     amplitude = rand()*maxAmplitude;  
%     % add Gauss to the matrix M
%     M = M + amplitude*exp(-exponent);
% end
% contour(M)
% shading interp
% save('obstacles6.mat','M')

% %%
% safeT = [];
% 
%% Function defining Obstacles:
close all
load obstacles
[X,Y] = meshgrid(1:1000,1:1000);

% Basic vision system function:
Zfun = @(x,y) interp2(X,Y,M,x*10,y*10);
% Can be thought of like taking the min value from a set of LIDAR bins.

OsimLim = 0.1;  % Maximum allowed value of vision function (interface between obstacle and safety)
offsett = OsimLim;
obstacles = find(M < offsett);
safes = find(M>=offsett);
ob2 = find(X<00);

Z2 = M;
Z2(obstacles) = 0;
Z2(safes) = 1;

%%
figure
contour(X,Y,M)
grid on
colorbar('southoutside')
%% Generate Control Run:

% Initial Conditions:
xVec = zeros(4,length(tint)+1);
xVec(4,1) = 0.5;
xVec(3,1) = pi/4;
xVec(1,1) = 5;
xVec(2,1) = 10;
control = zeros(2,length(tint));
NNdyn = fitnet(10);
NNdynV = fitnet(10);

metrics = zeros(tmax/dt,8);


for i = 1:(tmax/dt)
    tic
    if i>5
        disp([i,r(i-1),xVec(:,i)',control(:,i-1)'])
        metrics(i-5,:) = [i,r(i-1),xVec(:,i)',control(:,i-1)'];
    end

    % NN Training:
    if i > 1 %& (mod(i,5)==0 | i<=5)
        if i<100
            NNdyn = fitnet(10);
        end
        NNdyn.trainParam.showWindow=0;
        NNdyn = train(NNdyn, [xVec(:,1:i-1);control(:,1:i-1)], [xVec(:,2:i)]);           
        
    end
    
    
    
  
    % Evaluate N-step residuals:
    % This script only computes residuals after N-steps, giving a
    % conservative result.
    mult = round(i/hor);
    horizon = min([(mult+1),hor]);   
    states = [xVec(:,1:i-1)];   
    conts = control(:,1:i-1);
    nstates = xVec(:,2:i);
    cutoff = 2;  % When to stop using known temporarily safe actions
    
    if i > cutoff
        % Evaluate residuals:
        pred = NNdyn([states;conts]);
        Zpred = Zfun(pred(1,:), pred(2,:));
        Zreal = Zfun(nstates(1,:), nstates(2,:));

        resid = abs(Zpred - Zreal);
        
        
        r = resid; % residuals; %
        N = i; % Number of data samples
        beta = 0.999;% Confidence level, or probability true distribution 
        %  lies within Wasserstein ambiguity set
        rho = 0.0025; % Allowed risk level for CC
    
        % Normalize/center residuals distribution:
        SIG = std(r).^2; 
        mu = mean(r);        
        thet = (SIG.^(-0.5)).*(r - mu); % normalized/centered residuals
        % Compute C:
        options = optimoptions('fmincon','display','none');
        alpha = fmincon(@(alpha)obj_C(alpha, thet, mu, N),1,[],[],[],[],0.001,100,[],options);

        C = 2*alpha;
        Dd = 2*C;%;
        epsilon = Dd*sqrt((2/N)*log10(1/(1-beta))); % Ambiguity set radius  log10

        % Compute \sigma:
        % \sigma is the side length of a DRO hypercube we fit around the
        % empirical residuals distribution:
        sig_low = 0;
        sig_high = 150;
        
        % Compute \sigma via trisection search:
        while (sig_high - sig_low) > 1e-3 
            sig = (sig_high + sig_low)/2;
            [lambda, h_sig_lambda] = triSearch(sig, 0, 100, epsilon, thet);           
            if h_sig_lambda > rho
                sig_low = sig;
            else
                sig_high = sig;
            end
        end

        % Define ambiguity set:    
        r(i) = abs((SIG^0.5)*sig + mu);
    else  % Default value
        r(i) = 3;
    end

    % OVERRIDE DRO:
	override_DRO = 0; % Don't override

%     override_DRO = 1; % Override
%     r(i) = 0; % Override


    
   
    if i > cutoff

        
        % Random Search:
        samples = 500000;
        x = [xVec(:,i)];
        xSim = zeros(4,samples,horizon);
        uSim = zeros(2, samples, horizon);
        Vision = zeros(1,samples,horizon);
        
        % Collect states and control inputs:
        for j = 1:horizon
            
            % Modify maximum inputs if basic state ranges are exceeded:
            if xVec(4,i) > maxV
                umax2 = [0;0.65];
                uSimOpt = umin + 2*umax2.*rand(2, samples);
            elseif xVec(4,i) < -maxV
                umin2 = [0;-0.65];
                uSimOpt = umin2 + 2*umax.*rand(2, samples);
            else
               uSimOpt = umin + 2*umax.*rand(2, samples);
            end

            
            uSim(:,:,j) = uSimOpt;
            
            xSim(:,:,j) = x.*ones(4,samples); 
        end
        
        
        % Simulate 'horizon' steps into the future:
        for j = 1:horizon
            evalX = [xSim(:,:,j);squeeze(uSim(:,:,j))];
            nextState = NNdyn(evalX);
            xSim(:,:,j+1) = nextState;
            Vision(:,:,j) = Zfun(xSim(1,:,j),xSim(2,:,j));
        end
        
        % Calculate objective function values:
        J = -(xSim(1,:,end)) - (xSim(2,:,end));

        % Calculate constraint function values:
        mv = max(Vision,[],3);
        if override_DRO == 0
            indSafe = find(mv <= (OsimLim - r(i)));
        else
            indSafe = find(mv <= (OsimLim));
        end

        % If no predicted feasible actions:
        if isempty(indSafe)==1
            % Default to least unsafe action
            ind_best = find(mv == min(mv)); 
            control(:,i) = uSim(:,ind_best(1),1);
        else       
            % Find best predicted safe action
            FeasCost = J(indSafe);
            FeasAct = uSim(:,indSafe,:);
            bestsafeJ = find(FeasCost == min(FeasCost));
            actInd = FeasAct(:,bestsafeJ(1),:);

            if isempty(actInd)==1
                ind_best = find(mv == max(mv));
                control(:,i) = uSim(:,ind_best(1),1);
            else
                control(:,i) = actInd(:,:,1);
            end
        end
        
        
    else  % If using known safe initial inputs
        control(:,i) = 0.002*randn(2,1);
    end
    
   
        
        
        
    time_v(i,1) = toc;  % Keeping track of time
        
     
    % Test Actual Constraint Violation:
    ValueZ = Zfun(xVec(1,:),xVec(2,:));
    VioInd = ValueZ(1:length(r)) > OsimLim;
    vioind2 = find(VioInd>0);
    if isempty(vioind2) == 0
        disp('UNSAFE')
    end
    
    
    % Simulate actual battery in-the-loop:
    xVec(:,i+1) = env(xVec(:,i), control(:,i),p);

    
  
    % Plot results periodically:
    if mod(i,10)==0
        figure(1)
        clf
        hold on
        contourf(X./10,Y./10,Z2)
        view(2)
        grid on
        plot3(xVec(1,1:i),xVec(2,1:i),3*ones(size(xVec(1,1:i))),'r','Linewidth',2)
        if isempty(vioind2) ~= 1
            scatter3(xVec(1,vioind2), xVec(2,vioind2),3*ones(size(xVec(1,vioind2))),250,'r','filled','p')
        end
        xlabel('X Position [m]')
        ylabel('Y Position [m]')
        xlim([0, 100])
        ylim([0,100])
        legend('Safe Boundary','Boundary','Trajectory','Collisions')        
        drawnow
        

        end

  
    
end % END FOR

save('new_AV_data_00.mat')

%% Functions:


function xnext = env(x, u, p)
a = u(1);
psi = u(2);

dt = p.dt;
L = p.L;


x1 = x(1);
x2 = x(2);
theta = x(3);
v = x(4);


xnext = x + dt.*([v*cos(theta);v*sin(theta);v*tan(psi)/L;a]);

end
