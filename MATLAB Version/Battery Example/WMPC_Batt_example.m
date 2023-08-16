%% Wasserstein Distributionally Robust MPC
% Aaron Kandel
% 08/11/2023

%{
This code solves the constrained optimal fast charging problem for a
lithium-ion battery using the equivalent circuit model.

The method used is Wasserstein DRO-MPC:
%}


clear
close all
clc

%% Generate Training Data:

% Define Params:
episodes = 1; % Number of training trajectories
tmax = 500; % [s] max time
dt = 1;  % [s] timestep
tint = 0:dt:tmax; % [s] time vector
I_max = 45; % [A] maximum current
I_min = 0; % [A] minimum current

% Load OCV Data:
VOC_data = csvread('Voc.dat',1,0);
soc = VOC_data(:,1);
voc = VOC_data(:,2);
% Fit cubic polynomial to OCV:
Reg = [ones(length(soc),1),soc,soc.^2,soc.^3]; 
Reg2 = Reg(100:900,:);
theta = inv(Reg2'*Reg2)*Reg2'*voc(100:900);

% figure
% hold on
% plot(soc,voc)
% plot(soc,Reg*theta)
% grid on
% xlabel('SOC [-]')
% ylabel('VOC [V]')


% Battery Parameters:
C_Batt = [2.3*3600];
R_0 = 0.01;
R_1 = 0.01;
C_1 = 2500;
R_2 = 0.02;
C_2 = 7e4;

VsimLim = 3.6; % [V] MAX allowed voltage
z_targ = 0.8; % Target SOC
hor = 8; % Control Horizon Target

 %% Generate Safe Trajectory:
% Ltraj = numel(tint);
% randInp = 10 + 25*rand(Ltraj,1);
% randInp = randInp - (0.025*tint');
% 
% % Simulate safe trajectory:
% SOC_t = 0.2;
% vrc1_t = 0;
% vrc2_t = 0;
% 
% for i = 1:Ltraj
%     
%     
%     
%     SOC_t(i+1,1) = SOC_t(i) + dt/C_Batt(1)*randInp(i,1); %  + (-0.001 + 0.002*rand())
%     VOC_t(i,1) = interp1(soc,voc,SOC_t(i+1));%vocoffset + voc_theta*SOC(i+1);%voc(soc == round(SOC(i+1),3));%           , SOC(i+1)^2, SOC(i+1)^3
%     vrc1_t(i+1,1) = vrc1_t(i) - (dt/(R_1(1)*C_1(1)))*vrc1_t(i) + dt/C_1(1)*randInp(i,1);% + (-0.001 + 0.002*rand()); %  
%     vrc2_t(i+1,1) = vrc2_t(i) - (dt/(R_2(1)*C_2(1)))*vrc2_t(i) + dt/C_2(1)*randInp(i,1);%
%     Vsim_t(i,1) = VOC_t(i) + vrc1_t(i+1) + vrc2_t(i+1) + randInp(i,1).*R_0(1);% + (-0.0025 + 0.005*rand());%0.005*randn();
% end
% 
% safeT = [SOC_t';vrc1_t';vrc2_t';[randInp',0]];

%%
safeT = [];
%% Generate Control Run:
SOC = 0.2;
vrc1 = 0;
vrc2 = 0;
control = 0;
VOC(1) = interp1(soc,voc,SOC); % [V] initial OCV
Vsim(1) = VOC(1) + vrc1(1) + vrc2 + R_2(1)*control(1); 
num_neurons = 3;
NNdyn = fitnet(num_neurons);   % 2
NNdynV = fitnet(num_neurons);  % 2

for i = 1:(tmax/dt)
    tic  % Keep track of iteration time
%     if i>5
%         disp([i,Vsim(i-1),Finv(i-1)])
%     end
    % Training logic that helps:
    if i==5
        NNdyn = fitnet(num_neurons);
        NNdynV = fitnet(num_neurons);
    end

    % Periodically reset model parameters before training: 
    if i > 1 & (mod(i,5)==0 | i<=5) & (i<=100)

         
        NNdyn = fitnet(num_neurons);
        NNdynV = fitnet(num_neurons);
                
    end

    % Train Models:
    NNdyn.trainParam.showWindow=0;
    NNdyn = train(NNdyn, [SOC(1:i-1)';vrc1(1:end-1)';vrc2(1:end-1)';control'], [SOC(2:end)';vrc1(2:end)';vrc2(2:end)']);     

    
    NNdynV.trainParam.showWindow=0;
    NNdynV.trainParam.epochs = 10;
    NNdynV = train(NNdynV, [SOC(1:i-1)';vrc1(1:end-1)';vrc2(1:end-1)';control'], [Vsim']);        

    

    % Horizon increment rule:
    mult = round(i/hor);
    horizon = min([(mult+1),4]);   
    state = [ones(1, length(SOC)); SOC';SOC'.^2;SOC'.^3;vrc1';vrc2'];
    states = [SOC';vrc1';vrc2'];   
    

    % OVERRIDE DRO:
% 	override_DRO = 0; % Don't override
    override_DRO = 1; % Override
%     r(i) = 0;

    % Calculate residuals:
    % This implementation assumes residuals are uncorrelated through 
    % time, so only single-step residuals are computed:  
    if i > 2 && override_DRO==0

        resid = abs(NNdynV([SOC(1:i-1)';vrc1(1:end-1)';vrc2(1:end-1)';control'])-Vsim');
        Vhat = abs(resid(end,1:end));
        disp(max(resid))
        
        re = Vhat; 
        N = i; % Number of data samples
        beta = 0.99;%0.975;% Confidence level, or probability true distribution lies within Wasserstein ambiguity set
        rho = 0.025;%0.0025; % Allowed risk level for chance constraint
    
        % Normalize/center residuals distribution:
        SIG = std(re)^2;
        mu = mean(re);        
        thet = (SIG^(-0.5))*(re - mu); % normalized/centered residuals
        
        % Compute C:
        options = optimoptions('fmincon','display','none');
        alpha = fmincon(@(alpha)obj_C(alpha, thet, mu, N),1,[],[],[],[],0.001,100,[],options);

        C = 2*alpha;
        Dd = 2*C;
        epsilon = Dd*sqrt((2/N)*log10(1/(1-beta))); % Ambiguity set radius 

        % Compute \sigma:
        % \sigma is the side length of a DRO hypercube we fit around the
        % empirical residuals distribution:
        sig_low = 0;
        sig_high = 150;
        
        % Compute \sigma via trisection search:
        while (sig_high - sig_low) > 1e-4
            sig = (sig_high + sig_low)/2;
            [lambda, h_sig_lambda] = triSearch(sig, 0, 50, epsilon, thet);           
            if h_sig_lambda > rho % risk
                sig_low = sig;
            else
                sig_high = sig;
            end
        end

        % Define ambiguity set:    
        r(i) = abs((SIG^0.5)*sig + mu);
    elseif override_DRO==0
        r(i) = Vsim(1) - VOC(1);
    else
        r(i) = 0;
    end

    


    % MPC formulation:
    %{
    minimize sum( (SOC - SOC_targ)^2)
    s. to:
    x(k+1) = f_{NN}(x(k), u(k))
    V(k) = h(x(k),u(k))
    V <= 3.6  
    
    Becomes:

    V + r <= 3.6
    %}
   
    if i > 2 % 2

        
        % Random Search:
        samples = 250000;  % for 1+\lambda evolutionary strategy
        x = [SOC(i);vrc1(i);vrc2(i)];
        xSims = x.*ones(3,samples);
        xSimsPert = xSims;
        uSim = 40*rand(horizon,samples);
        uSimPert = -2.5 + 5*rand(horizon, samples); 
        neg_inds = find(uSimPert<0);
        uSimPert(neg_inds) = rand();
        u1 = uSim;
        u2 = uSim+uSimPert;
        
        for j = 1:horizon
            % Traj. 1:
            evalX = [xSims(:,:,j);uSim(j,:)];
            nextState = NNdyn(evalX);
            xSims(:,:,j+1) = nextState(1:3,:);
            nextV = NNdynV(evalX);
            VsimNN(:,:,j) = nextV;
            
            % Traj. 2 (PoE):
            evalXPert = [xSimsPert(:,:,j);u2(j,:)];
            nextStatePert = NNdyn(evalXPert);
            xSimsPert(:,:,j+1) = nextStatePert(1:3,:);
            nextVPert = NNdynV(evalXPert);
            VsimNNPert(:,:,j) = nextVPert;
        end
        
        J = sum((xSims-z_targ).^2,3);
        J = J(1,:);
        mv = max(VsimNN,[],3);
        
        mvx = max(VsimNNPert,[],3);
        if override_DRO == 0
            indSafe = find((mv <= (VsimLim - r(i))) & (mvx <= (VsimLim - r(i))));
        else
            indSafe = find((mv <= (VsimLim)) & (mvx<=VsimLim));
        end
        
        
        if isempty(indSafe)==1  % No predicted safe actions
            % Pick action with smallest predicted constraint violation
            indMin = find(mv==min(mv));
            control(i,1) = u2(1,indMin(1));
        else
            % Pick best predicted feasible action
            FeasCost = J(indSafe);
            FeasAct = u2(:,indSafe);
            bestsafeJ = find(FeasCost == min(FeasCost));
            actInd = FeasAct(:,bestsafeJ(1));
            control(i,1) = actInd(1);

        end
        
        
    else  % Initial known safe control inputs:
        control(i,1) = 10 + 2.5*rand();
    end
    
        
        
        
        
    time_v(i,1) = toc;  % Keeping track of iteration times
        
        
    
    
    % Simulate actual battery in-the-loop:
    SOC(i+1,1) = SOC(i) + dt/C_Batt(1)*control(i,1); %  + (-0.001 + 0.002*rand())
    VOC(i,1) = interp1(soc,voc,SOC(i+1));%vocoffset + voc_theta*SOC(i+1);%voc(soc == round(SOC(i+1),3));%           , SOC(i+1)^2, SOC(i+1)^3
    vrc1(i+1,1) = vrc1(i) - (dt/(R_1(1)*C_1(1)))*vrc1(i) + dt/C_1(1)*control(i,1);% + (-0.001 + 0.002*rand()); %  
    vrc2(i+1,1) = vrc2(i) - (dt/(R_2(1)*C_2(1)))*vrc2(i) + dt/C_2(1)*control(i,1);%
    Vsim(i,1) = VOC(i) + vrc1(i+1) + vrc2(i+1) + control(i,1).*R_0(1);% + (-0.0025 + 0.005*rand());%0.005*randn();

    

    % Plot results periodically:
    if mod(i,20) == 0
        clf
        subplot(231)
        hold on
        plot(tint(1:i), control, 'Linewidth', 2)
        grid on
        xlabel('Time [s]')
        ylabel('Current [A]')
        legend('DRO-MPC','Location','best')
        subplot(232)
        hold on
        plot(tint(1:i+1), SOC, 'Linewidth', 2)
        plot([0, tint(i+1)], [z_targ, z_targ], '--g', 'Linewidth',2)
        grid on
        legend('DRO-MPC','Target','Location','best')
        xlabel('Time [s]')
        ylabel('SOC [-]')
        subplot(233)
        hold on
        plot(tint(1:i), Vsim, 'Linewidth',2)
        plot([0, tint(i)], [VsimLim, VsimLim], '--k', 'Linewidth',2)
        grid on
        xlabel('Time [s]')
        ylabel('Voltage [V]')
        legend('DRO-MPC','Constraint','Location','best')
        subplot(2,3,[4,5,6])
        plot(tint(1:i), r)
        grid on
        xlabel('Time [s]')
        ylabel('DRO Offset [V]')
        ylim([0,0.4])
        
        drawnow
    end

  
    
end % END FOR


%%

save('batt_ndro_1.mat')




%% Print Metrics:
% mean(time_v)
disp("Proportion of Constraint Violation (x/1.0):")
pcv = length(find(Vsim>3.6))./500
disp("Maximum Voltage:")
vmax = max(Vsim)


