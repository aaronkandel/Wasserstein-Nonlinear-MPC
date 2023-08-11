%% Wasserstein Distributionally Robust MPC
% Plotting/Animating Code
% Aaron Kandel
% 08/11/2023

%{
This code plots results from simulations of W-MPC.
%}

load batt_data_run_00.mat

h = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = 'batt_traj_test.avi';

j=1;
for i=1:length(tint)
    if mod(i-1,5) == 0
        clf
        subplot(4,1,1)
        hold on
        grid on
        plot(tint(1:i)./60, control(1:i), 'Linewidth', 2)
        scatter(tint(i)./60, control(i), 'filled')
        grid on
        xlabel('Time [min]')
        ylabel('Current [A]')
        xlim([0,8])
        ylim([0,30])
        subplot(412)
        hold on
        plot(tint(1:i)./60, SOC(1:i), 'Linewidth', 2)
        scatter(tint(i)./60, SOC(i), 'filled')
        plot([0, 8], [z_targ, z_targ], '--g', 'Linewidth',2)
        grid on
    %     legend('DRO-MPC','Target','Location','best')
        xlabel('Time [min]')
        ylabel('SOC [-]')
        xlim([0,8])
        subplot(413)
        hold on
        plot(tint(1:i)./60, Vsim(1:i), 'Linewidth',2)
        scatter(tint(i)./60, Vsim(i), 'filled')
        plot([0, tint(i)./60], [VsimLim, VsimLim], '--k', 'Linewidth',2)
        plot([0, tint(i)./60], [VsimLim-r(i), VsimLim-r(i)], '--k', 'Linewidth',2)
        grid on
        xlabel('Time [s]')
        ylabel('Voltage [V]')
    %     legend('DRO-MPC','Constraint','Location','best')
        xlim([0,8])
        subplot(414)
        hold on
        plot(tint(1:i)./60, r(1:i))
        scatter(tint(i)./60, r(i), 'filled')
        set(gca, 'YScale', 'log')
        grid on
        xlabel('Time [s]')
        ylabel('DRO Offset [V]')
%         ylim([0,2])
        xlim([0,8])
        drawnow





        % Capture the plot as an image 
        f(j) = getframe(h); 
        j=j+1;

    end
end


writerObj = VideoWriter(filename);
writerObj.FrameRate=10;
open(writerObj);
for i=1:length(f)
    frame = f(i);
    writeVideo(writerObj, frame);
end

close(writerObj)

subplot(231)
hold on
plot(tint(1:i-1)./60, control, 'Linewidth', 2)
grid on
xlabel('Time [min]')
ylabel('Current [A]')
legend('DRO-MPC','Location','best')
subplot(232)
hold on
plot(tint(1:i)./60, SOC, 'Linewidth', 2)
plot([0, tint(i)]./60, [z_targ, z_targ], '--g', 'Linewidth',2)
grid on
legend('DRO-MPC','Target','Location','best')
xlabel('Time [min]')
ylabel('SOC [-]')
subplot(233)
hold on
plot(tint(1:i-1)./60, Vsim, 'Linewidth',2)
plot([0, tint(i-1)]./60, [VsimLim, VsimLim], '--k', 'Linewidth',2)
grid on
xlabel('Time [min]')
ylabel('Voltage [V]')
legend('DRO-MPC','Constraint','Location','best')
subplot(2,3,[4,5,6])
Finv2 = r;
indf = find(Finv2>0.4);
Finv2(indf) = 0.4;
semilogy(tint(1:i-1)./60, Finv2,'Linewidth',2)
grid on
xlabel('Time [min]')
ylabel('DRO Offset [V]')