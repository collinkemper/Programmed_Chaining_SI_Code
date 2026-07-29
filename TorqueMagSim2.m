function TorqueMagSim2

%Set parameters
K = 8.81e-12;        %anisotropy energy (J)
B = 0.020;           %magnetic field (T)
m = 1.56e-8;         %magnetic moment (A m^2)

theta = 0:1:360;  %list of anglular positions
alpha = nan(size(theta));  %preallocate a list of alphas

%establish the torque function
f = @(alpha, theta) K*sin(2*alpha)-m*B*sin(deg2rad(theta)-alpha);

%find the alpha within [0, 2pi] that minimizes torque for the first theta
alpha(1) = fzero(@(xx) f(xx,theta(1)), [0 2*pi]);
%find all the other alphas, using each value as the starting point for the next
for i = 2:length(theta)
    guess = min(max(alpha(i-1), 0), 2*pi); %require the guess to lie between limits [0, 2pi]
    alpha(i) = fzero(@(xx) f(xx, theta(i)), guess);
end

%calculate the magnetic torque for each theta
torque = m*B*sin(deg2rad(theta)-alpha)';

% Plot the torque as a function of angle
figure(1);
hold on
plot(theta, torque);
xlim([0,360])
xticks(0:45:360)
xlabel('Angle (degrees)');
ylabel('Torque (Nm)');
title('Magnetic Torque vs Angle');
grid on;
text(0.80, 0.95, sprintf('K/MB = %.3f', K/(m*B)), 'Units', 'normalized','VerticalAlignment', 'top', 'FontSize',11);

%Plot alpha vs theta
figure(2);
plot(theta, rad2deg(alpha));
xlim([0,360])
xticks(0:45:360)
xlabel('Theta (degrees)');
ylabel('Alpha (degrees)');
title('Alpha vs Theta');
grid on;
text(0.60, 0.95, sprintf('alpha/theta @ 45 deg = %.3f', rad2deg(alpha(45))/theta(45)), 'Units', 'normalized','VerticalAlignment', 'top', 'FontSize',11);