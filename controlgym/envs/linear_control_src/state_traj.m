% Sampling time (discrete system)
Ts = 0.01;  % 0.1 seconds

% Create the discrete-time state-space model
sys = ss(A_bar, B_bar, C_bar, D_bar, Ts);

% Define initial conditions (e.g., x(0) = [1; 0])
x0 = ones(size(A_bar,1),1);

% Time vector for simulation
t = 0:Ts:10;  % From 0 to 10 seconds with step size of Ts

% Simulate the response to the initial condition
[y, t, x] = initial(sys, x0, t);  % 'x' contains the state trajectories

% Plot the state trajectories
figure;
plot(t, x(:,1), 'r', 'DisplayName', 'State 1 (x_1)');
hold on;
plot(t, x(:,2), 'b', 'DisplayName', 'State 2 (x_2)');
xlabel('Time (s)');
ylabel('State Values');
title('State Trajectories for Discrete-Time System');
legend;
grid on;
