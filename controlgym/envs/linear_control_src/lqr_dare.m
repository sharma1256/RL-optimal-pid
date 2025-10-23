% Define system matrices
%A = [1.1, 2; 0, 0.95];
%B = [0; 0.0787];
%Q = eye(2); % State cost matrix
%R = 1;      % Control cost matrix

% Solve the Discrete Algebraic Riccati Equation (DARE)
eps = 0.00000000000001;
A_bar = [A,zeros(size(A, 1),size(C, 1)); C , 1] + eps*eye(size(A, 1)+1);
%A_bar = [A,zeros(size(A, 1),size(C, 1)); C , 1] + eps*eye(size(A, 1)+1);
B_bar = [B2;zeros(size(C, 1))];
C_bar = [C,0];
D_bar = 0;
Q = eye(size(A_bar, 1));
Q_mod = [C'*C,zeros(size(A, 1),1);zeros(1,size(A, 1)),0];
R = 1;
[X, L, G] = idare(A_bar, B_bar, Q_mod, R);

disp('Solution to Riccati Equation X:');
disp(X);
disp('Optimal Gain L:');
disp(L);
disp('G:');
disp(G);
