eps = 0.001;
A_bar = [A,zeros(size(A, 1),size(C, 1)); C , 1] + eps*eye(size(A, 1)+1);
B_bar = [B2;zeros(size(C, 1))];
C_bar = [C,0];
D_bar = 0;
Q = eye(size(A_bar, 1));
Q_mod = [C'*C,zeros(size(A, 1),1);zeros(1,size(A, 1)),0];
R = 1;
K = lqr(A_bar, B_bar, Q_mod, R);
sys_cl = ss(A_bar-B_bar*K, B_bar, C_bar, D_bar);
step(sys_cl)

