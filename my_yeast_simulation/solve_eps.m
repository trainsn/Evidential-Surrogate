A = load('Xmat');
y = load('PF_C42a_dat');
n = size(A,1);
d = 5;
m = nchoosek(35+d,35);
y = y(1:n);

path(path, './Optimization');

% Solve for coefficients
x0 = zeros(m,1);
epsilon = 5*1e-2;

%coeff = l1qc_logbarrier(x0, A, [], y, epsilon, 1e-3);

Afun = @(z) A*z;
Atfun = @(z) A'*z;
coeff = l1qc_logbarrier(x0, Afun, Atfun, y, epsilon, 1e-3, 50, 1e-8, 500);

fid = fopen('coeff_d5_eps5e-2','w');
fprintf(fid,'%g\n',coeff);
fclose(fid)
