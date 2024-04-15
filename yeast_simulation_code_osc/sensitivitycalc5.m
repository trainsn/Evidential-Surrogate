clear all

% Load data
coeff = load('coeff_d5_eps5e-2_npts9000'); % polynomial coefficients

d = 5; % degree of polynomial space
N = 35; % number of parameters
M = nchoosek(N+d,N); % dimension of polynomial space

% Legendre polynomials and their derivatives
phi{1} = @(x) ones(size(x));
phi{2} = @(x) x;
phi{3} = @(x) (1/2)*(3*x.^2-1);
phi{4} = @(x) (1/2)*(5*x.^3-3*x);
phi{5} = @(x) (1/8)*(35*x.^4 - 30*x.^2 + 3);
phi{6} = @(x) (1/8)*(63*x.^5 - 70*x.^3 + 15*x);
    
phidiff{1} = @(x) zeros(size(x));
phidiff{2} = @(x) ones(size(x));
phidiff{3} = @(x) 3*x;
phidiff{4} = @(x) (1/2)*(15*x.^2 - 3);
phidiff{5} = @(x) (1/8)*(140*x.^3 - 60*x);
phidiff{6} = @(x) (1/8)*(315*x.^4 - 210*x.^2+15);

% Integration of 1D basis polynomials
for i = 1:6
    phiint(i) = integral(phi{i},-1,1,'Arrayvalued',1);
    phidiffint(i)= integral(phidiff{i},-1,1,'Arrayvalued',1);
end
 
% Possible degree combinations for polynomial of total degree up to 5
% c1 = [1 zeros(1,34); 
%     1 1 zeros(1,33); 
%     1 1 1 zeros(1,32); 
%     1 1 1 1 zeros(1,31);
%     1 1 1 1 1 zeros(1,30);
%     2 zeros(1,34); 
%     2 1 zeros(1,33); 
%     2 1 1 zeros(1,32);
%     2 1 1 1 zeros(1,31);
%     2 2 zeros(1,33);
%     2 2 1 zeros(1,32);
%     3 zeros(1,34);
%     3 1 zeros(1,33);
%     3 1 1 zeros(1,32);
%     3 2 zeros(1,33);
%     4 zeros(1,34);
%     4 1 zeros(1,33);
%     5 zeros(1,34)];
% combs = zeros(1,35);
% for i = 1:size(c1,1);
%     combs = [combs; uperm(c1(i,:))];
% end
combs = load('combs');

% Calculate integrals of partial derivatives of basis polynomials
% ints(i,m) is the integral of the ith partial derivative of the mth basis
%   function
for m = 1:M
    c = combs(m,:)+1;
    for i = 1:35
        ints(i,m) = 1; 
        for j = 1:35
            if (i==j)
                ints(i,m) = ints(i,m)*(1/2)*phidiffint(c(j));
            else
                ints(i,m) = ints(i,m)*(1/2)*phiint(c(j));
            end
        end
    end
end

% Calculate sensitivity coefficients
S = zeros(N,1);
for m = 1:M
    S = S + coeff(m)*ints(:,m);
end

% Store sensitivity coefficients
[Y,I] = sort(abs(S));
names = ['k_RL      '
'k_RLm     '
'k_Rd0     '
'k_Rs      '
'k_Rd1     '
'k_Ga      '
'k_G1      '
'k_Gd      '
'k_24cm0   '
'k_24cm1   '
'k_24mc    '
'k_24d     '
'k_42a     '
'k_42d     '
'k_B1cm    '
'k_B1mc    '
'k_Cla4a   '
'k_Cla4d   '
'C24_t     '
'B1_t      '
'C42_t     '
'G_t       '
'R_t       '
'Gbgnq (q) '
'hpower (h)'
'D_R       '
'D_RL      '
'D_G       '
'D_Ga      '
'D_Gbg     '
'D_Gd      '
'D_c24m    '
'D_c42     '
'D_c42a    '
'D_B1m     '];

fid = fopen('sensitivity_d5eps5e-2_npts9000_unsorted','w');
for i = 1:N
    fprintf(fid,'%s\t',names(i,:));
    fprintf(fid,'%g\n', S(i));
end

[y,I] = sort(abs(S));
fid = fopen('sensitivity_d5eps5e-2_npts9000_sorted','w');
for i = 1:N
    fprintf(fid,'%s\t',names(I(i),:));
    fprintf(fid,'%g\n', S(I(i)));
end
    