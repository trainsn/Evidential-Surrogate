clear all
close all
% retreive the parameter range; P is an n x 2 matrix (n is the number of parameters);
% P's first row is the minimum and second is maximum of the range
P = para_range();
rng('shuffle')

Z = load('new_para_samples_for_pde');
n = size(Z,1);
% R = rand(n,35);
% Z = 1-2*R(end-n+1:end,:);
Ilog = [1:23,26:35];
P(:,Ilog) = log(P(:,Ilog));


% formula:nchoosek (y+1)*(c_max-c_min)/2 + c_min = x
tmp = (P(2,:)-P(1,:))/2; 
A = ones(n,1) * tmp;
B = ones(n,1) * P(1,:);

para = (Z+1) .* A + B;
para(:,Ilog) = exp(para(:,Ilog));

fid = fopen('parametersets_dat','a');
for i = 1:n
    fprintf(fid,'%g\t',para(i,:));
    fprintf(fid,'\n');
end

fid = fopen('psets_normalized_dat','a');
for i = 1:n
    fprintf(fid,'%g\t',Z(i,:));
    fprintf(fid,'\n');
end

for i = 1 : n
    copyfile('master',num2str(i))
    tmp = para(i,:);
    name = [num2str(i) '/data_new'];
    save(name,'tmp','-ascii')
end