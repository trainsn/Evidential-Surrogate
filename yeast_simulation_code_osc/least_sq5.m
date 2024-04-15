% Legendre polynomials
phi{1} = @(x) 1;
phi{2} = @(x) x;
phi{3} = @(x) (1/2)*(3*x.^2-1);
phi{4} = @(x) (1/2)*(5*x.^3-3*x);
phi{5} = @(x) (1/8)*(35*x.^4 - 30*x.^2 + 3);
phi{6} = @(x) (1/8)*(63*x.^5 - 70*x.^3 + 15*x);

% Determine combinations
d = 5;
m = nchoosek(35+d,35);
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
% 
% fid = fopen('combs','w');
% for i = 1:m
%     fprintf(fid,'%g\t',combs(i,:));
%     fprintf(fid,'\n');
% end

combs = load('combs');

% Load data
Spts = load('psets_normalized_dat'); %Spts
n = 6600;
Spts = Spts(1:n,:);

% Build matrix
X = zeros(n,m);
for j = 1:m
    c = combs(j,:)+1;
    X(:,j) = 1;
    for k = 1:35
        X(:,j) = X(:,j).*phi{c(k)}(Spts(:,k));
    end
end
disp('built')

fid = fopen('Xmat','w');
for i = 1:n
    fprintf(fid,'%g\t',X(i,:));
    fprintf(fid,'\n');
end
disp('done')
fclose(fid)
