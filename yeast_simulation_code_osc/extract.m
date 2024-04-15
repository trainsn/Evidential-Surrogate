for i = 1:36
    filename = [num2str(i),'/da_cont'];
    load(filename)
    filename = [num2str(i),'/da_ind'];
    load(filename)
    n = da_ind(1);
    m = length(da_ind)-1;
    ntot = n*18;
    
    for j = 1:n
        k1 = ntot*(m-1);
        R(i,j) = da_cont(k1+j);
        k2 = ntot*(m-1)+4*n;
        Gbg(i,j) = da_cont(k2+j);
        k3 = ntot*(m-1)+8*n;
        C42a(i,j) = da_cont(k3+j);
        k4 = ntot*(m-1)+7*n;
        C42(i,j) = da_cont(k4+j);
    end
end

fid = fopen('R_dat','w');
for ii = 1:size(R,1)
    fprintf(fid,'%g\t',R(ii,:));
    fprintf(fid,'\n');
end
fclose(fid)

fid = fopen('Gbg_dat','w');
for ii = 1:size(R,1)
    fprintf(fid,'%g\t',Gbg(ii,:));
    fprintf(fid,'\n');
end
fclose(fid)

fid = fopen('C42a_dat','w');
for ii = 1:size(R,1)
    fprintf(fid,'%g\t',C42a(ii,:));
    fprintf(fid,'\n');
end
fclose(fid)

fid = fopen('C42_dat','w');
for ii = 1:size(R,1)
    fprintf(fid,'%g\t',C42(ii,:));
    fprintf(fid,'\n');
end
fclose(fid)