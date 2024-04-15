clear all

C42a_dat = load('C42a');
N = size(C42a_dat,1);
para = load('parametersets_dat2');
da_xyp = load('master/da_xyp');
yval = da_xyp(:,2);
n = length(yval);
SA = 50.27;
V = 33.51;
for k = 1:N
     c42t = para(k,21);
     A = C42a_dat(k,:);
     C42a_t = sum(A);

    m = 0;
    amt = A(n/2+1);
    Sp = 0;
    while (amt < C42a_t/2)
        m = m + 1;
        amt = sum(A(n/2+1-m:n/2+1+m));
    end
    PF(k) = 1 - 2*((2*m+1)/n);
    
    y(k) = PF(k);
%     
     C42av = c42t/SA;
     %C42av = c42t/V;
     [Cmax,I] = max(A);
     x = Cmax;
     a = 2*(1/C42av);
     y(k) = PF(k)*((a*x)^5/(1+(a*x)^5));
end

fid2 = fopen('PF_C42a_crossval','w');
for i = 1:N
    fprintf(fid2,'%g\n',y(i));
end
fclose(fid2)
