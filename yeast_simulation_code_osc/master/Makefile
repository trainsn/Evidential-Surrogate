HOME=/homes/qingnie
FFLAG=
FLIB=  
HEADER=parame.h
OBJECT= mainYP_2d.o forceYP_2d.o const.o B1starGbnstar.o 
#OBJECT= main2d.o force2d.o  
#OBJECT1= const.o ode.o 
OBJECT1= ode.o
job: $(OBJECT) $(OBJECT1)
	gfortran ${FFLAG} -o job ${OBJECT} ${OBJECT1} ${FFLIB} ${FLIB}

$(OBJECT): $(HEADER)
.f.o:
	gfortran ${FFLAG} -c $*.f








