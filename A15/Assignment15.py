# Script only to calculate the System response time (class -indipendent) but without think time

#Values taken directly fomr the JMVA

# Class A response time, excluding think time
RA = 535.4408

# Class B response time, excluding think time
RB = 540.6370

# Class C response time, excluding think time
RC = 243.2171

# Throughputs
XA = 0.0367
XB = 0.0055
XC = 0.3333

X = XA + XB + XC

#Normalizing the contributions 
R = RA * XA / X + RB * XB / X + RC * XC / X

print("Class independent system response time: ", R)