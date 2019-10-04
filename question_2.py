import numpy as np
import matplotlib.pyplot as plt


# definition of parameter variables, we selected some random values to test
h = 1
xc = 5
yc = 5

# variance of x
varx = 1        

# variance of y
vary = 1 

# total number of sensors will be (k X k) after meshgrid'ing, hard coded for simplicity       
k = 10

# variance of noise vi
noiseVar = 0.2  


# we initialize f and J arrays to hold paramter values. these f(theta) and J(theta) arrays
f = np.zeros((3,1))
J = np.zeros((3,3))


# generate sensor positions along x and y axes as they are gaussian distributed, we take
Xi = np.random.normal(xc, varx, k)
Yi = np.random.normal(yc, vary, k)


# meshgrid to create a 2-D mesh to place sensors at each cross section of mesh
xi, yi = np.meshgrid(Xi, Yi)
xi = np.reshape(xi, (1, k*k))
yi = np.reshape(yi, (1, k*k))


# generate zi-dash (for simplicity, calling it just zi)-- this is the influence field as defined in the question 
bi = -0.5*(((xi - xc)**2/varx) + ((yi - yc)**2/vary))
gi = h*np.exp(bi)

# assuming noise to be zero mean gaussian
vi = np.random.normal(0, noiseVar, k**2)    
zi = gi + vi


# initialize array of paramter variables and calling it thetha
theta = np.transpose(np.array([np.mean(zi), np.mean(xi), np.mean(yi)]))


## iterating through each sensor and computing estimated paramter using newtown rapson method 
for i in range(10):
  print('\niteration {0:d}'.format(i))
  h  = theta[0]
  xc = theta[1]
  yc = theta[2]

  bi = -0.5*(((xi - xc)**2/varx) + ((yi - yc)**2/vary))


  f[0] = np.sum(np.exp(bi) * (zi - h*np.exp(bi)))
  f[1] = np.sum(h * np.exp(bi) * ((xi - xc)/varx) * (zi - h*np.exp(bi)))
  f[2] = np.sum(h * np.exp(bi) * ((yi - yc)/vary) * (zi - h*np.exp(bi)))


  J[0][0] = -np.sum(np.exp(2*bi))
  #print(bi)
  J[0][1] = np.sum(((xi - xc)/varx) * np.exp(bi) * (zi - 2*h*np.exp(bi)))
  J[0][2] = np.sum(((yi - yc)/vary) * np.exp(bi) * (zi - 2*h*np.exp(bi)))

  J[1][0] = J[0][1]
  J[1][1] = np.sum((h*np.exp(bi)) * ((zi -2*h*np.exp(bi)) * (((xi - xc)/varx)**2) + (zi - (h*np.exp(bi)))*(-1/varx)))
  J[1][2] = np.sum(((xi - xc)/varx)*((yi - yc)/vary)*(h*np.exp(bi))*(zi -2*h*np.exp(bi)))

  J[2][0] = J[0][2]
  J[2][1] = J[1][2]
  J[2][2] = np.sum((h*np.exp(bi)) * ((zi -2*h*np.exp(bi)) * (((yi - yc)/vary)**2) + (zi - (h*np.exp(bi)))*(-1/vary)))

  theta = np.reshape(theta, (3,1))
  theta = theta - np.dot(np.linalg.inv(J), f)
  print(theta)

