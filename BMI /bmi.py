from __future__ import division
from numpy import *

def computeError(b,m,points):
    totalError=0
    error=0

    for i in range(0, len(points)):
        x=points[i,0]
        y=points[i,1]

        error= (y-(m*x +b))**2
        totalError+= error
    
        print " At row no. {0} , using b {1} , using m {2} ,error {3}" .format(i,b,m,error)
    totalError=totalError/float(len(points))
    print "\n Total error {0}". format(totalError)
    return totalError

def step_gradient (b_current,m_current,points,learning_rate,iteration):
    b_gradient=0
    m_gradient=0
    N=float(len(points))
    for  i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        b_gradient+= -(2/N)*1*(y-(x*m_current +b_current))
        m_gradient+= -(2/N)*x*(y-(x*m_current +b_current))
    new_b=b_current-(learning_rate*b_gradient)

    new_m=m_current-(learning_rate*m_gradient)
    print "\n After {0} iterations the new b  is {1}  & new m  is {2}\n ".format(iteration+1,new_b,new_m)
    # print "   After {0} iterations the  is {1}".format(iteration+1,)
    return [new_b,new_m]

def gradien_descent_runner(points,starting_b,starting_m,learning_rate,num_iterations):
    b1=starting_b
    m1=starting_m
    for i in range(num_iterations):
        b1,m1=step_gradient(b1,m1,array(points),learning_rate,i)
    return [b1,m1]




def amitrix():
    points=genfromtxt('diabetes.csv',delimiter=',')
    learning_rate=0.001
    initial_b=1
    initial_m=1
    num_iterations=100
    computeError(initial_b,initial_m,points)

    [b,m]=gradien_descent_runner(points,initial_b,initial_m,learning_rate,num_iterations)

    print "\n Enter BMI to get Blood Sugar\n"
    X_test=27.2
    print "\n Test/Sample BMI is: {0}\n".format(X_test)
    y_test = m * X_test + b
    print "\n Blood Sugar is {0} \n".format(y_test)



if __name__ == '__main__':
	amitrix()
