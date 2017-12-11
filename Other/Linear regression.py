from numpy import *
#How to do linear regression with gradien descent with siraj raval

def compute_error_for_giver_error(b,m,points):
    totalError=0
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        totalError += (y-(m*x+b))**2
    return totolError / float(len(points))


def step_gradient(b_current,m_current,points,learning_rate):
    b_gradient=0
    m_gradient=0
    N=float(len(points))
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        b_gradient+= -(2/N)*(y-((m_current*x)+b_current ))
        m_gradient+= -(2/N)*x*(y-((m_current*x)+b_current))
    new_b=b_current-(learning_rate * b_gradient)
    new_m=m_current-(learning_rate*m_gradient)
    return [new_b,new_m]

def gradient_descent_runner(points,starting_b,starting_m,learning_rate,num_iterations):
    b=starting_b
    m=starting_m
    b,m=step_gradient(b,m,array(points),learning_rate)
    return [b,m]
    


def run():
    points=genfromtxt('data.csv',delimiter=',')
    learningRate=0.0001 #hyperparameters this defines how fast our model learns
    #if learning Rate is very low  , it will be very slow to converge else if it is very high then it will never converge
    #y=mx +b
    initial_b =0
    initial_m =0
    num_iterations=1000
    [b,m]=gradient_descent_runner(points,intial_b,initial_m,learning_rate,num_iterations)
    print(b)
    print(m)


if __name__=='__main__':
    run()
    
