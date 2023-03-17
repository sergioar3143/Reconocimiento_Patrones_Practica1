import math
import numpy
import matplotlib.pyplot as plt

def get_value(V, index):
    global num
    if index<0 or index>=num: ##Se definen las condiciones iniciales y en caso contrario se da el valor del vector
        val=0
    else:
        val=V[index]
    return val

def get_vector(V, index): ##Parecida a la funcion get_value pero se entrega un vector
    global order
    if index<0:
        vec=numpy.zeros(order)
    else:
        vec=V[index:index+order]
    return vec

def r_value(x,i, psi): ## Se reliza la operación x_i*x_i+psi
    r=get_value(x,i)*get_value(x,i+psi)
    return r

def Matrix_R(x): ##Se obtiene la matriz de correlación
    global order,num
    M=numpy.zeros((order,order))
    for i in range(order):
        for j in range(order):
            for n in range(num):
                M[i,j]=M[i,j]+r_value(x,n,j-i)
            M[i,j]=M[i,j]/num
    return M

def vector_r(x): ##Se obtiene un vector de correlación
    global order,num
    M=numpy.zeros(order)
    for i in range(order):
        for n in range(num):
            M[i]=M[i]+r_value(x,n,-1-i)
        M[i]=M[i]/num
    return M

def system(select, mu, delta): ##Delta corresponde a la variable con la cual se controla la cantidad de ruido en el sistema
    global order,num
    mean = 0 ## Media de la distribución gaussiana
    std = 5  ## Varianza de la distribución
    v=numpy.random.normal(mean, std, size=num) ##Función que genera el vector aleatorio
    x=numpy.zeros(num) ##Arreglo de la salida del sistema
    E=numpy.zeros(num)  ##Areglo que representa el error en el tiempo

    y=numpy.zeros(num) ###Sistema observado, se le agrega ruido
    l=numpy.random.normal(mean, std, size=num) ##ruido agregado al sistema
    #Filtros adaptables
    if select ==1:
        order=8
        w=numpy.zeros((num+1,order))  ##Vector que representa la evolución de los pesos W a lo largo del tiempo
        for n in range(num):
            x[n]=get_value(v,n)-0.3*get_value(x,n-1)+0.67*get_value(x,n-2)+0.075*get_value(x,n-3)+0.3214*get_value(x,n-4)
            x[n]=x[n]+0.1179*get_value(x,n-5)-0.2704*get_value(x,n-6)-0.033945*get_value(x,n-7)+0.0387*get_value(x,n-8)

            y[n]=x[n]+delta*l[n] ##Ruido agregado al sistema
            ##E[n]=x[n]-numpy.dot(w[n,:], numpy.flipud(get_vector(x,n-order))) Error sin agregar ruido

            E[n]=y[n]-numpy.dot(w[n,:], numpy.flipud(get_vector(y,n-order))) ##Calculo del error
            ##w[n+1,:]=w[n,:]+mu*E[n]*numpy.flipud(get_vector(x,n-order)) Actualizacion de los pesos W sin agregar error

            w[n+1,:]=w[n,:]+mu*E[n]*numpy.flipud(get_vector(y,n-order)) ##Actualización de los pesos W
            ##La función numpy.flipud cambia el orden del arreglo de x=[1,2] a x=[2,1]
        print("Valores reales: -0.3 0.67 0.075 0.3214 0.1179 -0.27 -0.034 0.0387")
        print("Solucion por filtros adaptables"+str(w[num,:]))
        fig, axs = plt.subplots(4)
        fig.suptitle('W1,W2,W3,W4')

        fig2, axs2 = plt.subplots(5)
        fig2.suptitle('W5,W6,W7,W8,E')

        fig3, axs3 = plt.subplots(2)
        fig3.suptitle('Señal original, señal generada los pesos de wiener')

        axs[0].plot(w[:,0])
        axs[1].plot(w[:,1])
        axs[2].plot(w[:,2])
        axs[3].plot(w[:,3])

        axs2[0].plot(w[:,4])
        axs2[1].plot(w[:,5])
        axs2[2].plot(w[:,6])
        axs2[3].plot(w[:,7])
        axs2[4].plot(numpy.abs(E))

        axs3[0].plot(x)
        axs3[1].plot(y-E)

    else:
        order=2
        w=numpy.zeros((num+1,order))  ##Vector que representa la evolución de los pesos W a lo largo del tiempo
        for n in range(num):
            x[n]=0.653*get_value(x,n-1)-0.7001*get_value(x,n-2)+get_value(v,n) ##Se genera la salida de acuerdo al sistema

            y[n]=x[n]+delta*l[n] ##Ruido agregado al sistema
            ##E[n]=x[n]-numpy.dot(w[n,:], numpy.flipud(get_vector(x,n-order))) Error sin agregar ruido

            E[n]=y[n]-numpy.dot(w[n,:], numpy.flipud(get_vector(y,n-order))) ##Calculo del error
            ##w[n+1,:]=w[n,:]+mu*E[n]*numpy.flipud(get_vector(x,n-order)) Actualizacion de los pesos W sin agregar error

            w[n+1,:]=w[n,:]+mu*E[n]*numpy.flipud(get_vector(y,n-order)) ##Actualización de los pesos W

        print("Valores reales: 0.653 -0.7001")
        print("Solucion por filtros adaptables"+str(w[num,:]))
        fig, axs = plt.subplots(3)
        fig.suptitle('W1,W2,E')

        fig2, axs2 = plt.subplots(2)
        fig2.suptitle('Señal original, señal generada los pesos de wiener')

        axs[0].plot(w[:,0])
        axs[1].plot(w[:,1])
        axs[2].plot(numpy.abs(E))

        axs2[0].plot(x)
        axs2[1].plot(y-E)

    return y ##Se regresa la señal con ruido observada

def main():
    global order, num
    mu=0.000065 ##Tasa a la que se adaptan los filtros
    num = 2000 ## Número de muestras
    sel=1 ##Se escoge un uno o dos dependiendo del sistema que se quiere usar
    #sel=2
    ##Filtros adaptables
    #delta=0
    delta=0.1
    x=system(sel,mu,delta)
    ##Wiener
    a = Matrix_R(x)
    b = vector_r(x)

    print("Solucion usando wiener:"+str(numpy.matmul(numpy.linalg.inv(a), b)))
    plt.show()

if __name__ == '__main__':
    main()
