import numpy as np

N=747
values=[0,1,2,3,4]
probabilities=[0.5,0.2,0.15,0.1,0.05]
'''Data generation follow Bayesian NonParameteric modeling'''
for i in range(1,11):
    TY=np.loadtxt('data/IHDP/csv/ihdp_npci_'+str(i)+'.csv',delimiter=',')
    treatment=TY[:,0]
    treated = np.where(treatment > 0)[0]
    N_treated=treated.shape[0]
    controlled = np.where(treatment < 1)[0]
    N_control=controlled.shape[0]
    y=TY[:,1]
    yc=TY[:,2]

    matrix = TY[:,5:]

    T=100
    Y=np.zeros((N,T))
    YC=np.zeros((N,T))

    Y[:,0]=y
    YC[:,0]=yc
    for t in range(1,T):
        beta=np.random.choice(values, 25, p=probabilities)
        Y[controlled,t]=np.random.normal(np.dot(matrix[controlled,],beta),1,N_control)+0.02*np.sum(Y[controlled,0:t-1],axis=1)
        Y[treated, t] = np.random.normal(np.dot(matrix[treated,], beta)+4, 1, N_treated) + 0.02 * np.sum(Y[treated, 0:t - 1], axis=1)
        YC[controlled,t]=np.random.normal(np.dot(matrix[controlled,],beta)+4,1,N_control)+0.02*np.sum(YC[controlled,0:t-1],axis=1)
        YC[treated, t] = np.random.normal(np.dot(matrix[treated,], beta), 1, N_treated) + 0.02 * np.sum(YC[treated, 0:t - 1],
                                                                                             axis=1)

    treatment=np.reshape(treatment,(N,1))
    data=np.concatenate((treatment,Y),axis=1)

    y_treated = np.concatenate((YC[controlled], Y[treated]), axis=0)
    y_controlled = np.concatenate((Y[controlled], YC[treated]), axis=0)
    causal_effects = np.mean(y_treated - y_controlled, axis=0)

    np.savetxt('data/IHDP/Series_groundtruth_'+str(i)+'.txt', causal_effects, delimiter=',', fmt='%.2f')
    np.savetxt('data/IHDP/Series_y_'+str(i)+'.txt', data, delimiter=',', fmt='%.2f')
