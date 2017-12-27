import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from math import sqrt
import matplotlib.pylab as plt
#Stochastic Gradient Descent to optimize weight vector
def SGD_sol(learning_rate,minibatch_size,num_epochs,L2_lambda,design_matrix,output_data):
    N,M=design_matrix.shape
    weights=np.random.rand(1,M)
    lower_bound=0
    upper_bound=N
    Phi=design_matrix[lower_bound:upper_bound,:]
    t=output_data[lower_bound:upper_bound,:]
    for epoch in range(num_epochs):
        for i in range(int(N / minibatch_size)):
            lower_bound = i * minibatch_size
            upper_bound = min((i+1)*minibatch_size, N)
            E_D = np.matmul((np.matmul(Phi,weights.T)-t).T,Phi)
            E = (E_D + L2_lambda * weights) / minibatch_size
            weights = weights - learning_rate * E
    return weights.flatten()#early stop for gradient descent
#compute centers of input data using Kmeans
def compute_centers_array(num_basis_fun,dataset,dim):
    kmeans = KMeans(num_basis_fun).fit(dataset)
    centersArray=kmeans.cluster_centers_
    clusterPoints=kmeans.labels_
    return centersArray,clusterPoints
#add new axis to centers(3d)
def compute_centers(centersArray):
    centers=centersArray[:,np.newaxis,:]
    return centers
#add new axis to input data(3d)
def newAxisDataSet(dataset):
    nex_axis_dataset=dataset[np.newaxis,:,:]
    return nex_axis_dataset
#Erms error of predicted data against actual output data
def rmse_error(y_actual,y_predicted):
    return sqrt(mean_squared_error(y_actual,y_predicted))
def compute_design_matrix(X,centers,spreads):
    basis_func_outputs=np.exp(np.sum(np.matmul(X-centers,spreads)*(X-centers),axis=2)/-2).T
    return np.insert(basis_func_outputs,0,1,axis=1)
#Closed Form Solution to optimize weight vector
def closed_form_sol(L2_lambda,design_matrix,output_data):
    return np.linalg.solve(L2_lambda * np.identity(design_matrix.shape[1]) +
                          np.matmul(design_matrix.T, design_matrix),
                          np.matmul(design_matrix.T, output_data)
                         ).flatten()
#closed form solution flow: data : letor and synthetic
def CFS_TRAIN_TEST(input_data,output_data):
    #global variables to be used in SGD() function
    global train_input_data
    global validate_input_data
    global test_input_data
    global train_output_data
    global validate_output_data
    global test_output_data
    global weight_vector_ML
    global design_matrix_ML
    global design_matrix_val_ML
    global centers_ML
    global spreads_ML
    global design_matrix_test
    global N
    global V
    global T
    global lambdaVal
    train_input_data,validate_input_data,test_input_data=np.split(input_data, [int(.8 * len(input_data)), int(.9 * len(input_data))])
    train_output_data,validate_output_data,test_output_data=np.split(output_data, [int(.8 * len(output_data)), int(.9 * len(output_data))])
    N,D=train_input_data.shape
    min_error_CFS=float('inf')
    basis_fun_ML=0
    #range of basis functions
    low=5
    high=20
    plotMatRows=high-low
    plotMatColumn=6
    plotMatCFS=np.zeros((high-low,plotMatColumn))
    #iMat and jMat are plot Mat dimensions
    iMat=0
    jMat=0
    start=0
    index=1
    X=newAxisDataSet(train_input_data)
    Vdata=newAxisDataSet(validate_input_data)
    V,D=validate_input_data.shape
    lambdaVal=0.1
    for x in range(low,high):
        centersArray,clusterLabels=compute_centers_array(x,train_input_data,D)
        centers=compute_centers(centersArray)
        spreads=np.zeros((x,D,D))
        spreadsInv=np.zeros((x,D,D))
        clusterVarMatDiag=np.zeros((D,D))
        for i in range(x):
            clusterPoints=train_input_data[np.where(clusterLabels == i)]
            clusterVarMat=np.var(clusterPoints,axis=0).reshape(1,D)
            extraVal = np.ones((clusterVarMat.shape[0],clusterVarMat.shape[1])) * 0.00001
            clusterVarMat[0] = clusterVarMat[0] + extraVal
            clusterVarMatDiag=np.diag(clusterVarMat[0])
            clusterVarMatDiagInv=np.linalg.inv(clusterVarMatDiag)
            spreads[i,:,:]=clusterVarMatDiagInv
        design_matrix=compute_design_matrix(X,centers,spreads)
        #find optimum value of lambda
        startJ=0
        for iMat in range(start,index):
            lambdaVal=0
            for jMat in range(startJ,plotMatColumn):
                lambdaVal=lambdaVal+0.01
                weightVector=closed_form_sol(lambdaVal,design_matrix,train_output_data)
                V,D=validate_input_data.shape
                Vdata=newAxisDataSet(validate_input_data)
                design_matrix_val=compute_design_matrix(Vdata,centers,spreads)
                #Predict validate data set using weight vector computed in training data set
                y_predicted_val=np.matmul(design_matrix_val,weightVector).reshape(V,1)
                error=rmse_error(validate_output_data,y_predicted_val)
                plotMatCFS[iMat][jMat]=error
                if error<min_error_CFS:
                    min_error_CFS=error
                    basis_fun_ML=x
                    weight_vector_ML=weightVector
                    design_matrix_ML=design_matrix
                    design_matrix_val_ML=design_matrix_val
                    centers_ML=centers
                    spreads_ML=spreads
        start=start+1
        index=index+1
        weightVector=closed_form_sol(lambdaVal,design_matrix,train_output_data)
        design_matrix_val=compute_design_matrix(Vdata,centers,spreads)
        field=[]
        #Predict validate data set using weight vector computed in training data set
        y_predicted_val=np.matmul(design_matrix_val,weightVector).reshape(V,1)
        error=rmse_error(validate_output_data,y_predicted_val)
        if error<min_error_CFS:
            min_error_CFS=error
            basis_fun_ML=x
            weight_vector_ML=weightVector
            design_matrix_ML=design_matrix
            design_matrix_val_ML=design_matrix_val
            centers_ML=centers
            spreads_ML=spreads
    #test the trained mod el on test data
    #compute design matrix with M minimum and corresponding weight vector
    minValueCFS=np.min(plotMatCFS)
    T,D=test_input_data.shape
    Tdata=newAxisDataSet(test_input_data)
    design_matrix_test=compute_design_matrix(Tdata,centers_ML,spreads_ML)
    y_predicted_test=np.matmul(design_matrix_test,weight_vector_ML).reshape(T,1)
    errorTest=rmse_error(y_predicted_test,test_output_data)
    print(weight_vector_ML)
#stochastic gradient solution flow: data : letor and synthetic
def SGD_TRAIN_TEST():
    global y_predicted_test_SGD_ES
    #early stopping parameters
    patience_steps=10
    count=0
    num_epochs=10000
    epochs=0
    L2_lambda=0.1
    minibatch_size=N
    error_min_SGD=float('inf')
    error_min_SGD_ES=float('inf')
    optimal_steps=0
    #calculate learning rate for 100 iterations
    initial_iter=100
    threshhold=0.00001
    for n in range(0,9):
        learning_rate=n/10
        weight_vector_SGD=(SGD_sol(learning_rate,minibatch_size,initial_iter,L2_lambda,design_matrix_ML,train_output_data)).T
        y_predicted_val_SGD=np.matmul(design_matrix_val_ML,weight_vector_SGD).reshape(V,1)
        v_error_SGD=rmse_error(y_predicted_val_SGD,validate_output_data)
        if v_error_SGD<error_min_SGD:
            error_min_SGD=v_error_SGD
            learning_rate_ML_SGD=learning_rate
            weight_vector_SGD_ML=weight_vector_SGD
    y_predicted_test_SGD=np.matmul(design_matrix_test,weight_vector_SGD_ML).reshape(T,1)
    errorTest=rmse_error(y_predicted_test_SGD,test_output_data)
    L2_lambda=0.1
    learning_rate_ML_SGD=0.3
    while count<patience_steps:
        epochs=epochs+10
        weight_vector_SGD=(SGD_sol(learning_rate_ML_SGD,minibatch_size,epochs,L2_lambda,design_matrix_ML,train_output_data)).T
        y_predicted_val_SGD=np.matmul(design_matrix_val_ML,weight_vector_SGD).reshape(V,1)
        v_error_SGD=rmse_error(y_predicted_val_SGD,validate_output_data)
        if v_error_SGD-error_min_SGD_ES<threshhold:
            count=count+1
        if v_error_SGD<error_min_SGD_ES:
            y_predicted_val_SGD_ML=y_predicted_val_SGD
            error_min_SGD_ES=v_error_SGD
            weight_vector_SGD_ML=weight_vector_SGD
            optimal_steps=epochs
        if epochs==num_epochs:
            break;
    y_predicted_test_SGD_ES=np.matmul(design_matrix_test,weight_vector_SGD_ML).reshape(T,1)
    error_test_SGD=rmse_error(y_predicted_test_SGD,test_output_data)
    print(weight_vector_SGD_ML)
#main function
letor_input_data=np.genfromtxt('Dataset/Querylevelnorm_X.csv',delimiter=',')
letor_output_data=np.genfromtxt('Dataset/Querylevelnorm_t.csv',delimiter=',').reshape([-1,1])
syn_input_data=np.loadtxt('Dataset/input.csv',delimiter=',')
syn_output_data=np.loadtxt('Dataset/output.csv',delimiter=',').reshape([-1,1])
CFS_TRAIN_TEST(letor_input_data,letor_output_data)
SGD_TRAIN_TEST()
CFS_TRAIN_TEST(syn_input_data,syn_output_data)
SGD_TRAIN_TEST()