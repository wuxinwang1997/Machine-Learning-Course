import numpy as np
import matplotlib.pyplot as plt
import cvxopt

def svm(x, y):
    r,c = x.shape
    #用于二项规划的参数
    H, _, k = kernel(x,y,r)

    H = cvxopt.matrix(H)
    f = cvxopt.matrix(np.ones(r) * -1)
    #形成单位矩阵
    A = cvxopt.matrix(np.diag(np.ones(r) * -1))
    Aeq = cvxopt.matrix(y,(1,r))
    Beq = cvxopt.matrix(np.zeros(r))
    b = cvxopt.matrix(0.0)

    alpha = np.ravel(cvxopt.solvers.qp(H, f, A, Beq, Aeq, b)['x'])
    print(alpha.shape)
    w = np.dot(np.multiply(alpha, y), x)

    counter = 0
    for i in range(len(alpha)):
        if(alpha[i] >= 1e-5):
            counter += 1
            b += y[i] - np.dot(np.multiply(alpha, y), k[i])
    if(counter != 0):
        b = b/counter
    return w, b

def svm_with_c(x,y,c):
    r,_ = x.shape
    H,_,k = kernel(x,y,r)

    H = cvxopt.matrix(H)
    f = cvxopt.matrix(np.ones(r) * -1)
    Aeq = cvxopt.matrix(y,(1,r))
    Beq = cvxopt.matrix(np.zeros(r))
    A = cvxopt.matrix(np.diag(np.ones(r) * -1))
    b = cvxopt.matrix(-c)

    alpha = np.ravel(cvxopt.solvers.qp(H,f,A,Beq, Aeq,b)['x'])
    w = np.dot(np.multiply(alpha,y),x)
    counter = 0
    for i in range(len(alpha)):
        if(alpha[i] >= 1e-5):
            counter += 1
            b += y[i] - np.dot(np.multiply(alpha,y),k[i])
    if(counter > 0):
        b = b/counter
    return w,b


def test(w, b, x, y):
    counter = 0
    y_final = []
    for i in range(len(y)):
        y_final.append((np.dot(w, x[i]) + b)[0][0])
        if(np.sign(np.dot(w, x[i]) + b) == y[i]):
            counter += 1
    return counter/len(y), y_final

def kernel(x,y,r):

    #应该是kij = xi.T*xj,应该可以直接矩阵乘
    H = np.zeros((r,r))
    k = np.zeros((r,r))
    for i in range(r):
        for j in range(r):
            H[i,j] = y[i] * y[j] * np.dot(np.transpose(x[i]), x[j])
            k[i,j] = np.dot(np.transpose(x[i]), x[j])
    return H, H.shape, k


#用于执行交叉测试

if __name__ == '__main__':
    mean_class1 = [1.0, 1.0]
    mean_class2 = [4.5, 4.5]
    cov = [[1,0],[0,1]]
    x_class1 = np.random.multivariate_normal(mean_class1, cov, 100)
    x_class2 = np.random.multivariate_normal(mean_class2, cov, 100)
    y = np.ones(200)
    y[100:200] = -1
    x = np.r_[x_class1, x_class2]
    temp_percent = []
    w, b = svm(x[40:],y[40:])
    percent, y_final = test(w, b, x[:40], y[:40])
    print('w:', w, 'b:',b)
    print(percent)
    temp_percent.append(percent)
    x_train = np.r_[x[:40], x[80:]]
    y_train = np.r_[y[:40], y[80:]]
    w, b = svm(x_train,y_train)
    percent, y_final = test(w, b, x[40:80], y[40:80])
    print('w:', w, 'b:', b)
    print(percent)
    temp_percent.append(percent)
    x_train = np.r_[x[:80], x[120:]]
    y_train = np.r_[y[:80], y[120:]]
    w, b = svm(x_train, y_train)
    percent, y_final = test(w, b, x[80:120], y[80:120])
    print('w:', w, 'b:', b)
    print(percent)
    temp_percent.append(percent)
    x_train = np.r_[x[:120], x[160:]]
    y_train = np.r_[y[:120], y[160:]]
    w, b = svm(x_train, y_train)
    percent, y_final = test(w, b, x[120:160], y[120:160])
    print('w:', w, 'b:', b)
    print(percent)
    temp_percent.append(percent)
    x_train = x[:160]
    y_train = y[:160]
    w, b = svm(x_train, y_train)
    percent, y_final = test(w, b, x[160:], y[160:])
    print('w:', w, 'b:', b)
    print(percent)
    temp_percent.append(percent)

    w, b = svm(x, y)
    percent, y_final = test(w, b, x, y)
    print('w:', w, 'b:', b)
    print(percent)
    temp_percent.append(percent)
    print('the accu for svm mult test is:', temp_percent)
    #plt.scatter(x_class1[:, 0], x_class1[:, 1])
    #plt.scatter(x_class2[:, 0], x_class1[:, 1])
    plt.scatter(x[:,0], np.sign(y_final)/1.05)
    plt.scatter(x[:,0], y)
    plt.show()