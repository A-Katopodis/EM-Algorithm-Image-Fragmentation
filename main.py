from __future__ import division
import numpy as np
from PIL import Image


# calculate the g(znk)
def get_Z(N, K, f, maximum):
    # substract the maximum from the f
    for k in range(K):
        f[:,k] = f[:,k] - maximum
    z = np.zeros((N, K))
    f = np.exp(f)

    denominator = np.sum(f, axis=1)

    for k in range(K):
        z[:, k] = f[:, k] / denominator
    return z


# calculate the f array. With shape NxK
def calculate_f(X, m, s, p):
    K = s.shape[0]
    f = np.zeros((X.shape[0], K))
    for k in range(K):
        pk = p[k]
        mk = m[k,:]
        sk = s[k]
        x_m = (X - mk)**2

        temp = np.sum((x_m / sk) + np.log(2*np.pi*sk), axis=1)
        f[:, k] = np.log(pk) - 0.5*temp
    # get the maximum of all f
    maximum = f.max(axis=1)
    return f, maximum


# initialize the values depending on K
def initialize_values(d, k):

    m = np.full((k, d), -1.0)

    pk = np.full(k, 1/k)
    # m[0, :] = 0
    # m[k-1, :] = 1

    s = np.random.uniform(0.3,0.7,k)
    # s = np.full(k,1)
    # m[:] = 127.5
    for i in range(k):
        m[i,:] = np.random.uniform(0.1,0.9,d)
        # s[i] = sum
    # print m
    # print "----------------------"
    # print s
    s = np.asarray(s)
    m = np.asarray(m)
    return m, s, pk


# calculate the logarithmic likehood
def get_L(f, maximum):
    temp = np.sum(np.exp(f-maximum[:,None]), axis=1)
    a = np.log(temp)
    return np.sum(maximum + a, axis=0)


def get_new_values(X, z):
    N = X.shape[0]
    K = z.shape[1]
    D = X.shape[1]
    m_new = np.zeros((K, D))
    p_new = np.zeros(K)
    s_new = np.zeros(K)
    z_sum = np.sum(z, axis=0)
    for k in range(K):

        zk = z[:, k]
        # calculate the new m
        for d in range(D):
            m_new[k, d] = np.sum(X[:,d] * zk, axis=0) / z_sum[k]

        # calculate the new sigma
        x_m = X - m_new[k, :]
        x_m = x_m**2
        x_m = np.sum(x_m, axis=1)

        s_new[k] = np.sum(zk*x_m, axis=0) / (z_sum[k]*X.shape[1])

        # calculate the new p
        p_new[k] = z_sum[k]* (1/ N)
    return m_new, s_new, p_new


def EM(X, k=2):
    N, D = X.shape

    # init the values randomly
    m, s, p = initialize_values(D, k)
    tol = 1e-6

    # the initial f
    f, maximum = calculate_f(X, m, s, p)

    print ("For k = ",k)
    for i in range(300):
        print ("\tIteration: ", i)

        # Expectation
        L_old = get_L(f, maximum)

        z = get_Z(N, k, f, maximum)

        m, s, p = get_new_values(X, z)

        # maximization
        f, maximum = calculate_f(X, m, s, p)
        L_new = get_L(f, maximum)

        if L_new - L_old < 0:
            # print L_old
            # print L_new
            print ("Logarithmic likelyhood not decreasing!")
            exit()
        if np.abs(L_new - L_old) < tol:
            print ("Reached tolerance for: ", k)

            return z, m

    return z, m


def run_experiments():
    k=2
    while k <= 64 :
        # calculate optimal z,m
        z, m = EM(X, k)

        # we change transform the m to the original
        # RGB value
        m = m * 255
        m = m.astype(np.uint8)

        # Our new image
        rec_img = np.zeros(X.shape)
        rec_img = rec_img.astype(np.uint8)
        # give each pixel the color of the highest z
        for n in range(X.shape[0]):
            indice = z[n].argmax()
            rec_img[n] = m[indice]

        N = X.shape[0]
        X_true = X
        X_r = rec_img / 255
        # calculate the error
        dist = np.sum((X_true - X_r)**2)
        error = dist / N
        print ("Error for ", k," is:", error)

        # return the image to it's original dimensions
        rec_img = rec_img.reshape((height, width, 3))
        rec_img = Image.fromarray(rec_img, 'RGB')
        img_name = "rec_img_" + str(k)+".jpg"
        rec_img.save(img_name)
        k *= 2



img = Image.open("im.jpg", mode='r')


data = np.asarray(img)

print ("Image shape: ", data.shape)
height = data.shape[0]
width = data.shape[1]
pixels = data.shape[0] * data.shape[1]
values = data.shape[2]

# NxD numpy array
# N = The total number of pixels
# D = Red, Green, Blue
data = data.reshape((pixels,values))

print ("The total pixels are: ", pixels)

# we normalize the data
X = data / 255

run_experiments()
