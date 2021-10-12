# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 21:23:27 2021

@author: Maysam
"""
from sklearn.preprocessing import normalize
from scipy.optimize import fminbound
from scipy.sparse import lil_matrix
import scipy.sparse as sp
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse
import sys
import math
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from scipy.sparse.linalg import ArpackNoConvergence
from scipy import linalg
 
from scipy.special import j1
   

def computeLoaderCombine(loader_tr,loader_va,loader_te,appriximate_phsi,N_scales,scales,m,epochs,thr):
     
    """       
    Input: 
    ----------
    loader : 
        loader: input loader (train, validation, or test).
    appriximate_phsi : 
    loader: if True approximate phsi (defualt=False)
    N_scales : 
        number of scales .
    m : 
        Order of polynomial approximation.
    epochs : 
        DESCRIPTION.

    Returns
    -------
    newLoader : 
        DESCRIPTION.

    """
    
    itrData_tr=loader_tr.__next__()
    itrData_va=loader_va.__next__()
    itrData_te=loader_te.__next__()
    
    data_adj=itrData_tr[0]
    
    x_data=itrData_tr[0][0]
    adj=itrData_tr[0][1]
    
    adj2=tf.sparse.to_dense(adj)
    adj2=adj2.numpy()
    
    L = laplacian2(adj2)
    # L=rescale_laplacian(adj2)
    
    
    if appriximate_phsi == True:
        psi,psi_inv=approximate_Psi(L,N_scales,m)
    else:
        psi,psi_inv=compute_Psi(L,scales)
    
    psi=np.asarray(psi)
    psi_inv=np.asarray(psi_inv)
    
    psi[psi<thr]=0
    psi_inv[psi_inv<thr]=0
    
    # newItem=psi[0]
    newItem=psi
    
    data_adj=changeTupleItem(data_adj,1,newItem)
    data_adj=list(data_adj)
    data_adj.append(psi_inv)
    data_adj.append(adj)
    data_adj=tuple(data_adj)
    
    itrData_tr=changeTupleItem(itrData_tr,0,data_adj)
    itrData_va=changeTupleItem(itrData_va,0,data_adj)
    itrData_te=changeTupleItem(itrData_te,0,data_adj)
    
    newLoader_tr=tf.data.Dataset.from_tensors(itrData_tr).repeat(epochs)
    newLoader_va=tf.data.Dataset.from_tensors(itrData_va).repeat(epochs)
    newLoader_te=tf.data.Dataset.from_tensors(itrData_te).repeat(epochs)
    
    return newLoader_tr,newLoader_va,newLoader_te
    


def laplacian2(A, laplacian_type='raw'):
    """Compute graph laplacian from connectivity matrix.
    Parameters
    ----------
    A : Adjancency matrix
    
    Return
    ------
    L : Graph Laplacian as a lil (list of lists) sparse matrix
    """

    N = A.shape[0]
    # TODO: Raise exception if A is not square

    degrees = A.sum(1)
    # To deal with loops, must extract diagonal part of A
    diagw = np.diag(A)

    # w will consist of non-diagonal entries only
    ni2, nj2 = A.nonzero()
    w2 = A[ni2, nj2]
    ndind = (ni2 != nj2).nonzero() # Non-diagonal indices
    ni = ni2[ndind]
    nj = nj2[ndind]
    w = w2[ndind]

    di = np.arange(N) # diagonal indices

    if laplacian_type == 'raw':
        # non-normalized laplaciand L = D - A
        L = np.diag(degrees - diagw)
        L[ni, nj] = -w
        L = lil_matrix(L)
    elif laplacian_type == 'normalized':
        # TODO: Implement the normalized laplacian case
        #   % normalized laplacian D^(-1/2)*(D-A)*D^(-1/2)
        #   % diagonal entries
        #   dL=(1-diagw./degrees); % will produce NaN for degrees==0 locations
        #   dL(degrees==0)=0;% which will be fixed here
        #   % nondiagonal entries
        #   ndL=-w./vec( sqrt(degrees(ni).*degrees(nj)) );
        #   L=sparse([ni;di],[nj;di],[ndL;dL],N,N);
        print("Not implemented")
    else:
        # TODO: Raise an exception
        print("Don't know what to do")

    return L


    
def approximate_Psi(L,N_scales,m):
    """
    approximate wavelet with Chebychev polynomial
    Input:
        L: sparse Laplacian matrix
        N_scales: scale of wavelet
         m: Order of polynomial approximation
    """    
    l_max = rough_l_max(L)
    (g, _, t) = filter_design(l_max, N_scales)
    arange = (0.0, l_max)
    
    c=[]
    for kernel in g:
        c.append(cheby_coeff(kernel, m, m+1, arange))

    # c2=[]
    # for s in range(N_scales+1):
    #     c2.append(cheby_coeff2(m,s+1))

    psi=cheby_op2(L, c, arange)
    
        
    
    psi_inv=[]
    for i in range(N_scales+1):
        psi[i]=np.float32(psi[i])  # convert psi to float 32
        psi_inv.append(np.linalg.inv(psi[i]))
        
    return psi,psi_inv


def compute_Psi(L,scales):
    """
    compute wavelet 
    Input:
        L: sparse Laplacian matrix
        N_scale: scale of wavelet
         m: Order of polynomial approximation
    """    
    lamb, U = fourier(L)
    psi=[]
    psi_inv=[]
    N_scales = len(scales)
    for s in range(N_scales):
        psi.append(weight_wavelet(scales[s],lamb,U))
        psi_inv.append(weight_wavelet_inverse(scales[s],lamb,U))
    del U,lamb

          
    return psi,psi_inv



def rough_l_max(L):
    """Return a rough upper bound on the maximum eigenvalue of L.

    Parameters
    ----------
    L: Symmetric matrix

    Return
    ------
    l_max_ub: An upper bound of the maximum eigenvalue of L.
    """
    # TODO: Check if L is sparse or not, and handle the situation accordingly

    l_max = np.linalg.eigvalsh(L.todense()).max()


    l_max_ub =  1.01 * l_max
    return l_max_ub


def changeTupleItem(tData,idx,newItem):
    """
    change tuple item in index idx with newItem
    Input:
        tData: input tuple
        idx: index must be changed 
        newItem: new Item must be replaced with tData[idx]
    return: output new Tuple
    """
    list_data=list(tData)
    list_data[idx]=newItem
    new_Tuble=tuple(list_data)
    
    return new_Tuble

def adj_matrix():
    names = [ 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format("cora", names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects = pkl.load(f, encoding='latin1')
            else:
                objects = pkl.load(f)
    graph = objects
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj

def laplacian(W, normalized=False):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        # d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L


def rescale_laplacian(L, lmax=None):
    """
    Rescales the Laplacian eigenvalues in [-1,1], using lmax as largest eigenvalue.
    :param L: rank 2 array or sparse matrix;
    :param lmax: if None, compute largest eigenvalue with scipy.linalg.eisgh.
    If the eigendecomposition fails, lmax is set to 2 automatically.
    If scalar, use this value as largest eigenvalue when rescaling.
    :return:
    """
    if lmax is None:
        try:
            if sp.issparse(L):
                lmax = sp.linalg.eigsh(L, 1, which="LM", return_eigenvectors=False)[0]
            else:
                n = L.shape[-1]
                lmax = linalg.eigh(L, eigvals_only=True, eigvals=[n - 2, n - 1])[-1]
        except ArpackNoConvergence:
            lmax = 2
    if sp.issparse(L):
        I = sp.eye(L.shape[-1], dtype=L.dtype)
    else:
        I = np.eye(L.shape[-1], dtype=L.dtype)
    L_scaled = (2.0 / lmax) * L - I
    return L_scaled
   




def cheby_coeff(g, m, N=None, arange=(-1,1)):
    """ Compute Chebyshev coefficients of given function.

    Parameters
    ----------
    g : function handle, should define function on arange
    m : maximum order Chebyshev coefficient to compute
    N : grid order used to compute quadrature (default is m+1)
    arange : interval of approximation (defaults to (-1,1) )

    Returns
    -------
    c : list of Chebyshev coefficients, ordered such that c(j+1) is 
      j'th Chebyshev coefficient
    """
    if N is None:
        N = m+1

    a1 = (arange[1] - arange[0]) / 2.0
    a2 = (arange[1] + arange[0]) / 2.0
    n = np.pi * (np.r_[1:N+1] - 0.5) / N
    s = g(a1 * np.cos(n) + a2)
    c = np.zeros(m+1)
    for j in range(m+1):
        c[j] = np.sum(s * np.cos(j * n)) * 2 / N

    return c


def cheby_coeff2(m,s):
    """ Compute  coefficients of given function.

    Parameters
    ----------
    m : maximum order Chebyshev coefficient to compute
    
    Returns
    -------
    c : list of Chebyshev coefficients, ordered such that c(j+1) is 
      j'th Chebyshev coefficient
      c_{j}=2*e^{-s}J_i(-s)
    """
    c = np.zeros(m+1)
    for j in range(m+1):
            c[j] = 2*np.exp(-s)*j1(-s)
            
    return c
        
def cheby_op2(L, c, arange):
    """Compute (possibly multiple) polynomials of laplacian (in Chebyshev
    basis) applied to input.

    Coefficients for multiple polynomials may be passed as a lis. This
    is equivalent to setting
    r[0] = cheby_op(f, L, c[0], arange)
    r[1] = cheby_op(f, L, c[1], arange)
    ...
 
    but is more efficient as the Chebyshev polynomials of L applied to f can be
    computed once and shared.

    Parameters
    ----------
    f : input vector
    L : graph laplacian (should be sparse)
    c : Chebyshev coefficients. If c is a plain array, then they are
       coefficients for a single polynomial. If c is a list, then it contains
       coefficients for multiple polynomials, such  that c[j](1+k) is k'th
       Chebyshev coefficient the j'th polynomial.
    arange : interval of approximation

    Returns
    -------
    r : If c is a list, r will be a list of vectors of size of f. If c is
       a plain array, r will be a vector the size of f.    
    """
    if not isinstance(c, list) and not isinstance(c, tuple):
        r = cheby_op2(L, [c], arange)
        return r[0]

    # L=tf.sparse.to_dense(L)
    
   
    N_scales = len(c)
    M = np.array([coeff.size for coeff in c])
    max_M = M.max()

    a1 = (arange[1] - arange[0]) / 2.0
    a2 = (arange[1] + arange[0]) / 2.0

    Twf_old = 0
    Twf_cur = (L-a2*np.identity(L.shape[0])) / a1
    r = [0.5*c[j][0]*Twf_old + c[j][1]*Twf_cur for j in range(N_scales)]

    for k in range(1, max_M):
        Twf_new = (2/a1) * (L*Twf_cur - a2*Twf_cur) - Twf_old
        for j in range(N_scales):
            if 1 + k <= M[j] - 1:
                r[j] = r[j] + c[j][k+1] * Twf_new

        Twf_old = Twf_cur
        Twf_cur = Twf_new

    return r




def filter_design(l_max, N_scales, design_type='default', lp_factor=20,
                  a=2, b=2, t1=1, t2=2):
    """Return list of scaled wavelet kernels and derivatives.
    
    g[0] is scaling function kernel, 
    g[1],  g[Nscales] are wavelet kernels

    Parameters
    ----------
    l_max : upper bound on spectrum
    N_scales : number of wavelet scales
    design_type: 'default' or 'mh'
    lp_factor : lmin=lmax/lpfactor will be used to determine scales, then
       scaling function kernel will be created to fill the lowpass gap. Default
       to 20.

    Returns
    -------
    g : scaling and wavelets kernel
    gp : derivatives of the kernel (not implemented / used)
    t : set of wavelet scales adapted to spectrum bounds
    """
    g = []
    gp = []
    l_min = l_max / lp_factor
    t = set_scales(l_min, l_max, N_scales)
    if design_type == 'default':
        # Find maximum of gs. Could get this analytically, but this also works
        f = lambda x: -kernel(x, a=a, b=b, t1=t1, t2=t2)
        x_star = fminbound(f, 1, 2)
        gamma_l = -f(x_star)
        l_min_fac = 0.6 * l_min
        g.append(lambda x: gamma_l * np.exp(-(x / l_min_fac)**4))
        gp.append(lambda x: -4 * gamma_l * (x/l_min_fac)**3 *
                  np.exp(-(x / l_min_fac)**4) / l_min_fac)
        for scale in t:
            g.append(lambda x,s=scale: kernel(s * x, a=a, b=b, t1=t1,t2=t2))
            gp.append(lambda x,s=scale: kernel_derivative(scale * x) * s)
    elif design_type == 'mh':
        l_min_fac = 0.4 * l_min
        g.append(lambda x: 1.2 * np.exp(-1) * np.exp(-(x/l_min_fac)**4))
        for scale in t:
            g.append(lambda x,s=scale: kernel(s * x, g_type='mh'))
    else:
        print("Unknown design type")
        # TODO: Raise exception
        
    return (g, gp, t)


def kernel(x, g_type='abspline', a=2, b=2, t1=1, t2=2):
    """Compute sgwt kernel.

    This function will evaluate the kernel at input x

    Parameters
    ----------
    x : independent variable values
    type : 'abspline' gives polynomial / spline / power law decay kernel
    a : parameters for abspline kernel, default to 2
    b : parameters for abspline kernel, default to 2
    t1 : parameters for abspline kernel, default to 1
    t2 : parameters for abspline kernel, default to 2

    Returns
    -------
    g : array of values of g(x)
    """
    if g_type == 'abspline':
        g = kernel_abspline3(x, a, b, t1, t2)
    elif g_type == 'mh':
        g = x * np.exp(-x)
    else:
        print("unknown type")
        #TODO Raise exception

    return g

def kernel_derivative(x, a, b, t1, t2):
    """Note: Note implemented in the MATLAB version."""
    return x



def kernel_abspline3(x, alpha, beta, t1, t2):
    """Monic polynomial / cubic spline / power law decay kernel

    Defines function g(x) with g(x) = c1*x^alpha for 0<x<x1
    g(x) = c3/x^beta for x>t2
    cubic spline for t1<x<t2,
    Satisfying g(t1)=g(t2)=1

    Parameters
    ----------
    x : array of independent variable values
    alpha : exponent for region near origin
    beta : exponent decay
    t1, t2 : determine transition region


    Returns
    -------
    r : result (same size as x)
"""
    # Convert to array if x is scalar, so we can use fminbound
    if np.isscalar(x):
        x = np.array(x, ndmin=1)

    r = np.zeros(x.size)

    # Compute spline coefficients
    # M a = v
    M = np.array([[1, t1, t1**2, t1**3],
                  [1, t2, t2**2, t2**3],
                  [0, 1, 2*t1, 3*t1**2],
                  [0, 1, 2*t2, 3*t2**2]])
    v = np.array([[1],
                  [1],
                  [t1**(-alpha) * alpha * t1**(alpha - 1)],
                  [-beta * t2**(-beta - 1) * t2**beta]])
    a = np.linalg.lstsq(M, v)[0]

    r1 = np.logical_and(x>=0, x<t1).nonzero()
    r2 = np.logical_and(x>=t1, x<t2).nonzero()
    r3 = (x>=t2).nonzero()
    r[r1] = x[r1]**alpha * t1**(-alpha)
    r[r3] = x[r3]**(-beta) * t2**(beta)
    x2 = x[r2]
    r[r2] = a[0]  + a[1] * x2 + a[2] * x2**2 + a[3] * x2**3

    return r

def set_scales(l_min, l_max, N_scales):
    """Compute a set of wavelet scales adapted to spectrum bounds.

    Returns a (possibly good) set of wavelet scales given minimum nonzero and
    maximum eigenvalues of laplacian.

    Returns scales logarithmicaly spaced between minimum and maximum
    'effective' scales : i.e. scales below minumum or above maximum will yield
    the same shape wavelet (due to homogoneity of sgwt kernel : currently
    assuming sgwt kernel g given as abspline with t1=1, t2=2)

    Parameters
    ----------
    l_min: minimum non-zero eigenvalue of the laplacian.
       Note that in design of transform with  scaling function, lmin may be
       taken just as a fixed fraction of lmax,  and may not actually be the
       smallest nonzero eigenvalue
    l_max: maximum eigenvalue of the laplacian
    N_scales: Number of wavelets scales

    Returns
    -------
    s: wavelet scales
    """
    t1=1
    t2=2
    s_min = t1 / l_max
    s_max = t2 / l_min
    # Scales should be decreasing ... higher j should give larger s
    s = np.exp(np.linspace(np.log(s_max), np.log(s_min), N_scales));

    return s


    

def fourier(L, algo='eigh', k=100):
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""

    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]

    if sp.issparse(L):
        L=L.toarray()


    if algo == 'eig':
        lamb, U = np.linalg.eig(L)
        lamb, U = sort(lamb, U)
    elif algo =='eigh':
        lamb, U = np.linalg.eigh(L)
        lamb, U = sort(lamb, U)
    elif algo == 'eigs':
        lamb, U = scipy.sparse.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo == 'eigsh':
        lamb, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM')

    return lamb, U


def weight_wavelet(s,lamb,U):
    s = s
    for i in range(len(lamb)):
        lamb[i] = math.pow(math.e,-lamb[i]*s)

    Weight = np.dot(np.dot(U,np.diag(lamb)),np.transpose(U))

    return Weight

def weight_wavelet_inverse(s,lamb,U):
    s = s
    for i in range(len(lamb)):
        lamb[i] = math.pow(math.e, lamb[i] * s)

    Weight = np.dot(np.dot(U, np.diag(lamb)), np.transpose(U))

    return Weight






from tensorflow.keras.callbacks import Callback
import time

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        