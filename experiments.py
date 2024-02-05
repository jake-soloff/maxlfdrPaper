import numpy as np

from scipy import stats
from scipy.optimize import bisect,minimize_scalar
from scipy.misc import derivative

from sklearn import isotonic

import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams.update({
    'lines.linewidth' : 1.,
    'lines.markersize' : 5,
    'font.size': 9,
    "text.usetex": True,
    'font.family': 'serif', 
    'font.serif': ['Computer Modern'],
    'text.latex.preamble' : r'\usepackage{amsmath,amsfonts}',
    'axes.linewidth' : .75})


### color palette
lblue = "#2866B2" # Green blue
cred = "#B1040E" # Crimson
nyllw = '#F5D547' # Naples yellow
bgray = '#7c898b' # battleship gray
tblue = '#75DDDD' # Tiffany blue

import os
from collections import defaultdict
import pickle as pkl

tt = np.linspace(0, 1, 1000)

def plot_arrows(fig, ax):
    # credit: https://3diagramsperpage.wordpress.com
    #           /2014/05/25/arrowheads-for-axis-in-matplotlib/
    
    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()

    # manual arrowhead width and length
    hw = 1./50.*(ymax-ymin) 
    hl = 1./50.*(xmax-xmin)
    lw = plt.rcParams['axes.linewidth'] # axis line width
    ohg = 0.1 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
             head_width=hw, head_length=hl, overhang = ohg, 
             length_includes_head= True, clip_on = False) 

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
             head_width=yhw, head_length=yhl, overhang = ohg, 
             length_includes_head= True, clip_on = False) 


######################### Figure 1 - SL procedure demo #########################
# setting (E) of BH 1995
m = 64 
pi0 = 3/4 
m0 = int(pi0*m)
m1 = m-m0
L = 5 
m1 = m-m0
mu = np.zeros(m)
for j, sp in enumerate(np.split(np.arange(m1),4)):
    mu[sp] = (j+1)*L/4
H = 1-(mu==0)

q = .25

np.random.seed(123)
x = np.random.randn(m) + mu
p = stats.norm.sf(x)
p_ = np.sort(p)
p__ = np.hstack([0, p_])
H_= H[np.argsort(p)]
T = np.arange(m+1)/m

# jitter this point for plotting purposes
p__[15]+=.004
p__[16]+=.005
ell = .5
loc = np.argmin(p__-ell*T)
BH = np.max(np.where(p__ <= q*T)[0])

fig, ax = plt.subplots(figsize=(3.5,2.6)) 

tau_bh = np.mean(p <= p__[BH])/q
xmax = T[sum(p__<=q)]

ax.plot(T[1:], p__[1:], 'k.')

tt = np.linspace(0, 1, 100000)
ax.plot(tt, tt*q, '-', c=cred)
ax.plot(tt, tt*ell + np.min(p__-ell*T), c=lblue, ls='-')

xlabs = [1, 2, '', '$\\cdots$', ''] + ['']*(int(xmax*m)-8) + ['Index $k$', '']
ax.set_xticks(T[1:(len(xlabs)+1)])
print(len(T[1:]), len(xlabs))
ax.set_xticklabels(xlabs) 

k = int(xmax*m)-2
ax.set_yticks([0])
ax.set_ylabel('Order statistics $p_{(k)}$')

ax.plot([T[loc]]*2, [0, p__[loc]], c=lblue, ls='-', alpha=0.5, lw=1.5)
ax.plot([T[BH]]*2, [0, q*T[BH]], c=cred, ls='-', alpha=0.5, lw=1.5)

ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')

ax.get_xticklines()[2*(BH-1)].set_markeredgewidth(1.5)
ax.get_xticklines()[2*(BH-1)].set_color(cred)

ax.get_xticklines()[2*(loc-1)].set_markeredgewidth(1.5)
ax.get_xticklines()[2*(loc-1)].set_color(lblue)

ax.text(T[BH-1]+.004, -.0265, '$R_q^{\\text{BH}}$', c=cred)
ax.text(T[loc-1]+.004, -.0265, '$R_\\ell$', c=lblue) 

ax.text(T[k]-.12, T[k]*q-.06, 'Slope $\\ell/m$', c=lblue)
ax.text(T[k]-.35, T[k]*q+.001-.05, 'Slope $q/m$', c=cred)

ax.set_ylabel('Order statistics $p_{(k)}$')
ax.yaxis.set_label_coords(-0.01,.6)

ax.set_ylim([-.06, q+.01])
ax.set_xlim([-.0, xmax+.005])

plot_arrows(fig, ax)

plt.tight_layout()
plt.savefig('1.pdf')

######################### Figure 2 - Population regret #########################
from matplotlib.patches import Rectangle
title_proxy = Rectangle((0,0), 0, 0, color='w')
markevery=2000

a = np.linspace(1e-8, .5, 10000) 

pi0 = .8
alpha = .2

tau = ((1/alpha - 1)*pi0/(a*(1-pi0)))**(-1/(1-a))
tau_1 = ((1/alpha - pi0)/(a*(1-pi0)))**(-1/(1-a))

colors = [tblue, cred, nyllw, bgray]
plt.figure(figsize=(2.8,2.8))
qs = [alpha/pi0, alpha, alpha/2, alpha/10]
qs_label = ['\\alpha/\\pi_0', '\\alpha', '\\alpha/2', '\\alpha/10']
markers = ['D', 'o', '>', '|']
ls = []
for cc, q_ in enumerate(qs):
    q = q_*pi0
    Tq = ((1/q - 1)*pi0/(1-pi0))**(-1/(1-a))
    regret = ((pi0*tau + (1-pi0)*tau**a-(pi0/alpha)*tau)
              -(pi0*Tq+(1-pi0)*Tq**a-(pi0/alpha)*Tq))
    l, = plt.plot(a, regret, color=colors[cc], 
                  marker=markers[cc], markevery=markevery, markersize=5)
    ls.append(l)
    
regret = ((pi0*tau + (1-pi0)*tau**a-(pi0/alpha)*tau)
          -(pi0*tau_1+(1-pi0)*tau_1**a-(pi0/alpha)*tau_1))
l1, = plt.plot(a, regret, '-*', color=lblue, markevery=markevery, markersize=5)

regret = ((pi0*tau + (1-pi0)*tau**a-(pi0/alpha)*tau)
          -(pi0*tau+(1-pi0)*tau**a-(pi0/alpha)*tau))
l2, = plt.plot(a, regret, '--', color='k')

levels = ['$\\alpha = 0.2$', '$\\alpha/\\pi_0 = 0.25$']
plt.legend([title_proxy] + ls + [title_proxy] + [l1, l2], 
           ['BH at level'] + ['$\\alpha/\\pi_0 = 0.25$', '$\\alpha = 0.2$'] 
           + ['$\\alpha/2 = 0.1$', '$\\alpha/10 = 0.02$'] 
           + ['SL at level'] + levels, fontsize=8)

plt.ylabel('Population regret')
plt.xlabel('Inverse signal strength $\\theta$,\nwhere $f_1$=Beta$(\\theta,1)$')

plt.gca().spines['top'].set_color('none')
plt.gca().spines['bottom'].set_color('none')
plt.gca().spines['left'].set_position('zero')
plt.gca().spines['bottom'].set_position('zero')
plt.gca().spines['right'].set_color('none')

plt.xlim([0, np.max(a)])
plt.ylim([-.01, .2])
plt.yticks([0, .04, .08, .12, .16, .2])
plt.tight_layout()
plt.savefig('2a.pdf')

bayes_risk_abs = np.abs(pi0*tau + (1-pi0)*tau**a-(pi0/alpha)*tau)

plt.figure(figsize=(2.8,2.8))
qs = [alpha/pi0, alpha, alpha/2, alpha/10]
ls = []
for cc, q_ in enumerate(qs):
    q = q_*pi0
    Tq = ((1/q - 1)*pi0/(1-pi0))**(-1/(1-a))
    regret = ((pi0*tau + (1-pi0)*tau**a-(pi0/alpha)*tau)
              -(pi0*Tq+(1-pi0)*Tq**a-(pi0/alpha)*Tq))
    l, = plt.plot(a, regret/bayes_risk_abs, color=colors[cc], 
                  marker=markers[cc], markevery=markevery, markersize=5)
    ls.append(l)
regret = ((pi0*tau + (1-pi0)*tau**a-(pi0/alpha)*tau)
          -(pi0*tau_1+(1-pi0)*tau_1**a-(pi0/alpha)*tau_1))
l1, = plt.plot(a, regret/bayes_risk_abs, '-*', color=lblue, 
               markevery=markevery, markersize=5)

regret = ((pi0*tau + (1-pi0)*tau**a-(pi0/alpha)*tau)
          -(pi0*tau+(1-pi0)*tau**a-(pi0/alpha)*tau))
l2, = plt.plot(a, regret/bayes_risk_abs, '--', color='k')


plt.ylabel('Normalized population regret')
plt.xlabel('Inverse signal strength $\\theta$,\nwhere $f_1$=Beta$(\\theta,1)$')

plt.gca().spines['top'].set_color('none')
plt.gca().spines['bottom'].set_color('none')
plt.gca().spines['left'].set_position('zero')
plt.gca().spines['bottom'].set_position('zero')
plt.gca().spines['right'].set_color('none')


plt.xlim([0, np.max(a)])
plt.ylim([-.02, 1.02])
plt.tight_layout()
plt.savefig('2b.pdf')

################ Figure 3 - ecdf and its least concave majorant ################
def cusum(x):
    return np.concatenate(([0], np.cumsum(x)))

def unsum(x):
    return x[1:] - x[:-1]

# given the point-set Pi = (Wi, Gi) s.t. W1 ≤...≤ Wn,
# return the sequence G* s.t. (Wi, G*_i) is the GCM of P
def GCM(W, G):
    w = unsum(W)
    g = unsum(G)/w
    g_= isotonic.isotonic_regression(g.copy(), sample_weight=w.copy())
    return cusum(g_*w)+G[0]

def LCM(W, G):
    return -GCM(W, -G)

# setting (E)
m = 64
pi0 = 3/4 
m0 = int(pi0*m)
m1 = m-m0
L = 5 # 5,10
m1 = m-m0
mu = np.zeros(m)
for j, sp in enumerate(np.split(np.arange(m1),4)):
    mu[sp] = (j+1)*L/4
H = 1-(mu==0)

q = .8

np.random.seed(123)
x = np.random.randn(m) + mu
p = stats.norm.sf(x)
p_ = np.sort(p)
p__ = np.hstack([0, p_])
H_= H[np.argsort(p)]
T = np.arange(m+1)/m
loc = np.argmin(p__-q*T)
BH = np.max(np.where(p__ <= q*T)[0])

BH, p__[BH], loc, p__[loc]

fig, ax = plt.subplots(figsize=(2.8,2.8))

xmax = T[sum(p__<=q)]

ax.step(p__, T, 'k.-', where='post')

tt = np.linspace(0, 1, 100000)
ax.plot(tt*q + np.min(p__-q*T), tt, c=lblue, ls='-', alpha=.75)

G = LCM(p__, T)
ax.plot(p__, G, '--', c=lblue)

ax.set_xticks([])
ax.set_yticks([])

ax.text(p__[loc], -.03, '$\\tau_{\\ell}$', c=lblue)
ax.text(.5+.01, -.03, '$t$', c='k')
ax.axvline(p__[loc], c=lblue, ls='-', alpha=.75)

ax.text(.43, .56, '$F_m(t)$', c='k')
ax.text(.35, .62, '$\\hat{F}_m(t)$', c=lblue)

ax.text(.16, .62, 'Slope $\\ell^{-1}$', c=lblue)

style = "Simple, tail_width=0.2, head_width=2, head_length=4"
kw = dict(arrowstyle=style)
a1 = patches.FancyArrowPatch((.2,.61), (.24,.55), color=lblue, 
                             connectionstyle="arc3,rad=.5", **kw)
ax.add_patch(a1)

ax.set_xlim([-.01, .5+.01])
ax.set_ylim([-.0, .65])

ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')

plot_arrows(fig, ax)

plt.tight_layout()
plt.savefig('3a.pdf')

tt = np.linspace(0, 1, 100000)
fms = np.array([np.mean(p_ <= t) for t in tt])

fig, ax = plt.subplots(figsize=(2.8,2.8))

xmax = T[sum(p__<=q)]

ax.plot(tt, fms-tt/q, 'k-')
ax.plot(p_, T[1:]-p_/q, 'k.')

ax.axhline(-np.min(T-p__/q), c=lblue, ls='-', alpha=.75)

G = LCM(p__, T)
ax.plot(p__, G-p__/q, '--', c=lblue)

ax.set_xticks([])
ax.set_yticks([])
ax.text(p__[loc], -.03, '$\\tau_{\\ell}$', c=lblue)
ax.text(.5+.01, -.03, '$t$', c='k')
ax.axvline(p__[loc], c=lblue, ls='-', alpha=.75)

ax.text(.25, .05, '$F_m(t)-t/\ell$', c='k')
ax.text(.35, .15, '$\\hat{F}_m(t)-t/\\ell$', c=lblue)

ax.set_xlim([-.01, .5+.01])
ax.set_ylim([-.0, .65])

ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')

plot_arrows(fig, ax)
plt.tight_layout()
plt.savefig('3b.pdf')

############################### Simulation code ################################
def last_argmin(x, axis=None):
    x = np.flip(x, axis=axis)
    n = x.shape[axis] if axis is not None else len(x)
    return(n-1-np.argmin(x, axis=axis))

def Proc(p_, level, method=None):
    """
    Run the BH procedure or the SL procedure
    
    p_ : ndarray of shape (m+1, S), sorted along axis=0
    level : float or ndarray of shape (,S)
    method : either 'BH' or 'SL'
    
    Returns # of rejections R and the p-value corresponding 
    to the last rejection tau (both ndarrays of shape (S,)). 
    
    Note: If p_ is a MaskedArrray, both methods will only 
    search over unmasked p-values.
    """
    if (len(p_.shape) == 1): p_ = p_[:, np.newaxis]
    m, S = p_.shape
    m -= 1
    T = np.repeat(np.arange(m+1)[:, np.newaxis], S, axis=1)/m
    Z = p_ - level*T
    Z[0] = 0
    if method=='BH':
        R = (Z <= 0).sum(0)-1
    elif method=='SL':
        R = last_argmin(Z, axis=0)
    else:
        raise Exception('method must be BH or SL')
    if S > 1:
        tau = p_[R, np.arange(S)]
    else:
        R = R[0]
        tau = p_[R,0]
    return(R, tau)


def Metrics(p_, level, H_, f, f_, pi0, method=None, is_monotone=True):
    """
    Run and evaluate the BH procedure or the SL procedure
    in simulations (i.e. where latent H's and two-groups 
    model parameters are known)
    
    p_ : ndarray of shape (m+1, S), sorted along axis=0
    level : float or ndarray of shape (,S)
    H_ : ndarray of shape (m+1, S)
    f : function to evaluate marginal density
    pi0 : the null proportion
    method : either 'BH' or 'SL'
    is_monotone : bool, (default True)
    
    Returns a dict containing arrays of shape (S,):
        - R : # of rejections
        - tau : p-value corresponding to the last rejection 
        - FDR : the false discovery proportion 
        - bdry-fdr : indicates whether last rejection is null
        - max-lfdr : maximum lfdr of rejections
        
    Notes: If p_ is a MaskedArrray, both methods will only 
    search over unmasked p-values. If is_monotone is True,
    max-lfdr evaluates lfdr(tau); otherwise it explicitly 
    computes the maximum.
    """
    S = p_.shape[1]
    R, tau = Proc(p_, level, method=method)
    
    mask = (p_ > tau)
    if type(p_) == np.ma.core.MaskedArray:
        mask = mask & p_.mask
    
    V = ((p_ <= tau) * (1-H_)).sum(0)

    FDR = V / np.maximum(1, R)    
    bdry_fdr = 1-H_[R, np.arange(S)]
    
    ## If density is monotone, only need to 
    ## evaluate lfdr at the last rejection
    if is_monotone: 
        max_lfdr = np.where(R > 0, pi0/f(tau), 0)
    else:
        lfdrs = pi0/f(p_)
        lfdrs[p_ > tau[np.newaxis, :]] = 0
        max_lfdr = np.max(lfdrs, axis=0)
    
    return({'R':R, 
            'tau':tau, 
            'FDR':FDR, 
            'bdry-fdr':bdry_fdr, 
            'max-lfdr':max_lfdr})

def Summarize(ls, kw, stat=np.mean):
    """
    Unpack and summarize results in list of dicts
    
    ls : list of dicts
    kw : key to evaluate
    stat : function to apply to each value in list
    
    Returns a list, parallel to ls, taking values
    in the range of stat. 
    """
    return([stat(x[kw]) for x in ls])


def gen_data(setting, m, S, **kwargs):
    S_rep = np.repeat(np.arange(S)[np.newaxis, :], m, axis=0)
    if setting in {'independent Gaussian', 'autoregressive', 'equicorrelated'}:
        pi0 = 3/4 
        L = 5
        mu = np.array([L*i/4 for i in range(5)])
        probs = np.array([pi0] + [(1-pi0)/4]*4)
        mu_rep = np.repeat(mu[:, np.newaxis], S, axis=1)
        density = (lambda t: probs @ 
                   (stats.norm.pdf(stats.norm.isf(t) - mu_rep) 
                    / stats.norm.pdf(stats.norm.isf(t))))
        density_scalar = (lambda t: probs @ 
                   (stats.norm.pdf(stats.norm.isf(t) - mu) 
                    / stats.norm.pdf(stats.norm.isf(t))))
        mu_ = np.random.choice(mu, p=probs, size=(m, S))
        if setting == 'independent Gaussian':
            x = np.random.randn(m, S) + mu_
        elif setting == 'equicorrelated':
            rho = kwargs.get('rho', None)
            if rho is None: raise Exception('rho not defined')
            cov = (1-rho)*np.eye(m) + rho*np.ones((m,m))
            mu_ = np.random.choice(mu, p=probs, size=(m, S))
            x = np.random.multivariate_normal(np.zeros(m), cov, size=S).T + mu_
        elif setting == 'autoregressive':
            rho = kwargs.get('rho', None)
            if rho is None: raise Exception('rho not defined')
            u = np.ones(m)
            u[1:-1] += rho**2
            v = -rho*np.ones(m-1)
            Theta0 = ((1/(1-rho**2))*(np.diag(u) 
                                      + np.diag(v, 1) + np.diag(v, -1)))
            cov = np.linalg.inv(Theta0)

            mu_ = np.random.choice(mu, p=probs, size=(m, S))
            x = np.random.multivariate_normal(np.zeros(m), cov, size=S).T + mu_
        p = stats.norm.sf(x)
        H = 1-(mu_==0)
    elif setting == 'misspecified':
        aa = np.array([[1,.01,2]])
        bb = np.array([[1,2,100]])
        K1 = int(aa.shape[1] - 1)
        pi0 = 3/4
        probs = np.array([pi0] + [(1-pi0)/K1]*K1)
        def g(t, a, b, tol=1e-300):
            ret = np.full(t.shape, float('Inf'))
            ret[t > tol] = stats.beta.pdf(t[t > tol], a, b)
            return(ret)
        density = lambda t: np.average(
            [g(t, aa[0,i], bb[0,i]) for i in range(3)], axis=0, weights=probs)
        density_scalar = lambda t : stats.beta.pdf(t, a=aa, b=bb)@probs
        
        inds = np.random.choice(K1+1, p=probs, size=(m, S))
        a_ = aa[0,inds]
        b_ = bb[0,inds]
        p = np.random.beta(a_, b_)
        H = 1-((a_==1) & (b_==1))
    o = np.argsort(p, axis=0)
    p_= np.vstack([np.zeros(S), p[o, S_rep]])
    H_ = np.vstack([np.ones(S), H[o, S_rep]])
    return(p_, H_, density, density_scalar, pi0)

def sim(setting, ms, S, **kwargs):
    res = {m : defaultdict(list) for m in ms}
    for j, m in enumerate(ms):
        if setting in {'independent Gaussian', 'misspecified'}:
            params = kwargs.get('qs', None)
            p_, H_, density, f, pi0 = gen_data(setting, m, S)
        elif setting in {'autoregressive', 'equicorrelated'}:
            params = kwargs.get('rhos', None)
        for i,param in enumerate(params):
            print(i+1,'out of',len(params),end='\r')
            if setting in {'autoregressive', 'equicorrelated'}:
                q = kwargs.get('q', None)
                p_, H_, density, f, pi0 = gen_data(setting, m, S, rho=param)
            else:
                q = param
            zeta = .5
            pi0_hat = (np.sum(p_>zeta, axis=0)+1)/((1-zeta)*m) 
            p_mask  = np.ma.masked_array(data=p_, mask=(p_>zeta))

            for method in ['BH', 'SL']:
                res[m][method].append(Metrics(p_, q, H_, density, 
                                              f, pi0, method, 
                                              is_monotone=
                                                (setting!='misspecified')))
                res[m][method+'_est'].append(
                    Metrics(p_mask, q/pi0_hat[np.newaxis, :], H_, density, 
                            f, pi0, method,
                            is_monotone=(setting!='misspecified'))
                )
                
        pname = kwargs.get('pname', None)
        
        methods = ['BH_est', 'BH', 'SL_est', 'SL']
        metrics = ['FDR', 'max-lfdr']
    
        ## TWEAK THESE BASED ON SETTING
        if setting in {'independent Gaussian', 'misspecified'}:
            labs = ['BH$(q/\\hat{\\pi}_0^\\lambda)$', 'BH$(q)$', 
                    'SL$(\\ell/\\hat{\\pi}_0^\\lambda)$', 'SL$(\\ell)$']
        else:
            labs = ['BH$(.2/\\hat{\\pi}_0^\\lambda)$', 'BH$(.2)$', 
                    'SL$(.2/\\hat{\\pi}_0^\\lambda)$', 'SL$(.2)$']
        for i,metric in enumerate(metrics):
            plt.figure(figsize=(2.8, 2.8))

            plt.grid(alpha=.5)

            colors = [cred]*2+[lblue]*2
            lss = ['--', '-', '--', '-']
            for num, method in enumerate(methods):
                plt.plot(params, Summarize(res[m][method], metric), 
                         c=colors[num], ls=lss[num])
                
            if i==0: plt.legend(labs, title='Procedure', loc='upper right')
            
            plt.ylabel(metric + '($\\mathcal{R}$)')
            plt.title(metric + ' control')
            
            ### add black lines
            if setting in {'independent Gaussian', 'misspecified'}:
                plt.plot(qs, pi0*qs, 'k-.')
                plt.plot(qs, qs, 'k-.')
            elif setting in {'equicorrelated', 'autoregressive'}:
                plt.axhline(pi0*q, color='k', ls='-.')
                plt.axhline(q, color='k', ls='-.')

                plt.gca().set_yticks(
                    [i/10 for i in range(7) if i!=2] + [pi0*q, q]
                    )
                plt.gca().set_yticklabels(
                    [i/10 for i in range(7) if i!=2] + ['$.2\\pi_0$', '$.2$']
                    )

            plt.ylim([0,.7])
            plt.xlim([params.min(), params.max()])
            plt.xlabel(pname)

            plt.tight_layout()
            plt.savefig(setting + ' ' + metric + 
                        ' control' + ' m = %d' %m + '.pdf')

        if setting in {'independent Gaussian', 'misspecified'}:
            labs = ['BH$(q)$ procedure', 'SL$(\\ell)$ procedure']
        else:
            labs = ['BH$(.2)$ procedure', 'SL$(.2)$ procedure']
        fig, ax = plt.subplots(figsize=(2.8,2.8))
        ax.plot(params, Summarize(res[m]['BH'], 'max-lfdr'), '.-', c=cred)
        ax.plot(params, Summarize(res[m]['SL'], 'max-lfdr'), '.-', c=lblue)
        if j == 0:
            ax.legend(labs, loc='upper left')
        ax.set_ylabel('$\\max_{i\in\\mathcal{R}}\\text{lfdr}(p_i)$')

        box = ax.boxplot(Summarize(res[m]['BH'], 'max-lfdr', lambda x: x), 
                         positions=params, widths=.005,showfliers=False,whis=0)
        # change the color of its elements
        for _, line_list in box.items():
            for line in line_list:
                line.set_color(cred)

        box = ax.boxplot(Summarize(res[m]['SL'], 'max-lfdr', lambda x: x), 
                         positions=params, widths=.005,showfliers=False,whis=0)
        # change the color of its elements
        for _, line_list in box.items():
            for line in line_list:
                line.set_color(lblue)


        ax.set_xlabel(pname)

        ax.set_xticks(params[::2])
        ax.set_xticklabels(np.round(params[::2], 2))
        ax.set_title('IQR of maximum lfdr $(m=%d)$' %m)
        ax.set_xlim([params.min(), params.max()])
        ax.set_ylim([0, .8])
        ax.grid(alpha=.5)

        if setting == 'independent Gaussian':            
            tqs = np.array([bisect(lambda t:f(t) - 1/q, 1e-12, .2) for q in qs])
            f_deriv_tqs = np.array([derivative(f, t, dx=1e-13) for t in tqs])

            err = .35*pi0*qs*(4*qs**2*np.abs(f_deriv_tqs))**(1/3)
            pi0*qs-m**(-1/3)*err, pi0*qs+m**(-1/3)*err
            colors = [cred, lblue]
            ax.plot(qs, pi0*qs-m**(-1/3)*err, 'x', c=lblue)
            ax.plot(qs, pi0*qs+m**(-1/3)*err, 'x', c=lblue)
            
        plt.tight_layout()
        plt.savefig(setting + ' max-lfdr IQR' + ' m = %d' %m + '.pdf')

############## Figures 4 & 5 - illustrate results from section 2 ###############
pi0 = 3/4 
L = 5
f1 = lambda t: (np.mean(
    [stats.norm.pdf(stats.norm.isf(t) - L*i/4) for i in range(1,5)], axis=0) 
    / stats.norm.pdf(stats.norm.isf(t)))
f = lambda t: pi0+(1-pi0)*f1(t)
lfdr = lambda t : pi0/f(t)

F1 = lambda t : np.mean(
    [stats.norm.sf(stats.norm.isf(t) - L*i/4) for i in range(1,5)], axis=0) 
F  = lambda t : pi0*t+(1-pi0)*F1(t)
Fdr = lambda t: pi0*t/F(t)
tt = np.linspace(0, 1, 10000)

plt.figure(figsize=(2.8, 2.8))

plt.gca().axhline(pi0, c='k', ls='--')
plt.gca().plot(tt, f(tt), 'k-')
plt.gca().text(0.1, pi0/2, '$\\pi_0 = 0.75$')
plt.gca().text(0.1, 2*pi0, '$f(t) = \\pi_0 +(1-\\pi_0)f_1(t)$')
plt.gca().set_ylim([0, 3])

plt.gca().fill_between(tt, pi0*np.ones(len(tt)), f(tt), color=cred, alpha=.25)
plt.gca().fill_between(tt, pi0*np.ones(len(tt)), color=lblue, alpha=.25)

plt.gca().set_xlabel('$t$')
plt.gca().set_xlim([0, 1])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout(pad=2.)
plt.savefig('4a.pdf', bbox_inches='tight') 

plt.figure(figsize=(2.8, 2.8))

plt.gca().plot(tt, lfdr(tt), 'k--')
plt.gca().text(0.2, pi0, 'lfdr$(t) = \\frac{\\pi_0}{f(t)}$') 

plt.gca().fill_between(tt, lfdr(tt), color=lblue, alpha=.25)
plt.gca().fill_between(tt, lfdr(tt), np.ones(len(tt)), color=cred, alpha=.25)
plt.gca().set_ylim([0, 1])

plt.gca().set_xlabel('$t$')
plt.gca().set_xlim([0, 1])
plt.gca().spines['right'].set_visible(False)

plt.tight_layout(pad=2.)
plt.savefig('4b.pdf', bbox_inches='tight') 

S = 100000
qs = np.linspace(.01, .3, 10)
ms = [64, 1024]
res = sim('independent Gaussian', ms, S,
          qs=qs, pname='Tuning parameter ($q$ or $\\ell$)')

########################### Figures 6 - asymptotics ############################
pi0 = 3/4 
L = 5
f1 = lambda t: (np.mean(
    [stats.norm.pdf(stats.norm.isf(t) - L*i/4) for i in range(1,5)], axis=0) 
    / stats.norm.pdf(stats.norm.isf(t)))
f = lambda t: pi0+(1-pi0)*f1(t)
lfdr = lambda t : pi0/f(t)

F1 = lambda t : np.mean(
    [stats.norm.sf(stats.norm.isf(t) - L*i/4) for i in range(1,5)], axis=0
    ) 
F  = lambda t : pi0*t+(1-pi0)*F1(t)
Fdr = lambda t: pi0*t/F(t)
tt = np.linspace(0, 1, 10000)

pi0 = 3/4 
lam = 4
alpha = 1/(1+lam)
L = 5
tau_opt = minimize_scalar(lambda t: np.abs(lfdr(t) - alpha), 
                          bounds=(0,1), method='bounded', tol=1e-10).x

plt.figure(figsize=(4,4))
tt = np.linspace(0, 1, 1000)
plt.plot(tt, lfdr(tt), 'k-')
plt.plot(tau_opt, lfdr(tau_opt), 'rx')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('$t$')
plt.ylabel('lfdr$(t)$')

ms = [2**i for i in range(5, 15)]
if (os.path.exists('asymptotics_means.pkl') 
    and os.path.exists('asymptotics_errs.pkl')):
    with open('asymptotics_means.pkl', 'rb') as handle:
        res = pkl.load(handle)
    with open('asymptotics_errs.pkl', 'rb') as handle:
        se = pkl.load(handle)
else:
    res = defaultdict(list)
    se = defaultdict(list)
    for ind, m in enumerate(ms):
        T = np.arange(m+1)/m 

        B = 10000

        mu = np.random.choice(L*np.arange(5)/4, p=[.75]+[1/16]*4, size=(m, B))
        p = stats.norm.sf(np.random.randn(m, B) + mu)
        p_= np.vstack([np.zeros((1, B)), np.sort(p, axis=0)])

        R = np.argmin(p_ - (alpha/pi0)*T[:, np.newaxis], axis=0)
        tau = p_[R, np.arange(B)]

        Loss = np.sum(np.abs(1-(1/alpha)*np.where(p_>0, lfdr(p_), 0)) 
                      * (p_ <= np.maximum(tau, tau_opt)) 
                      * (p_ >= np.minimum(tau, tau_opt)), axis=0)/m
        res['known'].append(np.mean(Loss))
        se['known'].append(np.std(Loss)/np.sqrt(B))

        R = np.argmin(p_ - alpha*T[:, np.newaxis], axis=0)
        tau = p_[R, np.arange(B)]

        Loss = np.sum(np.abs(1-(1/alpha)*np.where(p_>0, lfdr(p_), 0)) 
                      * (p_ <= np.maximum(tau, tau_opt)) 
                      * (p_ >= np.minimum(tau, tau_opt)), axis=0)/m
        res['unknown'].append(np.mean(Loss))
        se['unknown'].append(np.std(Loss)/np.sqrt(B))

        zeta = 1-m**(-.2)
        pi0_hat = (1+np.sum(p_ > zeta, axis=0))/((1-zeta)*m)
        print(ind+5, np.mean(pi0_hat))
        R = np.argmin(p_ - (alpha/pi0_hat[np.newaxis, :])*T[:, np.newaxis], 
                      axis=0)
        tau = p_[R, np.arange(B)]
        Loss = np.sum(np.abs(1-(1/alpha)*np.where(p_>0, lfdr(p_), 0)) 
                      * (p_ <= np.maximum(tau, tau_opt)) 
                      * (p_ >= np.minimum(tau, tau_opt)), axis=0)/m
        res['estimated'].append(np.mean(Loss))
        se['estimated'].append(np.std(Loss)/np.sqrt(B))
        with open('asymptotics_means.pkl', 'wb') as handle:
            pkl.dump(res, handle)
        with open('asymptotics_errs.pkl', 'wb') as handle:
            pkl.dump(se, handle)

tau_subopt = minimize_scalar(lambda t: np.abs(lfdr(t) - pi0*alpha), 
                             bounds=(0,1), method='bounded', tol=1e-10).x

F1 = lambda t: np.mean(
    [stats.norm.sf(stats.norm.isf(t) - 5*i/4) for i in range(1,5)], axis=0
    ) 
F = lambda t: pi0*t+(1-pi0)*F1(t)
limregret = (F(tau_opt)-F(tau_subopt)) - (pi0/alpha)*(tau_opt-tau_subopt)

h = 1e-12
f_opt_deriv = np.abs(f(tau_opt-h) - f(tau_opt))/h

plt.figure(figsize=(3.5,2.6))
plt.plot(np.log10(ms), np.log10(limregret)*np.ones(len(ms)), '-.', c=cred)
plt.plot(np.log10(ms), np.log10(res['unknown']), '.-', c=cred, 
         label='$\\mathcal{R}_\\alpha$ (bounded $\\pi_0\le 1$)')
plt.plot(np.log10(ms), np.log10(res['estimated']), 'k.-', 
         label=('$\\mathcal{R}_{\\alpha/\\hat{\\pi}_0^{\\lambda_m}}$'
                +' (estimated $\\pi_0$)'))
plt.plot(np.log10(ms), np.log10(res['known']), '.-', c=lblue, 
         label='$\\mathcal{R}_{\\alpha/\\pi_0}$ (known $\\pi_0$)')
plt.plot(np.log10(ms), (-(2/3)*np.log10(ms) + np.log10(0.2635596) 
                        + (1/3)*np.log10(2*(pi0/alpha)**2/f_opt_deriv)), 
                        '-.', c=lblue) 
xticks = ms[1::2]
xticklabs = ms[1::2]

plt.xticks(np.log10(xticks), xticklabs)
yticks = np.array([.0001, .0005, .001, .005, .01, .05])
plt.yticks(np.log10(yticks), yticks)
plt.xlabel('Sample size $m$ (log scale)')
plt.ylabel('Regret$_m(\\mathcal{R})$ (log scale)')
plt.title('Asymptotic rate of regret $(\\omega=%d)$'%lam)
plt.legend(fontsize=8)
plt.grid(alpha=.5)
plt.tight_layout()
plt.savefig('6.pdf') 

############################## Figures 7, 8, & 9 ###############################
S = 100000
ms = [64, 1024]

rhos = np.arange(0, 1, .1)
res = sim('equicorrelated', ms, S, 
          rhos=rhos, q=.2, pname='Equicorrelation $\\rho$')

rhos = np.arange(-.9, 1, .1)
res = sim('autoregressive', ms, S, 
          rhos=rhos, q=.2, pname='Autocorrelation $\\rho$')

tt = np.linspace(0, 1, 100000)

aa = np.array([[1,.01,2]])
bb = np.array([[1,2,100]])
K1 = int(aa.shape[1] - 1)
pi0 = 3/4
probs = np.array([pi0] + [(1-pi0)/K1]*K1)
ff = stats.beta.pdf(tt[:, np.newaxis], a=aa, b=bb)@probs
lfdr = pi0/ff
plt.plot(tt, lfdr, 'b-')
plt.xlim([0, .1])

plt.figure(figsize=(2.8, 2.8))

plt.gca().plot(tt, lfdr, 'k--')
plt.gca().text(0.2, pi0, 'lfdr$(t) = \\frac{\\pi_0}{f(t)}$') 

plt.gca().fill_between(tt, lfdr, color=lblue, alpha=.25)
plt.gca().fill_between(tt, lfdr, np.ones(len(tt)), color=cred, alpha=.25)
plt.gca().set_ylim([0, 1])

plt.gca().set_xlabel('$t$')
plt.gca().set_xlim([0, 1])
plt.gca().spines['right'].set_visible(False)

plt.tight_layout(pad=2.)
plt.savefig('8a.pdf', bbox_inches='tight') 

tt = np.linspace(0, .1, 100000)
ff = stats.beta.pdf(tt[:, np.newaxis], a=aa, b=bb)@probs
lfdr = pi0/ff

plt.figure(figsize=(2.8, 2.8))

plt.gca().plot(tt, lfdr, 'k--')
plt.gca().text(0.01, pi0, 'lfdr$(t) = \\frac{\\pi_0}{f(t)}$') 
plt.gca().text(0.0315, .9, '(zoomed in)') 

plt.gca().fill_between(tt, lfdr, color=lblue, alpha=.25)
plt.gca().fill_between(tt, lfdr, np.ones(len(tt)), color=cred, alpha=.25)
plt.gca().set_ylim([0, 1])

plt.gca().set_xlabel('$t$')
plt.gca().set_xlim([0, .05])
plt.gca().spines['right'].set_visible(False)

plt.tight_layout(pad=2.)
plt.savefig('8b.pdf', bbox_inches='tight') 

## very slow
qs = np.linspace(.01, .3, 10)
res = sim('misspecified', ms, S, qs=qs, 
          pname='Tuning parameter ($q$ or $\\ell$)')