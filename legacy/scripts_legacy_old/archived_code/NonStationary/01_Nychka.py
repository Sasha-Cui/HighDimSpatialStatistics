# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/archived_code/NonStationary/01_Nychka.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# This implements Kleiber and Nychka JMVA nonstationary model.

# %%
%run -i preambles
%run -i helper_functions

# %%
# The first step is to write down a function that computes gamma(i,j,x,y) based on Z(i,k).  This involves also defining K(h,k,l) and therefore Chat(i,j,k,l), which is a four-dimensional matrix. 
#
# The second step is to write down Chat_e(i,j,k,l), which is a four-dimensional matrix.
#
# The third step is to finally define the loss function.  This involves also writing down the matrix W(h, s). 
#
# Since everything we are talking about here are just matrices whose entries are based on the specific locations involved, we are not really working with some estimation of funcitons.  It is all in the finite world. 

# %%
# Suppose that we have a data set.  We load it so that we have $Z_1(s),\dots,Z_p(s)$ for $s\in \{x_1,\dots,x_n\}$.  Let us produce it now. 

# %%
# for _ in range(1,301):
_ = 1
print(f"data set {_}")
# Load the CSV file
df = pd.read_csv(f'~/project/synthetic_data/realisation_{_}.csv')
# Extract the spatial coordinates (first and second spatial coordinates)
X = torch.tensor(df[['0','1']].values[:200], dtype=torch.float64).detach()
# Prepare Y by reshaping the gene expression levels for each gene
expression = df['Expression'].values.reshape(200, 3)
Y = torch.tensor(expression, dtype=torch.float64).detach()
torch.autograd.set_detect_anomaly(True)

# %%
# This data set has $n=200$ locations in $d=2$ dimensions with $p=3$ features.  We are going to compute the corresponding $\hat C(i,j,x,y)$ from it.

# %%
def exp_kernel(X,lmbda, k,l):
    h = torch.norm(X[k]-X[l])
    return np.exp(-h/lmbda)
# Sanity Check
# print(X[3])
# print(X[2])
# print(exp_kernel(1,3,2))
# np.exp(-np.sqrt((0.1701-2.558)**2 + (0.5312-0.1608)**2)/1) 

# %%
# expression (7) in the paper.  Here $x=x_k, y=x_l, and k is represented by m$
def Chat(X, Y, kernel, lmbda, i , j, k, l):
    n = X.shape[0]  # Assuming X is a 2D tensor with shape (n, m)
    
    # Store the intermediate kernel results
    sum_k = torch.tensor(0.0, dtype=torch.float64)
    sum_l = torch.tensor(0.0, dtype=torch.float64)
    
    # Numerator: \sum_{m=1}^n \sqrt{exp_kernel(lmbda, m,k)} \sqrt {exp_kernel(lmbda, m,l} X[m][i] X[m][j]
    numerator = 0
    for b in range(n):
        sqrt_term_l = kernel(X,lmbda, b, l)
        sqrt_term_k = kernel(X,lmbda, b, k)
        numerator += torch.sqrt(sqrt_term_k) * torch.sqrt(sqrt_term_l) * Y[b,i] * Y[b,j]
        sum_k += sqrt_term_k
        sum_l += sqrt_term_l
    
    # Denominator: \sqrt{\sum_{m=1}^n exp_kernel(lmbda, m,k)} \sqrt{\sum_{m=1}^n exp_kernel(lmbda, m,l}}
    denominator = torch.sqrt(sum_k) * torch.sqrt(sum_l)
    
    # Final result
    result = numerator / denominator
    return result
    
Chat(X, Y, exp_kernel,0.5,1,2,1,2)

# %%
# \hat gamma
def gammahat(X,Y, kernel, lmbda,i,j,k,l):
    numerator = Chat(X,Y, kernel, lmbda, i,j,k,l)
    denominator = torch.sqrt(Chat(X,Y, kernel, lmbda, i,i,k,k) * Chat(X,Y, kernel, lmbda, j,j,l,l))
    result = numerator / denominator
    return result
gammahat(X,Y,exp_kernel,0.5,1,2,1,2)

# %%
# expression (8)
def Chat_e(X, Y, kernel, lmbda, i , j, k, l):
    n = X.shape[0]
    numerator = torch.tensor(0.0, dtype=torch.float64)
    denominator = torch.tensor(0.0, dtype=torch.float64)
    for a in range(n):
        for b in range(n):
            weight_term = kernel(X,lmbda, k,a) * kernel(X,lmbda,l,b)
            numerator += weight_term * Y[a,i] * Y[b,j]
            denominator += weight_term 
    result = numerator/denominator
    return result


Chat_e(X, Y, exp_kernel, 0.5,1,2,1,2)

# %%
# As the location $s$ is fixed, so we can treat these parameters to be estimated as just vectors and matrices.
#
# $\sigma= \sigma_1,\dots, \sigma_p$ is a componentwise positive vector.
#
# $\nu = \nu_1,\dots, \nu_p$ is a componentwise positive vector.
#
# $\Sigma_1,\dots, \Sigma_p$ is a length $p$ tensor, where each component is a positive definite $d$ by $d$ matrix.
#
# The other terms involved in the expression are 
#
# $\Sigma_{ij}= \frac12\Sigma_i+ \frac12\Sigma_j$
#
# $\nu_{ij}= \frac12\nu_i+ \frac12\nu_j$
#

# %%
def nonstationary_matern_kernel(nu,x):
    return torch.pow(x,nu) * BesselKFunction.apply(nu, x)

# %%
def C_M(X,Y,kernel, lmbda, i,j,k,l, sigma, Sigma, nu):
    numerator = sigma[i]*sigma[j]* nonstationary_matern_kernel(nu[i]/2+nu[j]/2, torch.sqrt( (X[k]-X[l]).T Sigma[i] ))
    denominator = torch.sqrt(torch.det(Sigma[i]/2 + Sigma[j]/2))
    result = numerator / denominator
    if i!=j:
        result *= gammahat(X,Y, kernel, lmbda,i,j,k,l)*torch.sqrt(torch.gamma(nu[i,k]) * torch.gamma(nu[j,l]))/torch.gamma(nu[i,k]/2+nu[j,k]/2) # typo in paper?
    return result

# %%
def loss_function(sigma, SIGMA, nu):
    
    

