
# __Aiyagari - Uninsured Idiosyncratic Risk and Aggregate Saving - The Quarterly Journal of Economic - (1994)__


# Replication made by Alejandro Rojas-Bernal. Commentaries or suggestions to `alejandro.rojas@alumni.ubc.ca`


## __1. The model__


### __1.1 Consumer Problem__ 


The individual problem is to maximize
$$E_0\left\{\sum_{t=0}^{\infty}\beta^t U(C_t)\right\}$$
subject to 
$$c_t + a_{t+1} = w y_t + (1 + r) a_t$$
$$c_t \ge 0$$
$$ a_t > - b $$
where $b$ (if positive) is the limit on borrowing and $y_t$ is assumed to be iid with bounded support given by $\left[y_{min}, y_{max}\right]$ with $y_{min} > 0$.


If $r<0$ we need to impose a borrowing constraint in order to have a solution. Otherwise, if $r>\ge 0$ a less restrictive constraint is to impose present value budget valance, i.e. $lim_{t \rightarrow \infty} \frac{a_t}{\left(1+r\right)^t} \ge 0 \, (a.s.)$. The nonnegativity of consumption imposes the natural borrowing constraint $a_t \ge -\frac{w y_{min}}{r}$. If $b>\frac{w y_{min}}{r}$ the borrowing limit $b$ will neve be binding. Therefore we can replace the borrowing constraint for:
$$a_t \ge -\varphi$$
$$\varphi \equiv min\left\{b, \frac{w y_{min}}{r}\right\}, \text{  for  }r>0; \text{  }\varphi\equiv b, \text{  for  } r\leq 0.$$


Let $V\left(a_t,y_t\vert b,w,r\right)$ be the optimal value function for the agent with savings $a_t$ and productivity endowment $y_t$. The function is the unique solution to the Bellman equation:
$$V\left(a_t,y_t\vert b,w,r\right) \equiv Max\left\{U\left(w y_t + (1 + r) a_t - a_{t+1}\right)+\beta \int  V\left(a_{t+1},y_{t+1}\vert b,w,r\right)dF\left(y_{t+1}\right)\right\}$$


In the steady state there is going to be a distribution of asset holdings given by $G\left(a\right)$ such that $E[a]=\int a dG(a)$.


### __1.2 Producer Problem__


Let $F\left(k,1\right)$ denote per capita output and $\delta$ be the depreciation rate. The producer profit maximization under steady state is given by $r = f_k\left(k,1\right)-\delta$ and $w(r)=f_L\left(k,1\right)$. $r = f_k\left(k,1\right)-\delta$ is an implicit function for the steady state demand of capital $K(r)$.


### __1.3 Stationary Equilibrium__


Under the stationary distribution we have that the equilibrium is given by $K(r)=E[a]$.


## __2. Model Specification and Parameterization.__


Aiyagari uses $U(c)=\frac{c^{1-\sigma}}{1-\sigma}$ with $\sigma \in \left\{1, 3, 5\right\}$. The borrowing constraint $b=0$. The labor endowment shock is given by a Markov chain specification with seven states that matches the following AR(1) representation for the logarithm of the labor endowment shock:
$$log\left(y_t\right)=\lambda \log\left(y_{t-1}\right) + \sigma_{\epsilon}\left(1-\lambda^2\right)^{1/2} \epsilon_t, \text{  where   }\epsilon_t \sim Normal(0,1)$$
$\sigma_{\epsilon} \in \left\{0.2, 0.4\right\}$ and $\rho \in \left\{0, 0.3, 0.6, 0.9\right\}$. The Markov Chain is defined using Tauchen(1986). 


For the producer $F\left(K,L\right)=K^\alpha L^{1-\alpha}$ with $\alpha = 0.36$ and $\delta = 8\%$.

~~~~{.julia}
u(c, σ) = σ == 1 ? log(c) : (c^(1 - σ)-1) / (1 - σ); # utility function
du(c, σ) = σ == 1 ? 1/c : (1/(c^σ)); # der. of utility finction
F(K, L, α) = K^(α) * L^(1-α); # Production function
fK(K, L , α) = α * K^(α-1) * L^(1-α); # MPK
fL(K, L , α) = (1-α) * K^α * L^(-α); # MPL
Kr(r, δ, α) = ((r + δ) / α)^(1 / (α-1)); # Firms K optimal demand
β = 0.96; # discount factor
σ = 1; # inverse of elasticity of substitution
cc = 0.0; # borrowing constraint
λ = 0.9; # AR(1) coefficient in productivity shock process for Tauchen
σϵ = 0.2; # variance error in productivity shock process for Tauchen
N = 7; # Size of Tauchen grid
a_max = 25.0; # Max level of capital
a_grid_size = 250; # Grid size
α = 0.36; # Share of capital in Cobb-Douglas
δ = 0.08; # Depreciation
pop = 10000; #Size of population for distribution convergence
θ = [β, σ, cc, λ, σϵ, N, a_max, a_grid_size, α, δ]; #θ compiles parameters
~~~~~~~~~~~~~




## __3. Model Computation__


### __3.1 Environment Set-up__

~~~~{.julia}
using LinearAlgebra, Statistics, Distributions, Expectations, NLsolve, Roots, Random, Plots, Parameters, BenchmarkTools, ProgressMeter, LaTeXStrings, Profile, BenchmarkTools, Roots, NLsolve, ForwardDiff, KernelDensity, AverageShiftedHistograms
~~~~~~~~~~~~~




### __3.2 Tauchen Method function__


Given a $\lambda$, $\sigma_\epsilon$ and a number of markov states the following function produces a vector $Y$ with the log productivity endowment and a matrix $M$ with the markov process.


__Steps__

1. Define $\sigma_y = \left(\frac{\sigma_\epsilon^2}{1-\lambda^2}\right)$
2. Establish the max and the min values $y_N = m \times \sigma_y$ and $y_1 = -Y_N$ (following Tauchen(1986) $m=3$).
3. Define $P\left(y_1 \lvert y_j\right) = \Phi\left(\frac{y_1 -\lambda y_j + \frac{w}{2}}{\sigma_\epsilon}\right)$, $P\left(y_N \lvert y_j\right) = 1-\Phi\left(\frac{y_N -\lambda y_j + \frac{w}{2}}{\sigma_\epsilon}\right)$, and $P\left(y_k \lvert y_j\right) = \Phi\left(\frac{y_k -\lambda y_j + \frac{w}{2}}{\sigma_\epsilon}\right) -  \Phi\left(\frac{y_k -\lambda y_j - \frac{w}{2}}{\sigma_\epsilon}\right)$.

~~~~{.julia}
function TauchenAR1(λ, σϵ, N; m = 3)
    # Unconditional Standard Distribution of y
    σy = sqrt(σϵ^2 / (1 - λ^2))
    # We define the max and the min values
    yN = m * σy
    y1 = -yN
    Y = collect(range(y1, yN, length=N))
    # w is the size of grid jump
    w = Y[2] - Y[1]
    # Markov Matrix
    M = fill(0.0, N, N)
    d = Normal(0,1)
    for j ∈ 1:N
        M[j,1] = cdf(d , (Y[1] - λ*Y[j] + w/2) / σϵ)
        M[j,N] = 1 - cdf(d , (Y[N] - λ*Y[j] - w/2) / σϵ)
    end
    for k ∈ 2:N-1
        for j ∈ 1:N
            M[j,k] = cdf(d, (Y[k] - λ*Y[j] + w/2)/ σϵ) - cdf(d, (Y[k] - λ*Y[j] - w/2)/ σϵ)
        end
    end
    for j ∈ 1:N
        M[j,:] = M[j,:] * (1/sum(M[j,:]))
    end
    @assert sum(M,dims=2) ≈ ones(N,1) "One of the conditional trayectories doesn't add to 1"
    return Y, M
end
~~~~~~~~~~~~~


~~~~
TauchenAR1 (generic function with 1 method)
~~~~




For example with $\lambda = 0.5$, $\sigma_\epsilon=1$, and $N=5$ we obtain:


The log productivity process:

~~~~{.julia}
@show TauchenAR1(0.5, 1, 5)[1]
~~~~~~~~~~~~~


~~~~
(TauchenAR1(0.5, 1, 5))[1] = [-3.4641016151377544, -1.7320508075688772, 0.0
, 1.7320508075688772, 3.4641016151377544]
5-element Array{Float64,1}:
 -3.4641016151377544
 -1.7320508075688772
  0.0               
  1.7320508075688772
  3.4641016151377544
~~~~




The Markov process:

~~~~{.julia}
@show TauchenAR1(0.5, 1, 5)[2]
~~~~~~~~~~~~~


~~~~
(TauchenAR1(0.5, 1, 5))[2] = [0.1932381153856163 0.6135237692287674 0.18855
073115589882 0.00467993306182124 7.451167896244115e-6; 0.04163225833177518 
0.4583677416682248 0.4583677416682248 0.04136625557920559 0.000266002752569
60515; 0.004687384229717438 0.18855073115589888 0.6135237692287674 0.188550
73115589882 0.004687384229717484; 0.0002660027525696245 0.04136625557920555
6 0.4583677416682248 0.4583677416682248 0.041632258331775196; 7.45116789621
3828e-6 0.004679933061821224 0.18855073115589888 0.6135237692287674 0.19323
81153856163]
5×5 Array{Float64,2}:
 0.193238     0.613524    0.188551  0.00467993  7.45117e-6 
 0.0416323    0.458368    0.458368  0.0413663   0.000266003
 0.00468738   0.188551    0.613524  0.188551    0.00468738 
 0.000266003  0.0413663   0.458368  0.458368    0.0416323  
 7.45117e-6   0.00467993  0.188551  0.613524    0.193238
~~~~




### __3.3 Policy Function__


We are going to define the function `KpolicyAiyagari(u,du,k,θ)` with inputs:
- u: utility function
- du: derivative of utility function
- k: level of capital, this can be given by the interest rate with the function Kr(r, δ, α) previously defined
- θ: vector of parameters previously define


__Steps for Value Function Iteration__

1. Extract the Tauchen Markov discretization with values $l \in \left\{exp(Y_n)\right\}_{n=1}^{N}$ and markov transition matrix $W$.
2. We calculate $r = f_k\left(k,1,\alpha\right)-\delta$ and  $w = f_L\left(k,1,\alpha\right)$.
3. We use Maliar, Maliar, Valli (2009) algorithm for building a grid for $a_t$ with more intervals in the lower values of $a_t$ with the equation $A = \left\{a_j =\left(\frac{j}{J}\right)^\rho a_{max}\right\}$ where $J$ stands for grid size and we follow them in using $\rho = 7$.
4. We iterate the value function:
$$V^{N+1}\left(a_j,y_i\vert b,w,r\right) \equiv Max_{a_{t+1} \in A}\left\{U\left(w y_i + (1 + r) a_{j} - a_{t+1}\right)+\beta \sum_{n=1}^{N}  V^{N}\left(a_{t+1},y_{n}\vert b,w,r\right)Prob\left\{y_n\lvert y_i\right\}\right\}$$
5. We iterate step 4 until the value function until $sup\lvert V^{N+1}-V^N \lvert<1E^{-10}$



*The algorithm is inefficient in the sense that there are loops that can be eliminated but are keep for the sole purpose of clarity*

*There are several controls inside the function: i) positive consumption is possible under worst possible state; ii) $r < (1-\beta)/\beta$; and iii) monotonic policy function *

~~~~{.julia}
function KpolicyAiyagari(u, du, k, θ; tol = 1E-10, max_iter = 10000, adj = 7, min_iter = 10)
    start = time()
    println("   Kpolicy(Aiyagari)...")
    # Extract and define the internal parameters
    β, σ, cc, λ, σϵ, N, a_max, a_grid_size, α, δ = θ[1], θ[2], θ[3], θ[4], θ[5], convert(Int,θ[6]), θ[7], convert(Int,θ[8]), θ[9], θ[10];
    b = -cc;
    # Define the Tauchen discretization and the Markov process
    lnY , W = TauchenAR1(λ, σϵ, N);
    Y = exp.(lnY);
    @assert size(Y)[1] == size(W)[1] == size(W)[2] == N;
    # Define the relevant vector of prices
    r = fK(k, 1, α) - δ;
    @assert r < (1-β)/β " r < (1-β)/β not satisfied"
    w = fL(k, 1, α);
    # Borrowing constraint as defined in Aiyagari (p. 666)
    φ = r > 0 ? min(b, w*Y[1] / r) : b;
    # Grid for with logarithmic adjustment - more weight to lower values of a by decreasing adj
    a_grid = cc .+ [(p/a_grid_size)^adj * (a_max-cc) for p in 1:a_grid_size]#exp.(collect(range(log(adj), log(a_max + φ + adj), length = a_grid_size))) .- (adj + φ);
    @assert w*Y[1] + r * (-φ) > 0 "Negative Consumption under worst possible scenario";
    # We define the initial matrices for the iteration
    sup = Inf;
    Ind1 = fill(convert(Int,floor(a_grid_size/2)), N, a_grid_size);
    V1 = zeros(size(Ind1));
    A_prime1 = a_grid[Ind1];
    iter = 0;
    while sup > tol && iter <= max_iter
        V0 = copy(V1);
        Ind0 = copy(Ind1);
        A_prime0 = copy(A_prime1);
        for j ∈ 1:a_grid_size
            for i ∈ 1:N
                a , b = findmax(u.(max.(w * Y[i] .+ (1+r) * a_grid[j] .- a_grid, 1E-40), σ) .+ β * (1 + r) * V0' * W[ i, :], dims = 1);
                V1[i, j] = a[1];
                Ind1[i, j] = b[1];
                A_prime1[i,j] = a_grid[Ind1[i, j]];
            end
        end
        sup = maximum(abs.(V0 - V1));
        iter += 1;
    end
    A_prime = A_prime1;
    @assert findmax(A_prime, dims = 2)[1] ≈ A_prime[:,end] "Final Column of A' does not coincide"
    Ind = Ind1;
    V = V1;
    elapsed = time() - start
    println("      Kpolicy Aiyagari solved in $iter iterations, with a sup metric of $sup and in $elapsed seconds")
    return sol = [A_prime, a_grid, Ind, V, Y, W]
end
~~~~~~~~~~~~~


~~~~
KpolicyAiyagari (generic function with 1 method)
~~~~



~~~~{.julia}
VFI = KpolicyAiyagari(u, du, Kr(0.015, δ, α), θ);
~~~~~~~~~~~~~


~~~~
Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 849 iterations, with a sup metric of 9.912
071163853398e-11 and in 23.9520001411438 seconds
~~~~




#### __3.3.1 Policy Function__


In general, as the productivity shock is bigger households are going to save more

~~~~{.julia}
plot(VFI[2],[VFI[1][s,:] for s in 1:N], title ="Policy Function",xlabel="Capital at t", ylabel="Capital at t+1")
~~~~~~~~~~~~~




#### __3.3.2 Value Function__


Concavity of value function and monotonicity with respect to the productivity shock

~~~~{.julia}
plot(VFI[2],[VFI[3][s,:] for s in 1:N], title ="Value Function",xlabel="Capital at t", ylabel="Value Function")
~~~~~~~~~~~~~




#### __3.3.3 Non-Uniform grid__

~~~~{.julia}
plot(collect(1:1:size(VFI[2])[1]),VFI[2], title ="Capital Grid",xlabel="Position", ylabel="Value on Grid")
~~~~~~~~~~~~~




### __3.4 Convergence to Stationary Distribution__


Using the results from `KpolicyAiyagari` we follow the following algorithm:

1. Randomly assign an initial combination of state variables $\left\{y_i,a_i\right\}_{i=1}^{pop}$ to each of the pop households. In our simulations we are going to use $pop = 10000$. From this initial assignment we define an initial probability mass function for the state combination $f_0\left(y_i,a_i\right)$.
2. Now we iterate with idiosyncratic shocks at least 100 times until $sup\lvert E_{n+1}\left[a\right]-E_{n}\left[a\right], Var_{n+1}\left[a\right]-Var_{n}\left[a\right], Skw_{n+1}\left[a\right]-Skw_{n}\left[a\right] \lvert< 1E^{-6}$.

~~~~{.julia}
function KlevelAiyagari(u, du, k, θ, pop; tol = 1E-6, max_iter = 10000, min_iter = 100)
    start = time()
    println("   Klevel(Aiyagari)...")
    # Extract and define the internal parameters
    β, σ, cc, λ, σϵ, N, a_max, a_grid_size, α, δ = θ[1], θ[2], θ[3], θ[4], θ[5], convert(Int,θ[6]), θ[7], convert(Int,θ[8]), θ[9], θ[10];
    b = -cc;
    # Define the relevant vector of prices
    r = fK(k, 1, α) - δ;
    @assert r < (1-β)/β " r < (1-β)/β not satisfied"
    w = fL(k, 1, α);
    # Extract elements from the KpolicyAiyagari algorithm
    KpolicyAiyagari_sol = KpolicyAiyagari(u, du, k, θ)
    A_prime = KpolicyAiyagari_sol[1];
    a_grid = KpolicyAiyagari_sol[2];
    Ind = KpolicyAiyagari_sol[3];
    V = KpolicyAiyagari_sol[4];
    Y = KpolicyAiyagari_sol[5];
    W = KpolicyAiyagari_sol[6];
    @assert size(Y)[1] == size(W)[1] == size(W)[2] == N;
    # Borrowing constraint as defined in Aiyagari (p. 666)
    φ = r > 0 ? min(b, w*Y[1] / r) : b;
    # Verify if crossing of the policy function with high productivity and the 45 degree line
    @assert minimum(A_prime[N,:] - a_grid) ≤ 0 "No crossing with 45 degree Line"
    # We build the elements conditional in the max level of savings at the steady state. The one that just crosses the 45 degree line
    lim_ind =  a_grid_size;
    a_sup = a_grid[lim_ind];
    new_a_grid = a_grid[1:lim_ind];
    dim_a = size(new_a_grid)[1];
    new_A_prime = A_prime[:,1:lim_ind];
    new_V = V[:, 1:lim_ind];
    new_Ind = Ind[:,1:lim_ind];
    # Initial Condition
    inda0 = fill(convert(Int,max(floor(dim_a/2),1)), pop, 1);
    A0 = new_a_grid[inda0];
    indy0 = rand(Categorical(W[1,:]), pop);
    Y0 = Y[indy0]
    f0 = zeros(size(new_A_prime));
    for j ∈ 1:dim_a
        for s ∈ 1:N
            f0[s, j] = count((Y0 .== Y[s]) .* (A0 .== new_a_grid[j])) * (1/pop)
        end
    end
    @assert sum(f0) ≈ 1 "Initial distribution doesn't add to 1"
    sup = Inf;
    iter = 0;
    f1 = f0;
    A1 = A0;
    Y1 = Y0;
    μ1 = Inf;
    σ1 = Inf;
    sk1 =Inf;
    # Loop for convergence in a finite set of moments
    while (sup > tol && iter <= max_iter)
        A0 = copy(A1);
        A1 = zeros(size(A0));
        Y0 = copy(Y1);
        Y1 = zeros(size(Y0));
        f0 = copy(f1);
        f1 = zeros(size(f0));
        μ0 = copy(μ1);
        σ0 = copy(σ1);
        sk0 = copy(sk1);
        for s ∈ 1:N
            draw = rand(Categorical(W[s,:]), pop)
            for p ∈ 1:N
                Y1 += (draw .== p) .* (Y0 .== Y[s]) * Y[p];
            end
        end
        for j ∈ 1:dim_a
            for s ∈ 1:N
                A1 += (Y0 .== Y[s]) .* (A0 .== new_a_grid[j]) * new_A_prime[s,j];
            end
        end
        for j ∈ 1:dim_a
            for s ∈ 1:N
                f1[s, j] = count((Y0 .== Y[s]) .* (A0 .== new_a_grid[j])) * (1/pop);
            end
        end
        μ1 = mean(A1);
        σ1 = var(A1);
        sk1 = skewness(A1);
        sup = iter>min_iter ? maximum([μ1 - μ0; σ1 - σ0; sk1 - sk0].^2) : Inf;
        f1 = f1 ./ sum(f1);
        iter += 1;
    end
    f = f1;
    Afin = A1;
    KS = μ1;
    Yfin = Y1;
    elapsed = time() - start
    println("      Klevel Aiyagari solved in $iter iterations, with a sup metric of $sup and in $elapsed seconds")
    return sol = [KS, r, f, a_sup, new_a_grid, new_A_prime, new_V, Y, W, Afin, Yfin]
end
~~~~~~~~~~~~~


~~~~
KlevelAiyagari (generic function with 1 method)
~~~~



~~~~{.julia}
KS = KlevelAiyagari(u, du, Kr(0.015, δ, α), θ, pop);
~~~~~~~~~~~~~


~~~~
Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 849 iterations, with a sup metric of 9.912
071163853398e-11 and in 20.193999767303467 seconds
      Klevel Aiyagari solved in 395 iterations, with a sup metric of 3.7591
333829966175e-7 and in 114.93299984931946 seconds
~~~~




#### __3.4.1 Unconditional CDF for asset holdings__


The unconditional distribution for asset holdings is:

~~~~{.julia}
plot(KS[5], [sum(KS[3][:,1:s]) for s in 1: size(KS[5])[1]])
~~~~~~~~~~~~~



~~~~{.julia}
a = sum(KS[3][:,1]) * 100
println("The probability mass function show us that $a% of the households are hand-to-mouth consumers at the borrowing constraint")
~~~~~~~~~~~~~


~~~~
The probability mass function show us that 9.040000000000001% of the househ
olds are hand-to-mouth consumers at the borrowing constraint
~~~~




#### __3.4.2 Unconditional PDF for asset holdings__


The density function for asset holdings is given by

~~~~{.julia}
plot(ash(vec(KS[10]); m =20); hist = false)
~~~~~~~~~~~~~




#### __3.4.3 Conditional CDF for asset holdings__


The conditional distribution for asset holdings given idiosyncratic productivity is:

~~~~{.julia}
plot(KS[5],[sum(KS[3][p,1:s])/sum(KS[3][p,:]) for s in 1:size(KS[5])[1], p in 1:size(KS[8])[1]], legend=:bottomright)
~~~~~~~~~~~~~




We can see that as the idiosyncratic productivity shock increases the probability of being a hand-to-mouth consumer diminishes.  


#### __3.4.4 Conditional PDF for asset holdings__


The conditional PDF for asset holdings are

~~~~{.julia}
plot(ash(vec(KS[10][KS[11] .== KS[8][1]]); m=30),  label = "y1" ; hist = false)
plot!(ash(vec(KS[10][KS[11] .== KS[8][3]]); m=30),  label = "y3"  ; hist = false)
plot!(ash(vec(KS[10][KS[11] .== KS[8][5]]); m=30),  label = "y5"  ; hist = false)
plot!(ash(vec(KS[10][KS[11] .== KS[8][7]]); m=30),  label = "y7"  ; hist = false)
~~~~~~~~~~~~~




Again, as productivity increases the probability of being a hand-to-mouth consumer decreases


#### __3.4.5 Supply and Demand in Bewley models__


Let's look at the behaviour of the demand and supply function under $\sigma =5$, $\lambda = 0.6$, and $\sigma_\epsilon=0.2$

~~~~{.julia}
θ[5] = 0.2
θ[2] = 5.0
θ[4] = 0.6

r_range = collect(range(-0.0, 0.025, length=30));
Supply1 = fill(0.0, size(r_range)[1], 1);
Demand1 = fill(0.0, size(r_range)[1], 1);

for r ∈ 1 : size(r_range)[1]
    Supply1[r] = KlevelAiyagari(u, du, Kr(r_range[r], θ[10], θ[9]), θ, pop)[1];
    Demand1[r] = Kr(r_range[r], θ[10], θ[9]);
    @show r, r_range[r], Supply1[r], Demand1[r]
end
~~~~~~~~~~~~~


~~~~
Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 525 iterations, with a sup metric of 9.838
174719334347e-11 and in 37.86800003051758 seconds
      Klevel Aiyagari solved in 1657 iterations, with a sup metric of 7.585
585702919834e-7 and in 430.3589999675751 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (1, 0.0, 2.1751560159086343, 10.4
8683784717663)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 536 iterations, with a sup metric of 9.906
653275493227e-11 and in 38.40299987792969 seconds
      Klevel Aiyagari solved in 350 iterations, with a sup metric of 2.9732
806774960087e-7 and in 120.94600009918213 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (2, 0.0008620689655172414, 2.3043
671925612546, 10.312674983554945)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 548 iterations, with a sup metric of 9.773
781783906088e-11 and in 39.90499997138977 seconds
      Klevel Aiyagari solved in 231 iterations, with a sup metric of 5.0362
69207857301e-7 and in 91.91100001335144 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (3, 0.0017241379310344827, 2.3536
63649572684, 10.143205831695491)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 560 iterations, with a sup metric of 9.840
039894015717e-11 and in 39.644999980926514 seconds
      Klevel Aiyagari solved in 1156 iterations, with a sup metric of 1.825
2306166332776e-7 and in 323.228000164032 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (4, 0.002586206896551724, 2.40186
2678337776, 9.978256362290475)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 573 iterations, with a sup metric of 9.738
609918485963e-11 and in 40.45900011062622 seconds
      Klevel Aiyagari solved in 215 iterations, with a sup metric of 4.6345
817614510246e-7 and in 95.25399994850159 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (5, 0.0034482758620689655, 2.5007
16316855023, 9.817660724606434)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 586 iterations, with a sup metric of 9.852
119120523639e-11 and in 44.32100009918213 seconds
      Klevel Aiyagari solved in 439 iterations, with a sup metric of 8.6994
33105473399e-7 and in 141.89000010490417 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (6, 0.004310344827586207, 2.46573
1118360383, 9.661260782666488)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 600 iterations, with a sup metric of 9.831
602199028566e-11 and in 41.40299987792969 seconds
      Klevel Aiyagari solved in 217 iterations, with a sup metric of 2.3650
81291100474e-7 and in 88.44999980926514 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (7, 0.005172413793103448, 2.58213
12793096047, 9.508905682151791)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 615 iterations, with a sup metric of 9.701
661696226438e-11 and in 42.2960000038147 seconds
      Klevel Aiyagari solved in 796 iterations, with a sup metric of 6.1233
01926168324e-7 and in 217.04399991035461 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (8, 0.00603448275862069, 2.593035
8273343437, 9.360451445700878)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 630 iterations, with a sup metric of 9.819
967061730495e-11 and in 42.89100003242493 seconds
      Klevel Aiyagari solved in 139 iterations, with a sup metric of 2.9571
581988646284e-7 and in 72.07500004768372 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (9, 0.006896551724137931, 2.75313
9780368347, 9.215760594482523)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 646 iterations, with a sup metric of 9.862
866079402011e-11 and in 43.60700011253357 seconds
      Klevel Aiyagari solved in 257 iterations, with a sup metric of 3.3690
08014359332e-7 and in 101.64100003242493 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (10, 0.007758620689655172, 2.7958
85463465428, 9.074701794095665)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 663 iterations, with a sup metric of 9.854
69483794077e-11 and in 45.20700001716614 seconds
      Klevel Aiyagari solved in 487 iterations, with a sup metric of 6.9352
16234313079e-7 and in 150.4760000705719 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (11, 0.008620689655172414, 2.8585
393500502314, 8.937149523011778)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 681 iterations, with a sup metric of 9.820
677604466255e-11 and in 46.30000019073486 seconds
      Klevel Aiyagari solved in 1951 iterations, with a sup metric of 5.772
202764113532e-7 and in 481.2369999885559 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (12, 0.009482758620689655, 2.9810
12835920873, 8.80298376192179)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 700 iterations, with a sup metric of 9.786
21628178189e-11 and in 48.375 seconds
      Klevel Aiyagari solved in 227 iterations, with a sup metric of 2.8757
36306422175e-7 and in 98.61899995803833 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (13, 0.010344827586206896, 3.0910
810216968847, 8.672089702483182)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 720 iterations, with a sup metric of 9.775
735776429428e-11 and in 48.812999963760376 seconds
      Klevel Aiyagari solved in 797 iterations, with a sup metric of 5.6000
22854182059e-7 and in 224.27099990844727 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (14, 0.011206896551724138, 3.1740
676628514732, 8.54435747408426)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 741 iterations, with a sup metric of 9.814
637991212294e-11 and in 51.170000076293945 seconds
      Klevel Aiyagari solved in 635 iterations, with a sup metric of 3.5227
21216514289e-7 and in 193.64100003242493 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (15, 0.01206896551724138, 3.22722
65082461424, 8.419681887353196)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 763 iterations, with a sup metric of 9.928
68010030179e-11 and in 59.4689998626709 seconds
      Klevel Aiyagari solved in 786 iterations, with a sup metric of 2.8190
27980293538e-7 and in 255.8420000076294 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (16, 0.01293103448275862, 3.39408
24682498687, 8.297962193240288)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 787 iterations, with a sup metric of 9.874
767670225992e-11 and in 53.01700019836426 seconds
      Klevel Aiyagari solved in 392 iterations, with a sup metric of 8.8478
33519695368e-8 and in 140.46400022506714 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (17, 0.013793103448275862, 3.6038
850430383333, 8.179101856593729)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 812 iterations, with a sup metric of 9.964
296054931765e-11 and in 54.58999991416931 seconds
      Klevel Aiyagari solved in 681 iterations, with a sup metric of 9.5139
32556231073e-7 and in 206.4909999370575 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (18, 0.014655172413793103, 3.7720
193921617806, 8.063008343233282)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 839 iterations, with a sup metric of 9.977
085824175447e-11 and in 56.40499997138977 seconds
      Klevel Aiyagari solved in 907 iterations, with a sup metric of 6.9098
53141314211e-7 and in 251.75600004196167 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (19, 0.015517241379310345, 4.2068
15056330746, 7.949592919602913)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 868 iterations, with a sup metric of 9.981
349080590007e-11 and in 61.245999813079834 seconds
      Klevel Aiyagari solved in 1270 iterations, with a sup metric of 2.046
106210707858e-7 and in 337.0059998035431 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (20, 0.016379310344827588, 5.2136
796243532695, 7.8387704641537015)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 900 iterations, with a sup metric of 9.869
438599707792e-11 and in 59.86399984359741 seconds
      Klevel Aiyagari solved in 730 iterations, with a sup metric of 8.8346
53607422343e-7 and in 222.1119999885559 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (21, 0.017241379310344827, 7.8024
293378003, 7.730459289672512)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 934 iterations, with a sup metric of 9.823
608593251265e-11 and in 62.169999837875366 seconds
      Klevel Aiyagari solved in 727 iterations, with a sup metric of 5.7695
94421424351e-7 and in 226.8619999885559 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (22, 0.01810344827586207, 8.73163
6995467163, 7.624580975830904)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 970 iterations, with a sup metric of 9.924
68329741314e-11 and in 64.47500014305115 seconds
      Klevel Aiyagari solved in 849 iterations, with a sup metric of 4.9303
40109328642e-7 and in 249.81500005722046 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (23, 0.01896551724137931, 10.0643
11022402082, 7.521060211282674)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1010 iterations, with a sup metric of 9.80
9042467168183e-11 and in 66.92499995231628 seconds
      Klevel Aiyagari solved in 1191 iterations, with a sup metric of 4.858
038119942293e-7 and in 325.5260000228882 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (24, 0.019827586206896553, 11.456
705673982183, 7.419824644687936)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1052 iterations, with a sup metric of 9.95
0085200216563e-11 and in 70.21300005912781 seconds
      Klevel Aiyagari solved in 665 iterations, with a sup metric of 7.4876
36894116139e-7 and in 214.40899991989136 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (25, 0.020689655172413793, 12.492
710819156969, 7.320804744087232)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1099 iterations, with a sup metric of 9.82
1121693676105e-11 and in 73.94299983978271 seconds
      Klevel Aiyagari solved in 911 iterations, with a sup metric of 5.3550
78171275853e-7 and in 280.78400015830994 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (26, 0.021551724137931036, 13.799
12484791632, 7.223933664090859)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1149 iterations, with a sup metric of 9.90
8873721542477e-11 and in 76.81500005722046 seconds
      Klevel Aiyagari solved in 673 iterations, with a sup metric of 9.1169
93059667254e-7 and in 225.6949999332428 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (27, 0.022413793103448276, 14.694
881451988957, 7.129147120387139)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1204 iterations, with a sup metric of 9.95
8256441677804e-11 and in 80.93099999427795 seconds
      Klevel Aiyagari solved in 903 iterations, with a sup metric of 9.9736
57177705446e-7 and in 283.17000007629395 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (28, 0.02327586206896552, 16.8624
96506047098, 7.036383271108701)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1266 iterations, with a sup metric of 9.90
6031550599437e-11 and in 84.9079999923706 seconds
      Klevel Aiyagari solved in 751 iterations, with a sup metric of 3.4636
65422308832e-7 and in 253.31299996376038 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (29, 0.02413793103448276, 22.0311
87130082852, 6.945582604628445)
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1333 iterations, with a sup metric of 9.96
0032798517204e-11 and in 89.83700013160706 seconds
      Klevel Aiyagari solved in 900 iterations, with a sup metric of 4.5905
13711348224e-7 and in 290.21399998664856 seconds
(r, r_range[r], Supply1[r], Demand1[r]) = (30, 0.025, 23.44704433236027, 6.
856687833386916)
~~~~



~~~~{.julia}
plot([Supply1 Demand1], r_range)
~~~~~~~~~~~~~




### __3.5 Convergence to Equilibrium__


__Steps for Convergence to Equilibrium__

1. We start with an interest rate $r_1=\frac{1-\beta}{\beta}-\epsilon$ and find the demand for capital $K(r_1)$.
2. We evaluate $KlevelAiyagari(u, du, Kr(r_1, δ, α), θ, pop)$ and obtain the supply of capital $A(r_1)$, with this we establish the interest rate $r_2=Min\left\{f_K\left(A(r_1),1,\alpha\right)-\delta, \frac{1-\beta}{\beta}\right\}$. By definition $r_1$ and $r_2$ are at opposite sides of the equilibrium. Without loss of generality less assume $r_1>r_2$.
3. We define $r =\frac{r_1+r_2}{2}$ and run $KlevelAiyagari(u, du, Kr(r_1, δ, α), θ, pop)$ in order to find $A(r)$. If $A(r)> K(r)$ we replace $r_1 =r$, if $A(r) < K(r)$ we replace $r_2 =r$.
4. If $\lvert A(r) - K(r)\lvert<0.01$ or if $\lvert r_1 -r_2 \lvert < 1E^{-4}$ we declare $(A(r), r)$ as the equilibrium. Otherwise repeat 3. 

~~~~{.julia}
function EquilibriumAiyagari(u, du, θ; tol = 0.01, max_iter = 100)
    start = time()
    println("  ")
    println("  ")
    println("   Equilibrium(Aiyagari)...")
    # Extract and define the internal parameters
    β, σ, cc, λ, σϵ, N, a_max, a_grid_size, α, δ = θ[1], θ[2], θ[3], θ[4], θ[5], convert(Int,θ[6]), θ[7], convert(Int,θ[8]), θ[9], θ[10];
    b = -cc;
    # Initial interest condition
    r1 = (1-β)/β - eps();
    K1 = Kr(r1, δ, α);
    w1 = fL(K1, 1, α);
    @assert fK(K1, 1, α) - δ ≈ r1 "Equality in implied interest rates"
    K_path = [K1];
    r_path = [r1];
    w_path = [w1];
    iter  = 0;
    println("In iteration $iter we have (K0, r0, w0) = ($K1, $r1, $w1).")
    # Extract elements from the KlevelAiyagari algorithm
    K2 = KlevelAiyagari(u, du, Kr(r1, θ[10], θ[9]), θ, pop)[1];
    r2 = min(fK(K2, 1, α) - δ, (1-β)/β);
    w2 = fL(K2, 1, α);
    K_path = [K_path; K2];
    r_path = [r_path; r2];
    w_path = [w_path; w2];
    iter += 1;
    r = r1;
    K = K2;
    w = w2;
    sup = Inf;
    println("In iteration $iter we have (K, r1, r2, w, sup) = ($K, $r1, $r2, $w, $sup).")
    while (isnan(K) == 1) | (sup > tol && iter ≤ max_iter)
        iter += 1;
        r = (r1 + r2) /2;
        KlevelAiyagari_sol  = KlevelAiyagari(u, du, Kr(r, θ[10], θ[9]), θ, pop);
        K = KlevelAiyagari_sol[1];
        if isnan(K) == 1
            K = 0.0
        end
        if  K > Kr(r, δ, α)
            if r1 < r2
                r2 = r;
            elseif r1 > r2
                r1 = r
            end
        elseif K < Kr(r, δ, α)
            if r1 < r2
                r1 = r;
            elseif r1 > r2
                r2 = r
            end
        end
        w = fL(K, 1, α);
        sup = abs(Kr(r, δ, α) - K);
        K_path = [K_path; K];
        r_path = [r_path; r];
        w_path = [w_path; w];
        println("In iteration $iter we have (K, r1, r2, w, sup) = ($K, $r1, $r2, $w, $sup).")
        if (sup ≤ tol) | (abs(r1-r2) < 1E-4)
            f = KlevelAiyagari_sol[3];
            a_sup = KlevelAiyagari_sol[4];
            a_grid = KlevelAiyagari_sol[5];
            A_prime = KlevelAiyagari_sol[6];
            V = KlevelAiyagari_sol[7];
            Y = KlevelAiyagari_sol[8];
            W = KlevelAiyagari_sol[9];
            Afin = KlevelAiyagari_sol[10];
            Yfin = KlevelAiyagari_sol[11];
            break
        end
    end
    K_path = K_path
    r_path = r_path
    w_path = w_path
    K_end = K_path[iter+1];
    r_end = r_path[iter+1];
    w_end = w_path[iter+1];
    f = f;
    a_sup = a_sup;
    a_grid = a_grid;
    A_prime = A_prime;
    Afin = Afin;
    Yfin = Yfin;
    V = V;
    Y = Y;
    W = W;
    savings = δ*α  / (r + δ);
    elapsed = time() - start
    println("Equilibrium found in $elapsed seconds")
    return sol = K_end, r_end, w_end, savings, f, a_sup, a_grid, A_prime, V, Y, K, K_path, r_path, w_path, Afin, Yfin
end
~~~~~~~~~~~~~


~~~~
EquilibriumAiyagari (generic function with 1 method)
~~~~




To have an idea about the way in which this algorithm solves this problem let's run non silently one time

~~~~{.julia}
θ[5] = 0.4;
θ[2] = 3.0;
θ[4] = 0.6;
E = EquilibriumAiyagari(u, du, θ)
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.3
9433607637101886 and in 672.0439999103546 seconds
      Klevel Aiyagari solved in 262 iterations, with a sup metric of 8.1891
6199636994e-7 and in 731.3789999485016 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.090463130973486, 0.0416666
66666666484, -0.03301907045968469, 2.012075290577474, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 596 iterations, with a sup metric of 9.850
253945842269e-11 and in 39.62100005149841 seconds
      Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.00
012803928504117617 and in 2344.9100000858307 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (5.644547421804616, 0.04166666
6666666484, 0.004323798103490897, 1.1933462133167063, 4.0143050517329995).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1210 iterations, with a sup metric of 9.89
0754881780595e-11 and in 81.39000010490417 seconds
      Klevel Aiyagari solved in 780 iterations, with a sup metric of 9.0077
83916472509e-7 and in 261.45499992370605 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (18.89615526509504, 0.02299523
238507869, 0.004323798103490897, 1.8436327554500291, 11.829792897018539).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 798 iterations, with a sup metric of 9.969
802761133906e-11 and in 55.00599980354309 seconds
      Klevel Aiyagari solved in 320 iterations, with a sup metric of 6.5433
69426278254e-7 and in 129.5019998550415 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (8.384375541809163, 0.01365951
5244284793, 0.004323798103490897, 1.3760348528585757, 0.18703825579185995).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 683 iterations, with a sup metric of 9.721
468074985751e-11 and in 45.50800013542175 seconds
      Klevel Aiyagari solved in 446 iterations, with a sup metric of 7.1071
30360237938e-7 and in 146.54500007629395 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (7.13468935433615, 0.013659515
244284793, 0.008991656673887845, 1.2983596156669024, 1.7443174574891618).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 736 iterations, with a sup metric of 9.852
385574049549e-11 and in 48.79299998283386 seconds
      Klevel Aiyagari solved in 1901 iterations, with a sup metric of 5.226
394770261449e-7 and in 482.7260000705719 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (7.724978846992573, 0.01365951
5244284793, 0.01132558595908632, 1.3360508927987846, 0.8020341932944879).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 766 iterations, with a sup metric of 9.838
707626386167e-11 and in 51.95099997520447 seconds
      Klevel Aiyagari solved in 1962 iterations, with a sup metric of 6.691
91768712892e-7 and in 500.760999917984 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (8.053595570564703, 0.01365951
5244284793, 0.012492550601685557, 1.3562392473980458, 0.3059150032362137).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 782 iterations, with a sup metric of 9.814
016266318504e-11 and in 51.99499988555908 seconds
      Klevel Aiyagari solved in 622 iterations, with a sup metric of 7.5908
61702780849e-7 and in 193.64299988746643 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (8.135350853929884, 0.01365951
5244284793, 0.013076032922985176, 1.3611796181593974, 0.1424217923055).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 790 iterations, with a sup metric of 9.869
438599707792e-11 and in 52.54099988937378 seconds
      Klevel Aiyagari solved in 7570 iterations, with a sup metric of 4.344
945761368411e-7 and in 1799.6829998493195 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (8.289810101759608, 0.01336777
4083634983, 0.013076032922985176, 1.370427369431562, 0.052416144905469864).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 786 iterations, with a sup metric of 9.836
043091127067e-11 and in 52.2279999256134 seconds
      Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.00
08102922151328672 and in 2355.59299993515 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (8.177667626398618, 0.0133677
74083634983, 0.01322190350331008, 1.3637242956576483, 0.07987519839303658).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 788 iterations, with a sup metric of 9.851
319759945909e-11 and in 53.34500002861023 seconds
      Klevel Aiyagari solved in 3239 iterations, with a sup metric of 8.483
692643414498e-7 and in 806.4760000705719 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (8.224718058744928, 0.0133677
74083634983, 0.01329483879347253, 1.3665437523171755, 0.02274024106915995).
Equilibrium found in 9752.78299999237 seconds
(8.224718058744928, 0.01329483879347253, 1.3665437523171755, 0.308698748745
9491, [0.0 0.0 … 0.0 0.0; 9.999999999999998e-5 0.0 … 0.0 0.0; … ; 0.0 0.0 …
 0.00019999999999999996 0.0016999999999999997; 0.0 0.0 … 0.0 0.000199999999
99999996], 25.0, [4.0960000000000004e-16, 5.2428800000000005e-14, 8.9579520
00000001e-13, 6.710886400000001e-12, 3.2000000000000006e-11, 1.146617856000
0002e-10, 3.373232128e-10, 8.589934592000001e-10, 1.959104102399999e-9, 4.0
96000000000001e-9  …  19.3410142982354, 19.90982807496902, 20.4929208725502
75, 21.090590156950757, 21.703138331168, 22.3308727964269, 22.9741060138848
5, 23.633155566842625, 24.308344223463116, 25.0], [4.0960000000000004e-16 4
.0960000000000004e-16 … 22.97410601388485 23.633155566842625; 4.09600000000
00004e-16 4.0960000000000004e-16 … 22.97410601388485 23.633155566842625; … 
; 2.2293220545737324 2.2293220545737324 … 25.0 25.0; 4.391392213728001 4.39
1392213728001 … 25.0 25.0], [-0.7474660931823891 -0.7474660931805346 … 12.9
73532827064554 13.04344361455025; 4.234784466642006 4.23478446664242 … 13.0
62015740099936 13.128285444748657; … ; 10.308805890773472 10.30880589077348
8 … 13.601008950659592 13.615093516207686; 11.271336649990749 11.2713366499
9076 … 13.696710286382915 13.699799423687711], [0.22313016014842982, 0.3678
7944117144233, 0.6065306597126334, 1.0, 1.6487212707001282, 2.7182818284590
45, 4.4816890703380645], 8.224718058744928, [5.446807380113248, 24.09046313
0973486, 5.644547421804616, 18.89615526509504, 8.384375541809163, 7.1346893
5433615, 7.724978846992573, 8.053595570564703, 8.135350853929884, 8.2898101
01759608, 8.177667626398618, 8.224718058744928], [0.041666666666666484, -0.
03301907045968469, 0.004323798103490897, 0.02299523238507869, 0.01365951524
4284793, 0.008991656673887845, 0.01132558595908632, 0.012492550601685557, 0
.013076032922985176, 0.013367774083634983, 0.01322190350331008, 0.013294838
79347253], [1.1781242629578268, 2.012075290577474, 1.1933462133167063, 1.84
36327554500291, 1.3760348528585757, 1.2983596156669024, 1.3360508927987846,
 1.3562392473980458, 1.3611796181593974, 1.370427369431562, 1.3637242956576
483, 1.3665437523171755], [3.0378200682559995; 2.7093160131680527; … ; 11.5
9033733726119; 1.0523006004525135], [1.0, 1.0, 1.0, 1.0, 1.6487212707001282
, 1.0, 1.0, 2.718281828459045, 0.36787944117144233, 1.6487212707001282  …  
1.0, 0.6065306597126334, 1.0, 0.36787944117144233, 0.6065306597126334, 0.60
65306597126334, 1.0, 0.36787944117144233, 1.0, 0.22313016014842982])
~~~~




We can see the way in which the algorithm converges to the equilibrium valu of $K$ and $r$

~~~~{.julia}
plot(1:size(E[12])[1], E[12], title ="Convergence Path of K",xlabel="Algorithm iteration", ylabel="Level of per capita aggregate Capital", legend=:bottomright, label="K")
~~~~~~~~~~~~~



~~~~{.julia}
plot(1:size(E[13])[1], E[13], title ="Convergence Path of r",xlabel="Algorithm iteration", ylabel="Level of interest rate", legend=:bottomright, label="K")
~~~~~~~~~~~~~




Now we estimate the final results in table A and B of Aiyagari and compare our results with his estimations (the following line of commands takes around one day to run):

~~~~{.julia}
#TABLE A
θ[5] = 0.2;
# Column 1
θ[2] = 1.0;
θ[4] = 0.0;
AE11 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.8
027649081013806 and in 204.34899997711182 seconds
      Klevel Aiyagari solved in 102 iterations, with a sup metric of NaN an
d in 227.09899997711182 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (25.0, 0.041666666666666484, -
0.034120265586010584, 2.0390993072884185, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 597 iterations, with a sup metric of 9.832
135106080386e-11 and in 12.1489999294281 seconds
      Klevel Aiyagari solved in 116 iterations, with a sup metric of 1.1134
975191865916e-7 and in 38.83800005912781 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (0.24044859952072403, 0.041666
666666666484, 0.00377320054032795, 0.38313161004239993, 9.517778618754374).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1208 iterations, with a sup metric of 9.88
0452012112073e-11 and in 24.5770001411438 seconds
      Klevel Aiyagari solved in 457 iterations, with a sup metric of 3.6163
705484080583e-7 and in 127.66000008583069 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (24.1931830415082, 0.022719933
603497215, 0.00377320054032795, 2.0151596473705498, 17.09720697841067).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 794 iterations, with a sup metric of 9.738
698736327933e-11 and in 16.200000047683716 seconds
      Klevel Aiyagari solved in 250 iterations, with a sup metric of 2.6791
6698403066e-7 and in 74.2979998588562 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (0.40489682381134634, 0.022719
933603497215, 0.013246567071912583, 0.4621936209382296, 7.849233587770718).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 951 iterations, with a sup metric of 9.847
767046267109e-11 and in 22.82200002670288 seconds
      Klevel Aiyagari solved in 683 iterations, with a sup metric of 4.4396
719934260654e-7 and in 183.77600002288818 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (0.7125004612814515, 0.0227199
33603497215, 0.017983250337704898, 0.5664774519880301, 6.926699962337437).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1056 iterations, with a sup metric of 9.95
8611713045684e-11 and in 21.527999877929688 seconds
      Klevel Aiyagari solved in 1302 iterations, with a sup metric of 7.556
573663816092e-7 and in 322.71799993515015 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (2.4912566732817663, 0.0227199
33603497215, 0.020351591970601057, 0.8889772049830998, 4.8681193079523934).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1130 iterations, with a sup metric of 9.92
521620446496e-11 and in 22.9520001411438 seconds
      Klevel Aiyagari solved in 1586 iterations, with a sup metric of 5.176
87673922019e-7 and in 396.2699999809265 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (23.368946770944408, 0.0215357
62787049134, 0.020351591970601057, 1.9901694903901943, 16.143238657596058).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1091 iterations, with a sup metric of 9.88
5425811262394e-11 and in 22.450999975204468 seconds
      Klevel Aiyagari solved in 1699 iterations, with a sup metric of 5.884
694791298115e-7 and in 421.30399990081787 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (12.333917003909303, 0.0209436
77378825097, 0.020351591970601057, 1.5811619030741213, 5.04187722246087).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1073 iterations, with a sup metric of 9.86
5175343293231e-11 and in 22.450999975204468 seconds
      Klevel Aiyagari solved in 1086 iterations, with a sup metric of 3.407
1844711255647e-7 and in 279.25300002098083 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (9.15142503898559, 0.020647634
674713075, 0.020351591970601057, 1.4200901516240445, 1.8258440404132052).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1065 iterations, with a sup metric of 9.79
8384326131782e-11 and in 23.288000106811523 seconds
      Klevel Aiyagari solved in 425 iterations, with a sup metric of 1.3429
616936342354e-7 and in 123.58700013160706 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (4.905391024954213, 0.0206476
34674713075, 0.020499613322657068, 1.1345469733060112, 2.437055577782562).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1069 iterations, with a sup metric of 9.82
7161306930066e-11 and in 22.91000008583069 seconds
      Klevel Aiyagari solved in 963 iterations, with a sup metric of 4.5803
545584751234e-7 and in 250.83699989318848 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (7.621879051301215, 0.0205736
2399868507, 0.020499613322657068, 1.3296040000776064, 0.28787320154163965).
Equilibrium found in 2445.6490001678467 seconds
~~~~



~~~~{.julia}
θ[4] = 0.3;
AE21 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.8
033331071692373 and in 212.68899989128113 seconds
      Klevel Aiyagari solved in 102 iterations, with a sup metric of NaN an
d in 236.37700009346008 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (25.0, 0.041666666666666484, -
0.034120265586010584, 2.0390993072884185, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 597 iterations, with a sup metric of 9.784
17347141658e-11 and in 12.79200005531311 seconds
      Klevel Aiyagari solved in 235 iterations, with a sup metric of 8.3108
56138510572e-7 and in 68.45799994468689 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (0.2831653345492499, 0.0416666
66666666484, 0.00377320054032795, 0.4063632309827007, 9.475061883725848).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1208 iterations, with a sup metric of 9.87
0149142443552e-11 and in 24.603000164031982 seconds
      Klevel Aiyagari solved in 459 iterations, with a sup metric of 1.6661
346952429348e-7 and in 131.1890001296997 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (24.104271233244862, 0.0227199
33603497215, 0.00377320054032795, 2.0124903936376723, 17.008295170147335).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 793 iterations, with a sup metric of 9.983
835980165168e-11 and in 16.378000020980835 seconds
      Klevel Aiyagari solved in 322 iterations, with a sup metric of 8.6813
47033696897e-7 and in 92.58500003814697 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (0.5430208556117457, 0.0227199
33603497215, 0.013246567071912583, 0.5137051474449255, 7.711109555970319).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 951 iterations, with a sup metric of 9.862
68844371807e-11 and in 19.457000017166138 seconds
      Klevel Aiyagari solved in 119 iterations, with a sup metric of 3.2773
651402088434e-7 and in 46.85800004005432 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (1.03380431475251, 0.022719933
603497215, 0.017983250337704898, 0.6477057858314784, 6.605396108866378).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1057 iterations, with a sup metric of 9.89
9636665977596e-11 and in 21.384999990463257 seconds
      Klevel Aiyagari solved in 788 iterations, with a sup metric of 7.7861
51811406918e-7 and in 205.21799993515015 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (5.758541644852679, 0.02271993
3603497215, 0.020351591970601057, 1.2019668173441722, 1.6008343363814808).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1130 iterations, with a sup metric of 9.90
7452636070957e-11 and in 22.98800015449524 seconds
      Klevel Aiyagari solved in 766 iterations, with a sup metric of 9.0267
8028377072e-7 and in 199.60400009155273 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (23.196517949944326, 0.0215357
62787049134, 0.020351591970601057, 1.984870518084036, 15.970809836595976).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1091 iterations, with a sup metric of 9.83
7819447966467e-11 and in 22.2020001411438 seconds
      Klevel Aiyagari solved in 743 iterations, with a sup metric of 7.8028
22289042447e-7 and in 195.2960000038147 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (12.303945969432068, 0.0209436
77378825097, 0.020351591970601057, 1.5797776433757384, 5.011906187983635).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1073 iterations, with a sup metric of 9.97
3533110496646e-11 and in 21.45299983024597 seconds
      Klevel Aiyagari solved in 702 iterations, with a sup metric of 4.5295
803571995944e-7 and in 182.7850000858307 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (9.561447754202126, 0.02064763
4674713075, 0.020351591970601057, 1.442674947671138, 2.2358667556297416).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1065 iterations, with a sup metric of 9.91
2071163853398e-11 and in 21.69000005722046 seconds
      Klevel Aiyagari solved in 439 iterations, with a sup metric of 5.7086
2214213154e-7 and in 124.34400010108948 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (6.859429756728951, 0.0206476
34674713075, 0.020499613322657068, 1.2800992959074993, 0.4830168460078239).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1069 iterations, with a sup metric of 9.93
9071787812281e-11 and in 21.23199987411499 seconds
      Klevel Aiyagari solved in 735 iterations, with a sup metric of 2.1658
479551011958e-7 and in 193.30299997329712 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (8.60819618705661, 0.02057362
399868507, 0.020499613322657068, 1.3891474634919971, 1.2741903372970347).
Equilibrium found in 1676.023999929428 seconds
~~~~



~~~~{.julia}
θ[4] = 0.6;
AE31 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.8
060872231908434 and in 204.1529998779297 seconds
      Klevel Aiyagari solved in 102 iterations, with a sup metric of NaN an
d in 226.86999988555908 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (25.0, 0.041666666666666484, -
0.034120265586010584, 2.0390993072884185, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 597 iterations, with a sup metric of 9.771
738973540778e-11 and in 12.094000101089478 seconds
      Klevel Aiyagari solved in 1244 iterations, with a sup metric of 8.771
664674325858e-7 and in 306.6579999923706 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (0.3946113986788464, 0.0416666
66666666484, 0.00377320054032795, 0.4579320490841394, 9.363615819596252).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1208 iterations, with a sup metric of 9.87
4412398858112e-11 and in 25.21399998664856 seconds
      Klevel Aiyagari solved in 1119 iterations, with a sup metric of 9.695
199880022036e-7 and in 290.83899998664856 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (23.858625717873245, 0.0227199
33603497215, 0.00377320054032795, 2.005082870026346, 16.762649654775718).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 794 iterations, with a sup metric of 9.773
870601748058e-11 and in 16.688000202178955 seconds
      Klevel Aiyagari solved in 144 iterations, with a sup metric of 6.5091
148214881e-7 and in 51.83300018310547 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (0.901905659332677, 0.02271993
3603497215, 0.013246567071912583, 0.6166489187464939, 7.352224752249388).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 952 iterations, with a sup metric of 9.842
082704381028e-11 and in 20.155999898910522 seconds
      Klevel Aiyagari solved in 310 iterations, with a sup metric of 4.7810
405166344945e-8 and in 95.33699989318848 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (2.036637792727779, 0.02271993
3603497215, 0.017983250337704898, 0.8267780563536635, 5.602562630891109).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1060 iterations, with a sup metric of 9.82
7516578297946e-11 and in 22.0479998588562 seconds
      Klevel Aiyagari solved in 961 iterations, with a sup metric of 7.6669
67375831459e-7 and in 252.93999981880188 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (9.74223113184212, 0.020351591
970601057, 0.017983250337704898, 1.4524360120617021, 2.38285515060796).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1002 iterations, with a sup metric of 9.90
8518450174597e-11 and in 21.75499987602234 seconds
      Klevel Aiyagari solved in 144 iterations, with a sup metric of 8.0082
19553533355e-7 and in 56.549999952316284 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (2.6159632422821364, 0.0203515
91970601057, 0.01916742115415298, 0.9047474409788344, 4.881184414355267).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1029 iterations, with a sup metric of 9.97
2112025025126e-11 and in 21.21499991416931 seconds
      Klevel Aiyagari solved in 120 iterations, with a sup metric of 5.8787
96658721665e-7 and in 49.644999980926514 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (3.1428245714167393, 0.0203515
91970601057, 0.019759506562377016, 0.9665297457754889, 4.284913416353238).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1044 iterations, with a sup metric of 9.90
0702480081236e-11 and in 22.100000143051147 seconds
      Klevel Aiyagari solved in 495 iterations, with a sup metric of 5.7347
53815878104e-7 and in 139.9210000038147 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (6.933844427536611, 0.02035159
1970601057, 0.020055549266489038, 1.285081429077345, 0.45958297897770795).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1052 iterations, with a sup metric of 9.84
1727433013148e-11 and in 21.793999910354614 seconds
      Klevel Aiyagari solved in 592 iterations, with a sup metric of 2.7568
5218369594e-7 and in 163.80099987983704 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (8.153053494423482, 0.0202035
70618545046, 0.020055549266489038, 1.3622451772651356, 0.7766840246041395).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1048 iterations, with a sup metric of 9.86
6241157396871e-11 and in 21.19100022315979 seconds
      Klevel Aiyagari solved in 672 iterations, with a sup metric of 1.2123
954940182705e-7 and in 178.69900012016296 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (7.499163642696772, 0.0201295
59942517042, 0.020055549266489038, 1.3218573585394553, 0.114273281759111).
Equilibrium found in 1813.1019999980927 seconds
~~~~



~~~~{.julia}
θ[4] = 0.9;
AE41 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.8
322313018379646 and in 206.03099989891052 seconds
      Klevel Aiyagari solved in 102 iterations, with a sup metric of NaN an
d in 230.4670000076294 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (25.0, 0.041666666666666484, -
0.034120265586010584, 2.0390993072884185, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 597 iterations, with a sup metric of 9.868
728056972031e-11 and in 12.23800015449524 seconds
      Klevel Aiyagari solved in 183 iterations, with a sup metric of 8.7114
84962422957e-7 and in 56.23900008201599 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (1.379314888167871, 0.04166666
6666666484, 0.00377320054032795, 0.7185529994705138, 8.378912330107227).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1210 iterations, with a sup metric of 9.96
6782954506925e-11 and in 24.76900005340576 seconds
      Klevel Aiyagari solved in 510 iterations, with a sup metric of 9.2509
30507839168e-7 and in 144.46799993515015 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (16.89954892354954, 0.02271993
3603497215, 0.00377320054032795, 1.7709855373787804, 9.803572860452011).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 796 iterations, with a sup metric of 9.887
912710837554e-11 and in 16.567999839782715 seconds
      Klevel Aiyagari solved in 483 iterations, with a sup metric of 4.4502
127580298635e-7 and in 129.76799988746643 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (4.033277610141211, 0.02271993
3603497215, 0.013246567071912583, 1.0573454373410842, 4.220852801440853).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 959 iterations, with a sup metric of 9.784
88401415234e-11 and in 19.819000005722046 seconds
      Klevel Aiyagari solved in 125 iterations, with a sup metric of 2.9034
186957382405e-7 and in 49.21300005912781 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (7.8539015085742765, 0.0179832
50337704898, 0.013246567071912583, 1.3440354887274624, 0.2147010849553883).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 870 iterations, with a sup metric of 9.772
449516276538e-11 and in 17.759000062942505 seconds
      Klevel Aiyagari solved in 453 iterations, with a sup metric of 7.6024
75814983545e-7 and in 123.77900004386902 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (6.210250938062597, 0.01798325
0337704898, 0.01561490870480874, 1.2350918833079512, 1.7266577607566234).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 912 iterations, with a sup metric of 9.847
411774899228e-11 and in 18.873000144958496 seconds
      Klevel Aiyagari solved in 1850 iterations, with a sup metric of 5.882
934329001125e-7 and in 451.8829998970032 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (7.043374401096339, 0.01798325
0337704898, 0.016799079521256817, 1.2923526950516178, 0.7423470939235548).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 935 iterations, with a sup metric of 9.793
055255613581e-11 and in 19.28500008583069 seconds
      Klevel Aiyagari solved in 169 iterations, with a sup metric of 2.3275
479793885982e-7 and in 58.604000091552734 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (7.4716627463908925, 0.0179832
50337704898, 0.017391164929480857, 1.3201102033540946, 0.24022756783765153)
.
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 946 iterations, with a sup metric of 9.983
125437429408e-11 and in 19.581000089645386 seconds
      Klevel Aiyagari solved in 4309 iterations, with a sup metric of 3.166
9756475624295e-7 and in 1040.5039999485016 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (7.750628021755323, 0.01768720
7633592876, 0.017391164929480857, 1.3376461868898586, 0.07522377456390661).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 940 iterations, with a sup metric of 9.996
27047804097e-11 and in 19.263000011444092 seconds
      Klevel Aiyagari solved in 479 iterations, with a sup metric of 6.0113
6972419892e-7 and in 136.08399987220764 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (7.5376101800052595, 0.017687
207633592876, 0.01753918628153687, 1.3242930395638879, 0.1560016294945017).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 943 iterations, with a sup metric of 9.987
388693843968e-11 and in 19.058000087738037 seconds
      Klevel Aiyagari solved in 835 iterations, with a sup metric of 8.2434
60726366645e-7 and in 219.39100003242493 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (7.558415738249893, 0.0176872
07633592876, 0.017613196957564872, 1.3256078075005227, 0.12608344624340795)
.
Equilibrium found in 2640.4079999923706 seconds
~~~~



~~~~{.julia}
# Column 2
θ[2] = 3.0;
θ[4] = 0.0;
AE12 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.3
972162672821469 and in 659.295000076294 seconds
      Klevel Aiyagari solved in 112 iterations, with a sup metric of 3.1554
436208840472e-30 and in 684.7360000610352 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.985129400804457, 0.0416666
66666666484, -0.034102791228462916, 2.0386625782979406, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 587 iterations, with a sup metric of 9.861
97790098231e-11 and in 39.35599994659424 seconds
      Klevel Aiyagari solved in 147 iterations, with a sup metric of 5.1344
31841695314e-7 and in 73.62199997901917 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (0.573447811968491, 0.04166666
6666666484, 0.003781937719101784, 0.5238871884306928, 9.183189399309354).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1188 iterations, with a sup metric of 9.99
2717764362169e-11 and in 81.42400002479553 seconds
      Klevel Aiyagari solved in 1581 iterations, with a sup metric of 9.131
585217300397e-7 and in 444.06700015068054 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (23.334577120955824, 0.0227243
02192884134, 0.003781937719101784, 1.9891152660324745, 16.239072572300916).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 782 iterations, with a sup metric of 9.935
163802765601e-11 and in 52.258999824523926 seconds
      Klevel Aiyagari solved in 214 iterations, with a sup metric of 9.0031
78911048675e-7 and in 101.38199996948242 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (0.830601472085777, 0.02272430
2192884134, 0.013253119955992959, 0.5986339522173176, 7.422622681433283).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 939 iterations, with a sup metric of 9.781
153664789599e-11 and in 63.003000020980835 seconds
      Klevel Aiyagari solved in 216 iterations, with a sup metric of 8.1838
25571129114e-7 and in 111.4670000076294 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (1.280533726585738, 0.02272430
2192884134, 0.017988711074438547, 0.6995854069015885, 6.358001521421448).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1045 iterations, with a sup metric of 9.91
5268606164318e-11 and in 68.77800011634827 seconds
      Klevel Aiyagari solved in 730 iterations, with a sup metric of 7.5049
51868229514e-7 and in 238.9610002040863 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (6.455331575281151, 0.02272430
2192884134, 0.020356506633661342, 1.2524219013690805, 0.9034812829751795).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1112 iterations, with a sup metric of 9.87
9741469376313e-11 and in 73.08699989318848 seconds
      Klevel Aiyagari solved in 1031 iterations, with a sup metric of 3.001
5031327623546e-7 and in 312.1899998188019 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (12.180325159282862, 0.0215404
04413272738, 0.020356506633661342, 1.574045099348536, 4.9551331367529015).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1077 iterations, with a sup metric of 9.89
2886509987875e-11 and in 74.15100002288818 seconds
      Klevel Aiyagari solved in 936 iterations, with a sup metric of 4.8631
78635138588e-7 and in 297.09500002861023 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (10.235978874009266, 0.0209484
5552346704, 0.020356506633661342, 1.478517778547018, 2.9444783832066594).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1061 iterations, with a sup metric of 9.85
558301636047e-11 and in 70.90400004386902 seconds
      Klevel Aiyagari solved in 916 iterations, with a sup metric of 2.2554
005808435052e-7 and in 299.7519998550415 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (8.934067642459693, 0.02065248
107856419, 0.020356506633661342, 1.407854259221522, 1.609037770440465).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1053 iterations, with a sup metric of 9.87
2280770650832e-11 and in 70.99199986457825 seconds
      Klevel Aiyagari solved in 659 iterations, with a sup metric of 4.9575
0120566852e-7 and in 228.7350001335144 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (7.225013901612801, 0.0206524
8107856419, 0.020504493856112766, 1.30425317111234, 0.1168755965748991).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1057 iterations, with a sup metric of 9.85
878045867139e-11 and in 68.90199995040894 seconds
      Klevel Aiyagari solved in 1083 iterations, with a sup metric of 4.979
013691919333e-7 and in 328.65499997138977 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (8.494109717061006, 0.0205784
8746733848, 0.020504493856112766, 1.3824912965410254, 1.1606579778175137).
Equilibrium found in 3120.6689999103546 seconds
~~~~



~~~~{.julia}
θ[4] = 0.3;
AE22 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.3
9704988096491434 and in 652.3789999485016 seconds
      Klevel Aiyagari solved in 118 iterations, with a sup metric of 2.6707
674807162576e-26 and in 680.0120000839233 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.980979466145236, 0.0416666
66666666484, -0.03409791161905917, 2.0385406707510523, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 587 iterations, with a sup metric of 9.766
232267338637e-11 and in 38.598999977111816 seconds
      Klevel Aiyagari solved in 251 iterations, with a sup metric of 6.6679
21581703122e-7 and in 95.87599992752075 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (0.8110614029890371, 0.0416666
66666666484, 0.0037843775238036587, 0.5935254123667423, 8.94513188427129).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1188 iterations, with a sup metric of 9.95
0795742952323e-11 and in 78.60899996757507 seconds
      Klevel Aiyagari solved in 1692 iterations, with a sup metric of 8.494
351171313448e-7 and in 470.8659999370575 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (23.273630864196015, 0.0227255
2209523507, 0.0037843775238036587, 1.9872434080342425, 16.17825797394001).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 782 iterations, with a sup metric of 9.890
754881780595e-11 and in 51.04900002479553 seconds
      Klevel Aiyagari solved in 155 iterations, with a sup metric of 2.1756
408375037747e-7 and in 87.51600003242493 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (1.2984481549530822, 0.0227255
2209523507, 0.013254949809519365, 0.7030930983692315, 6.954522960590387).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 939 iterations, with a sup metric of 9.806
555567593023e-11 and in 61.75699996948242 seconds
      Klevel Aiyagari solved in 369 iterations, with a sup metric of 5.9596
45266401851e-7 and in 146.5369999408722 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (1.8746894195836838, 0.0227255
2209523507, 0.01799023595237722, 0.8024805816307351, 5.763660099094334).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1046 iterations, with a sup metric of 9.84
4924875324068e-11 and in 68.49200010299683 seconds
      Klevel Aiyagari solved in 1032 iterations, with a sup metric of 5.831
841972330487e-7 and in 311.489000082016 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (8.462566535601752, 0.02035787
9023806147, 0.01799023595237722, 1.3806408805920654, 1.103910913433599).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 988 iterations, with a sup metric of 9.962
164426724485e-11 and in 63.979000091552734 seconds
      Klevel Aiyagari solved in 228 iterations, with a sup metric of 7.8260
75521881392e-7 and in 118.39199995994568 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (3.0143235106943402, 0.0203578
79023806147, 0.019174057488091683, 0.9521126249495383, 4.482040286731297).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1016 iterations, with a sup metric of 9.90
7452636070957e-11 and in 68.58999991416931 seconds
      Klevel Aiyagari solved in 490 iterations, with a sup metric of 7.3954
90995153611e-7 and in 185.31200003623962 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (5.6362166179311215, 0.0203578
79023806147, 0.019765968255948917, 1.1927118588912418, 1.7907696904751687).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1031 iterations, with a sup metric of 9.82
8582392401586e-11 and in 69.62200021743774 seconds
      Klevel Aiyagari solved in 978 iterations, with a sup metric of 9.1691
38643885059e-7 and in 306.87400007247925 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (7.345111818323424, 0.02035787
9023806147, 0.020061923639877532, 1.3120168183961871, 0.04757967479355063).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1038 iterations, with a sup metric of 9.93
196636045468e-11 and in 68.97399997711182 seconds
      Klevel Aiyagari solved in 1722 iterations, with a sup metric of 2.379
0802898390747e-7 and in 490.1840000152588 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (8.043099438066115, 0.0202099
0133184184, 0.020061923639877532, 1.3556026578396592, 0.6674580782242376).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1034 iterations, with a sup metric of 9.97
4598924600286e-11 and in 69.61500000953674 seconds
      Klevel Aiyagari solved in 696 iterations, with a sup metric of 4.3092
86837029954e-7 and in 235.1449999809265 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (7.256470947933149, 0.0202099
0133184184, 0.020135912485859685, 1.3062946250820389, 0.127687407905543).
Equilibrium found in 3128.2109999656677 seconds
~~~~



~~~~{.julia}
θ[4] = 0.6;
AE32 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.3
963668504738962 and in 658.2510001659393 seconds
      Klevel Aiyagari solved in 169 iterations, with a sup metric of 2.8477
878678478526e-28 and in 697.0250000953674 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.96776884081338, 0.04166666
6666666484, -0.03408236933988199, 2.038152512070527, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 586 iterations, with a sup metric of 9.950
085200216563e-11 and in 38.64699983596802 seconds
      Klevel Aiyagari solved in 231 iterations, with a sup metric of 4.1027
50099065068e-7 and in 92.16100001335144 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (1.4594710854202522, 0.0416666
66666666484, 0.003792148663392246, 0.7333146347540146, 8.295308458811407).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1186 iterations, with a sup metric of 9.97
7796366911207e-11 and in 78.77300000190735 seconds
      Klevel Aiyagari solved in 906 iterations, with a sup metric of 3.2704
494852638523e-7 and in 290.69799995422363 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (18.704531069048215, 0.0227294
07665029365, 0.003792148663392246, 1.8368801916421793, 11.609577503603969).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 782 iterations, with a sup metric of 9.881
517826215713e-11 and in 51.93500018119812 seconds
      Klevel Aiyagari solved in 406 iterations, with a sup metric of 1.9660
613281707947e-8 and in 145.56200003623962 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (2.432450676797521, 0.02272940
7665029365, 0.013260778164210806, 0.8813650592366683, 5.819714560085169).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 940 iterations, with a sup metric of 9.774
758780167758e-11 and in 62.068000078201294 seconds
      Klevel Aiyagari solved in 293 iterations, with a sup metric of 4.0009
11036433905e-7 and in 130.41499996185303 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (3.7973952901118477, 0.0227294
07665029365, 0.017995092914620085, 1.0346533414282149, 3.840362702485756).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1048 iterations, with a sup metric of 9.99
662574940885e-11 and in 69.74699997901917 seconds
      Klevel Aiyagari solved in 957 iterations, with a sup metric of 3.9112
828376770556e-7 and in 294.3369998931885 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (11.772748604934073, 0.0203622
50289824725, 0.017995092914620085, 1.5548768847375893, 4.414593766289061).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 991 iterations, with a sup metric of 9.838
529990702227e-11 and in 64.96900010108948 seconds
      Klevel Aiyagari solved in 1401 iterations, with a sup metric of 2.346
1025466697672e-7 and in 400.1489999294281 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (8.273377727171031, 0.01917867
1602222405, 0.017995092914620085, 1.3694488037971155, 0.7775588526257735).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 964 iterations, with a sup metric of 9.966
427683139045e-11 and in 64.98499989509583 seconds
      Klevel Aiyagari solved in 470 iterations, with a sup metric of 2.5252
84028251378e-7 and in 178.72199988365173 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (6.979454842191105, 0.01917867
1602222405, 0.018586882258421245, 1.2881181943744289, 0.5867877689485734).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 977 iterations, with a sup metric of 9.976
908188491507e-11 and in 66.50100016593933 seconds
      Klevel Aiyagari solved in 335 iterations, with a sup metric of 7.4816
46152063299e-7 and in 147.18200016021729 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (7.576998388594348, 0.01888277
6930321823, 0.018586882258421245, 1.326780146153624, 0.0461026473123729).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 971 iterations, with a sup metric of 9.852
918481101369e-11 and in 65.29200005531311 seconds
      Klevel Aiyagari solved in 461 iterations, with a sup metric of 5.5419
72870171465e-7 and in 176.1729998588562 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (7.341136234914033, 0.0188827
76930321823, 0.018734829594371533, 1.311761124949613, 0.20739901070384548).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 974 iterations, with a sup metric of 9.911
538256801578e-11 and in 65.6399998664856 seconds
      Klevel Aiyagari solved in 367 iterations, with a sup metric of 5.3184
39752815863e-7 and in 152.85199999809265 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (7.3951974137898535, 0.018882
776930321823, 0.018808803262346678, 1.315230561591377, 0.1445096196333715).
Equilibrium found in 2705.2809998989105 seconds
~~~~



~~~~{.julia}
θ[4] = 0.9;
AE42 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.3
903775074072655 and in 666.9969999790192 seconds
      Klevel Aiyagari solved in 222 iterations, with a sup metric of 0.0 an
d in 717.074000120163 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.99829009731561, 0.04166666
6666666484, -0.03411825715641328, 2.0390490982656275, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 579 iterations, with a sup metric of 9.700
862335648708e-11 and in 39.64199995994568 seconds
      Klevel Aiyagari solved in 197 iterations, with a sup metric of 1.9641
426277016603e-7 and in 85.90299987792969 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (8.448669432925634, 0.04166666
6666666484, 0.0037742047551266034, 1.3798242347299599, 1.3093750149793149).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1184 iterations, with a sup metric of 9.87
9386198008433e-11 and in 78.41100001335144 seconds
      Klevel Aiyagari solved in 320 iterations, with a sup metric of 4.9194
70326016466e-7 and in 153.1579999923706 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (17.352194124849515, 0.0227204
35710896544, 0.0037742047551266034, 1.787917837124756, 10.256272258264284).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 777 iterations, with a sup metric of 9.904
255193760036e-11 and in 51.90000009536743 seconds
      Klevel Aiyagari solved in 918 iterations, with a sup metric of 2.4366
65628251022e-7 and in 265.85700011253357 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (12.034825406375333, 0.0132473
20233011574, 0.0037742047551266034, 1.5672500771559446, 3.7807991646067123)
.
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 663 iterations, with a sup metric of 9.842
615611432848e-11 and in 44.15599989891052 seconds
      Klevel Aiyagari solved in 832 iterations, with a sup metric of 5.8338
18929233688e-7 and in 238.9559998512268 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (9.356030436804708, 0.00851076
249406909, 0.0037742047551266034, 1.431439375112057, 0.4015317068895623).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 618 iterations, with a sup metric of 9.817
924251365184e-11 and in 41.3529999256134 seconds
      Klevel Aiyagari solved in 248 iterations, with a sup metric of 9.3470
52175861303e-7 and in 99.96799993515015 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (8.79936962134411, 0.008510762
49406909, 0.006142483624597846, 1.4001756989046297, 0.5427513596441855).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 640 iterations, with a sup metric of 9.731
948580338212e-11 and in 42.34100008010864 seconds
      Klevel Aiyagari solved in 756 iterations, with a sup metric of 8.3392
28551134392e-7 and in 219.21099996566772 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (9.03815688222291, 0.008510762
49406909, 0.007326623059333468, 1.4137373409572904, 0.10678587780377669).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 651 iterations, with a sup metric of 9.880
096740744193e-11 and in 43.25 seconds
      Klevel Aiyagari solved in 387 iterations, with a sup metric of 4.0568
12973469458e-7 and in 137.010999917984 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (9.133710707553368, 0.00791869
2776701278, 0.007326623059333468, 1.4190999499221755, 0.08481156098014253).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 645 iterations, with a sup metric of 9.953
105006843543e-11 and in 43.49400019645691 seconds
      Klevel Aiyagari solved in 3364 iterations, with a sup metric of 5.897
817558673948e-7 and in 847.1670000553131 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (9.0553300864755, 0.0079186927
76701278, 0.007622657918017373, 1.4147037896916768, 0.04138299387585853).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 648 iterations, with a sup metric of 9.911
715892485518e-11 and in 44.04800009727478 seconds
      Klevel Aiyagari solved in 4044 iterations, with a sup metric of 4.588
566288511929e-7 and in 1014.0700001716614 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (9.05542929214635, 0.00791869
2776701278, 0.007770675347359325, 1.4147093692356862, 0.017325165221238947)
.
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 650 iterations, with a sup metric of 9.732
836758757912e-11 and in 48.46799993515015 seconds
      Klevel Aiyagari solved in 3632 iterations, with a sup metric of 2.837
94993782553e-7 and in 901.8250000476837 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (9.083593868676724, 0.0078446
84062030302, 0.007770675347359325, 1.41629182721986, 0.02277994197458888).
Equilibrium found in 4680.203999996185 seconds
~~~~



~~~~{.julia}
# Column 3
θ[2] = 5.0;
θ[4] = 0.0;
AE13 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.2
3896124740485902 and in 691.933000087738 seconds
      Klevel Aiyagari solved in 290 iterations, with a sup metric of 4.7000
27236802104e-7 and in 761.1530001163483 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.938401464414422, 0.0416666
66666666484, -0.03404777030655422, 2.037289159609131, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 579 iterations, with a sup metric of 9.870
504413811432e-11 and in 38.9760000705719 seconds
      Klevel Aiyagari solved in 911 iterations, with a sup metric of 8.1248
96876837872e-7 and in 260.94400000572205 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (0.8260775247691204, 0.0416666
66666666484, 0.003809448180056133, 0.5974581149444732, 8.925556059008656).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1171 iterations, with a sup metric of 9.85
540538067653e-11 and in 86.25699996948242 seconds
      Klevel Aiyagari solved in 3409 iterations, with a sup metric of 9.740
323247123205e-7 and in 916.1959998607635 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (22.51040198738746, 0.02273805
742336131, 0.003809448180056133, 1.9635318226824492, 15.416381744846827).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 773 iterations, with a sup metric of 9.847
056503531348e-11 and in 53.83200001716614 seconds
      Klevel Aiyagari solved in 189 iterations, with a sup metric of 5.2995
57478025389e-7 and in 99.98499989509583 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (1.200864511211642, 0.02273805
742336131, 0.013273752801708722, 0.6835933318863927, 7.049507203572069).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 928 iterations, with a sup metric of 9.896
083952298795e-11 and in 63.75599980354309 seconds
      Klevel Aiyagari solved in 285 iterations, with a sup metric of 8.3799
12944843008e-7 and in 131.79899978637695 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (1.5933615891832025, 0.0227380
5742336131, 0.018005905112535015, 0.7568557899822765, 6.043079863026769).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1034 iterations, with a sup metric of 9.87
9030926640553e-11 and in 70.21199989318848 seconds
      Klevel Aiyagari solved in 887 iterations, with a sup metric of 5.1088
78893485837e-7 and in 283.62099981307983 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (7.672954202078732, 0.02037198
126794816, 0.018005905112535015, 1.3328046895304562, 0.31591396873838296).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 978 iterations, with a sup metric of 9.801
404132758762e-11 and in 69.70499992370605 seconds
      Klevel Aiyagari solved in 718 iterations, with a sup metric of 5.8319
49413544227e-7 and in 258.87200021743774 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (3.813708972804537, 0.02037198
126794816, 0.019188943190241586, 1.0362513073828559, 3.6808970693425596).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1005 iterations, with a sup metric of 9.86
7662242868391e-11 and in 68.49900007247925 seconds
      Klevel Aiyagari solved in 487 iterations, with a sup metric of 6.2475
46503417515e-7 and in 193.83200001716614 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (4.797507227331336, 0.02037198
126794816, 0.019780462229094874, 1.1255002700555812, 2.6277934720542078).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1019 iterations, with a sup metric of 9.94
7243029273523e-11 and in 72.45900011062622 seconds
      Klevel Aiyagari solved in 976 iterations, with a sup metric of 6.8891
05453732632e-8 and in 315.3659999370575 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (6.807995202221349, 0.02037198
126794816, 0.020076221748521517, 1.2766354525521657, 0.583046029087332).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1027 iterations, with a sup metric of 9.80
7621381696663e-11 and in 83.97200012207031 seconds
      Klevel Aiyagari solved in 576 iterations, with a sup metric of 7.9140
02086628388e-7 and in 270.93300008773804 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (6.520752267817555, 0.0203719
8126794816, 0.020224101508234837, 1.2569764653655844, 0.8532563255369947).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1030 iterations, with a sup metric of 9.94
2269230123202e-11 and in 93.49799990653992 seconds
      Klevel Aiyagari solved in 998 iterations, with a sup metric of 7.2293
79372329205e-7 and in 352.38699984550476 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (7.601185820523923, 0.0202980
413880915, 0.020224101508234837, 1.3283033251437935, 0.2356694208196659).
Equilibrium found in 3845.0940001010895 seconds
~~~~



~~~~{.julia}
θ[4] = 0.3;
AE23 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.2
3887223437350258 and in 719.3739998340607 seconds
      Klevel Aiyagari solved in 431 iterations, with a sup metric of 3.9581
88478036949e-26 and in 826.404000043869 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.9766185085685, 0.041666666
666666484, -0.034092782450258166, 2.0384125502751336, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 579 iterations, with a sup metric of 9.649
3479873061e-11 and in 44.134000062942505 seconds
      Klevel Aiyagari solved in 626 iterations, with a sup metric of 4.4284
26898341836e-7 and in 214.46199989318848 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (1.1920540800324182, 0.0416666
66666666484, 0.003786942108204159, 0.6817835510143472, 8.563672615173672).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1170 iterations, with a sup metric of 9.86
7839878552331e-11 and in 82.87199997901917 seconds
      Klevel Aiyagari solved in 2107 iterations, with a sup metric of 8.905
170225724391e-7 and in 609.8939998149872 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (22.59740524417086, 0.02272680
4387435323, 0.003786942108204159, 1.966260526297893, 15.502170741441663).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 772 iterations, with a sup metric of 9.946
798940063672e-11 and in 54.01199984550476 seconds
      Klevel Aiyagari solved in 1168 iterations, with a sup metric of 8.077
67725435003e-7 and in 352.4689998626709 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (1.7403348647849082, 0.0227268
04387435323, 0.013256873247819741, 0.7812819473469677, 6.512370285297725).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 928 iterations, with a sup metric of 9.797
496147712081e-11 and in 66.61400008201599 seconds
      Klevel Aiyagari solved in 651 iterations, with a sup metric of 3.3188
67647099107e-7 and in 233.78200006484985 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (3.0626827987785763, 0.0227268
04387435323, 0.01799183881762753, 0.9575836031221878, 4.5754714997407335).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1034 iterations, with a sup metric of 9.84
0839254593448e-11 and in 73.97399997711182 seconds
      Klevel Aiyagari solved in 525 iterations, with a sup metric of 7.6199
08588650065e-7 and in 208.4430000782013 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (8.357037997668108, 0.02035932
1602531427, 0.01799183881762753, 1.3744179854302598, 0.998547647223134).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 978 iterations, with a sup metric of 9.818
457158417004e-11 and in 69.66999983787537 seconds
      Klevel Aiyagari solved in 696 iterations, with a sup metric of 4.0109
406310467127e-7 and in 247.6949999332428 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (6.07383922392334, 0.020359321
602531427, 0.01917558021007948, 1.2252557775362118, 1.4223447341729916).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1005 iterations, with a sup metric of 9.87
1747863599012e-11 and in 74.3510000705719 seconds
      Klevel Aiyagari solved in 966 iterations, with a sup metric of 6.8195
96162129556e-7 and in 321.7180001735687 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (7.607827335640128, 0.01976745
0906305453, 0.01917558021007948, 1.3287210248686043, 0.18101348418518004).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 991 iterations, with a sup metric of 9.903
544651024276e-11 and in 69.11100006103516 seconds
      Klevel Aiyagari solved in 850 iterations, with a sup metric of 7.8343
93626031744e-7 and in 280.51200008392334 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (6.69913444864286, 0.019767450
906305453, 0.019471515558192466, 1.2692486314797589, 0.7622322432694775).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 998 iterations, with a sup metric of 9.879
563833692373e-11 and in 69.93400001525879 seconds
      Klevel Aiyagari solved in 667 iterations, with a sup metric of 5.9084
05559811358e-7 and in 236.53600001335144 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (7.030137782273017, 0.0197674
50906305453, 0.01961948323224896, 1.291477829722098, 0.4139196110586729).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1001 iterations, with a sup metric of 9.97
6020010071807e-11 and in 77.14400005340576 seconds
      Klevel Aiyagari solved in 446 iterations, with a sup metric of 7.2044
58760112941e-7 and in 199.71399998664856 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (6.746936543755693, 0.0197674
50906305453, 0.019693467069277208, 1.272501664771578, 0.6884908807781338).
Equilibrium found in 3731.6370000839233 seconds
~~~~



~~~~{.julia}
θ[4] = 0.6;
AE33 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.2
3849623078513105 and in 722.4400000572205 seconds
      Klevel Aiyagari solved in 365 iterations, with a sup metric of 5.1990
87523698697e-7 and in 814.6080000400543 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.949879606503913, 0.0416666
66666666484, -0.034061301175096356, 2.0376266754858468, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 578 iterations, with a sup metric of 9.884
448815000724e-11 and in 40.75499987602234 seconds
      Klevel Aiyagari solved in 918 iterations, with a sup metric of 7.8198
43383523998e-9 and in 269.1010000705719 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (2.460496147319753, 0.04166666
6666666484, 0.003802682745785064, 0.8850099270384203, 7.292367549486593).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1169 iterations, with a sup metric of 9.89
7505037770316e-11 and in 82.7279999256134 seconds
      Klevel Aiyagari solved in 709 iterations, with a sup metric of 7.1023
87818025403e-7 and in 263.17599987983704 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (15.216960658945865, 0.0227346
74706225776, 0.003802682745785064, 1.7053676658225363, 8.122575439735199).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 772 iterations, with a sup metric of 9.969
003400556176e-11 and in 55.42700004577637 seconds
      Klevel Aiyagari solved in 204 iterations, with a sup metric of 8.2370
93455117719e-7 and in 108.42600011825562 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (3.476850876901925, 0.02273467
4706225776, 0.01326867872600542, 1.0023212943337219, 4.774222166086308).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 930 iterations, with a sup metric of 9.795
542155188741e-11 and in 66.10300016403198 seconds
      Klevel Aiyagari solved in 944 iterations, with a sup metric of 7.3865
98726917167e-7 and in 311.99000000953674 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (8.641855648100893, 0.01800167
67161156, 0.01326867872600542, 1.3911004680648529, 1.0048993722131776).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 843 iterations, with a sup metric of 9.939
959966231982e-11 and in 60.455000162124634 seconds
      Klevel Aiyagari solved in 446 iterations, with a sup metric of 6.2224
8626963096e-7 and in 181.3970000743866 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (4.424706203887039, 0.01800167
67161156, 0.01563517772106051, 1.0931968512553267, 3.509574282151025).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 884 iterations, with a sup metric of 9.934
453260029841e-11 and in 63.3030002117157 seconds
      Klevel Aiyagari solved in 965 iterations, with a sup metric of 9.4790
51902427178e-7 and in 308.73600006103516 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (7.225107200384473, 0.01800167
67161156, 0.016818427218588054, 1.304259234283754, 0.558183402203503).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 906 iterations, with a sup metric of 9.965
361869035405e-11 and in 64.80199980735779 seconds
      Klevel Aiyagari solved in 675 iterations, with a sup metric of 1.8868
745368485452e-7 and in 237.90399980545044 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (7.765762063441361, 0.01741005
1967351826, 0.016818427218588054, 1.3385858899236647, 0.05620798817103356).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 895 iterations, with a sup metric of 9.920
597676682519e-11 and in 68.46800017356873 seconds
      Klevel Aiyagari solved in 393 iterations, with a sup metric of 6.4933
46317719914e-7 and in 178.08000016212463 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (7.4839399803406925, 0.0174100
51967351826, 0.01711423959296994, 1.3208906940780136, 0.26233847233199903).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 901 iterations, with a sup metric of 9.819
700608204585e-11 and in 64.32099986076355 seconds
      Klevel Aiyagari solved in 761 iterations, with a sup metric of 5.2898
71717857983e-7 and in 279.7869999408722 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (7.871978619411285, 0.0172621
45780160883, 0.01711423959296994, 1.3451483402344233, 0.1440981321704724).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 898 iterations, with a sup metric of 9.867
484607184451e-11 and in 68.65300011634827 seconds
      Klevel Aiyagari solved in 199 iterations, with a sup metric of 5.4152
75529748832e-7 and in 124.49900007247925 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (7.087103106177452, 0.0172621
45780160883, 0.017188192686565414, 1.295235456065891, 0.6499673953481322).
Equilibrium found in 3077.7109999656677 seconds
~~~~



~~~~{.julia}
θ[4] = 0.9;
AE43 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.2
348562130555365 and in 727.2990000247955 seconds
      Klevel Aiyagari solved in 249 iterations, with a sup metric of 0.0 an
d in 789.058000087738 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.997038792169263, 0.0416666
66666666484, -0.03411678724678763, 2.0390123539580536, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 567 iterations, with a sup metric of 9.689
671287560486e-11 and in 40.32000017166138 seconds
      Klevel Aiyagari solved in 2233 iterations, with a sup metric of 7.833
987622019435e-7 and in 610.7360000610352 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (13.365453856245285, 0.0037749
39709939428, -0.03411678724678763, 1.627549139849646, 3.6075431689565054).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 376 iterations, with a sup metric of 9.546
852197672706e-11 and in 26.623000144958496 seconds
      Klevel Aiyagari solved in 1019 iterations, with a sup metric of 4.298
9668150124237e-7 and in 282.3680000305176 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (9.67740793148434, 0.003774939
709939428, -0.0151709237684241, 1.4489494399768648, 4.888323331578102).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 450 iterations, with a sup metric of 9.947
598300641403e-11 and in 32.032999992370605 seconds
      Klevel Aiyagari solved in 2939 iterations, with a sup metric of 9.985
946477611987e-7 and in 772.385999917984 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (10.472690612652203, 0.0037749
39709939428, -0.005697992029242336, 1.490736731377864, 1.2975217145270381).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 501 iterations, with a sup metric of 9.730
882766234572e-11 and in 34.819000005722046 seconds
      Klevel Aiyagari solved in 1181 iterations, with a sup metric of 8.027
835564301194e-7 and in 328.31699991226196 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (11.339604944492256, -0.000961
5261596514538, -0.005697992029242336, 1.5340348531079093, 0.652749632307118
7).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 474 iterations, with a sup metric of 9.864
820071925351e-11 and in 33.24399995803833 seconds
      Klevel Aiyagari solved in 1045 iterations, with a sup metric of 9.972
430447107865e-7 and in 299.6619999408722 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (10.816069562846659, -0.000961
5261596514538, -0.0033297590944468946, 1.5081515466218374, 0.39103022559812
75).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 487 iterations, with a sup metric of 9.842
793247116788e-11 and in 34.507999897003174 seconds
      Klevel Aiyagari solved in 668 iterations, with a sup metric of 6.9037
26240976701e-7 and in 203.60399985313416 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (11.11916066219177, -0.0021456
426270491744, -0.0033297590944468946, 1.523231442920208, 0.1772519226600231
).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 481 iterations, with a sup metric of 9.602
629802429874e-11 and in 33.25200009346008 seconds
      Klevel Aiyagari solved in 1063 iterations, with a sup metric of 9.171
954465014954e-7 and in 304.768000125885 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (10.993552519064158, -0.002145
6426270491744, -0.0027377008607480345, 1.517014297612988, 0.079649911904635
22).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 484 iterations, with a sup metric of 9.713
11919784057e-11 and in 34.092000007629395 seconds
      Klevel Aiyagari solved in 2970 iterations, with a sup metric of 8.754
343592443745e-7 and in 782.5220000743866 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (11.01641623290493, -0.0024416
717438986045, -0.0027377008607480345, 1.518149340044742, 0.0091816828709010
42).
Equilibrium found in 4373.427999973297 seconds
~~~~



~~~~{.julia}
#TABLE B
θ[5] = 0.4;
# Column 1
θ[2] = 1.0;
θ[4] = 0.0;
BE11 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.8
22026036295938 and in 221.56699991226196 seconds
      Klevel Aiyagari solved in 240 iterations, with a sup metric of 8.6341
40689384665e-7 and in 279.78600001335144 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.567373839393795, 0.0416666
66666666484, -0.03360481501256987, 2.026325073660492, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 604 iterations, with a sup metric of 9.846
345960795588e-11 and in 13.199999809265137 seconds
      Klevel Aiyagari solved in 1323 iterations, with a sup metric of 5.754
771151063896e-7 and in 351.13100004196167 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (0.8699371182303949, 0.0416666
66666666484, 0.004030925827048306, 0.6086892112720791, 8.841566759837137).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1222 iterations, with a sup metric of 9.98
2059623325767e-11 and in 30.61299991607666 seconds
      Klevel Aiyagari solved in 1520 iterations, with a sup metric of 7.888
107133874745e-7 and in 432.1909999847412 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (22.298051527940853, 0.0228487
96246857393, 0.004030925827048306, 1.956843361485565, 15.2159624156078).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 804 iterations, with a sup metric of 9.977
441095543327e-11 and in 17.555999994277954 seconds
      Klevel Aiyagari solved in 2469 iterations, with a sup metric of 3.572
068516614385e-7 and in 648.8199999332428 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (1.3627657593350366, 0.0228487
96246857393, 0.01343986103695285, 0.7154373525682871, 6.864700692933193).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 965 iterations, with a sup metric of 9.947
598300641403e-11 and in 21.60800004005432 seconds
      Klevel Aiyagari solved in 6439 iterations, with a sup metric of 9.849
338949581785e-7 and in 1679.2820000648499 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (2.2794058508049413, 0.0228487
96246857393, 0.01814432864190512, 0.8609853684087105, 5.340213377169185).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1075 iterations, with a sup metric of 9.96
9625125449966e-11 and in 22.644999980926514 seconds
      Klevel Aiyagari solved in 1787 iterations, with a sup metric of 9.405
05564616686e-7 and in 478.75300002098083 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (9.784680011832172, 0.02049656
2444381257, 0.01814432864190512, 1.4547111241338528, 2.441885121341361).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1016 iterations, with a sup metric of 9.98
0993809222127e-11 and in 22.249000072479248 seconds
      Klevel Aiyagari solved in 2304 iterations, with a sup metric of 7.738
39701012105e-7 and in 582.6419999599457 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (3.293197768148487, 0.02049656
2444381257, 0.01932044554314319, 0.98292953847873, 4.185909336927542).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1044 iterations, with a sup metric of 9.92
947946087952e-11 and in 21.5239999294281 seconds
      Klevel Aiyagari solved in 770 iterations, with a sup metric of 5.3572
58968395752e-7 and in 207.9340000152588 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (3.8493943893732663, 0.0204965
62444381257, 0.019908503993762223, 1.0397315946520025, 3.561042618985973).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1059 iterations, with a sup metric of 9.83
1779834712506e-11 and in 21.9760000705719 seconds
      Klevel Aiyagari solved in 8661 iterations, with a sup metric of 2.066
944585101354e-7 and in 2290.746999979019 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (4.207961592298615, 0.02049656
2444381257, 0.02020253321907174, 1.0736081759761964, 3.1685272024743796).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1067 iterations, with a sup metric of 9.85
522774499259e-11 and in 23.943000078201294 seconds
      Klevel Aiyagari solved in 6334 iterations, with a sup metric of 3.321
2391444727564e-7 and in 1699.6679999828339 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (5.388974951645753, 0.0204965
62444381257, 0.0203495478317265, 1.1736056547055926, 1.970635268198703).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1071 iterations, with a sup metric of 9.90
3544651024276e-11 and in 24.76900005340576 seconds
      Klevel Aiyagari solved in 2747 iterations, with a sup metric of 4.058
408478947639e-7 and in 722.9639999866486 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (7.536798968000741, 0.0204230
55138053876, 0.0203495478317265, 1.3242417295360969, 0.18560429791667765).
Equilibrium found in 9373.924000024796 seconds
~~~~



~~~~{.julia}
θ[4] = 0.3;
BE21 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.8
224149234592915 and in 217.51399993896484 seconds
      Klevel Aiyagari solved in 119 iterations, with a sup metric of 3.8144
846282636137e-7 and in 246.70299983024597 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.774774546106155, 0.0416666
66666666484, -0.03385376369166254, 2.0324668456718196, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 602 iterations, with a sup metric of 9.766
765174390457e-11 and in 13.229999780654907 seconds
      Klevel Aiyagari solved in 191 iterations, with a sup metric of 2.0002
158337897285e-7 and in 61.38099980354309 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (1.1293290789370716, 0.0416666
66666666484, 0.003906451487501971, 0.6686446251275103, 8.60469499245014).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1218 iterations, with a sup metric of 9.91
0650078381877e-11 and in 26.16600012779236 seconds
      Klevel Aiyagari solved in 3741 iterations, with a sup metric of 3.648
9962253370693e-7 and in 1020.9880001544952 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (21.431667115497508, 0.0227865
59077084226, 0.003906451487501971, 1.9291238832906983, 14.342876551856214).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 802 iterations, with a sup metric of 9.812
950452214864e-11 and in 19.117000102996826 seconds
      Klevel Aiyagari solved in 6304 iterations, with a sup metric of 9.741
330055376388e-7 and in 1648.0610001087189 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (1.8795422703504656, 0.0227865
59077084226, 0.013346505282293098, 0.803227796059438, 6.360784487822653).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 963 iterations, with a sup metric of 9.838
174719334347e-11 and in 22.371000051498413 seconds
      Klevel Aiyagari solved in 1638 iterations, with a sup metric of 9.178
189065698293e-7 and in 443.3769998550415 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (3.740061350227875, 0.02278655
9077084226, 0.018066532179688663, 1.0290022287693676, 3.8890047752014842).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1072 iterations, with a sup metric of 9.91
5979148900078e-11 and in 23.5239999294281 seconds
      Klevel Aiyagari solved in 8524 iterations, with a sup metric of 3.984
9216963052934e-8 and in 2218.7009999752045 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (10.911717065210643, 0.0204265
45628386444, 0.018066532179688663, 1.5129392459208637, 3.5609216145096942).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1014 iterations, with a sup metric of 9.90
7807907438837e-11 and in 21.95300006866455 seconds
      Klevel Aiyagari solved in 4098 iterations, with a sup metric of 3.041
7809677559894e-7 and in 1042.2060000896454 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (4.490534983885508, 0.02042654
5628386444, 0.019246538904037554, 1.0990242728450101, 2.9972763199809975).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1042 iterations, with a sup metric of 9.87
5478212961752e-11 and in 22.519999980926514 seconds
      Klevel Aiyagari solved in 3009 iterations, with a sup metric of 2.598
074838827718e-7 and in 771.4750001430511 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (5.224694327274539, 0.02042654
5628386444, 0.019836542266212, 1.160598201655088, 2.1940903250748347).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1056 iterations, with a sup metric of 9.98
8099236579728e-11 and in 23.01099991798401 seconds
      Klevel Aiyagari solved in 3633 iterations, with a sup metric of 1.894
1185662254779e-7 and in 978.3289999961853 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (5.475010206997087, 0.02042654
5628386444, 0.02013154394729922, 1.1803166946542485, 1.909651523812438).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1064 iterations, with a sup metric of 9.88
9333796309074e-11 and in 23.82100009918213 seconds
      Klevel Aiyagari solved in 493 iterations, with a sup metric of 4.1189
109393547675e-7 and in 150.9630000591278 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (6.886630487587181, 0.0204265
45628386444, 0.020279044787842832, 1.28192440348098, 0.4810661910574412).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1068 iterations, with a sup metric of 9.88
8623253573314e-11 and in 24.307000160217285 seconds
      Klevel Aiyagari solved in 474 iterations, with a sup metric of 8.2538
15975109977e-7 and in 145.92799997329712 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (8.419525648857514, 0.0203527
95208114637, 0.020279044787842832, 1.3781088405761974, 1.0602875413270931).
Equilibrium found in 8728.123000144958 seconds
~~~~



~~~~{.julia}
θ[4] = 0.6;
BE31 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.8
315887066764844 and in 219.49699997901917 seconds
      Klevel Aiyagari solved in 163 iterations, with a sup metric of 8.8773
85188731388e-7 and in 260.16100001335144 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.80828616304787, 0.04166666
6666666484, -0.03389366806819707, 2.033456135986914, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 602 iterations, with a sup metric of 9.839
595804805867e-11 and in 13.126999855041504 seconds
      Klevel Aiyagari solved in 7142 iterations, with a sup metric of 9.861
803842774155e-7 and in 1810.8259999752045 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (1.9761091328079659, 0.0416666
66666666484, 0.003886499299234708, 0.8178467027239664, 7.761532700264316).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1219 iterations, with a sup metric of 9.89
9281394609716e-11 and in 26.597000122070312 seconds
      Klevel Aiyagari solved in 2620 iterations, with a sup metric of 5.993
263632495659e-7 and in 694.364000082016 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (19.360131629545386, 0.0227765
82982950596, 0.003886499299234708, 1.859803086198517, 12.270265912606863).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 803 iterations, with a sup metric of 9.947
598300641403e-11 and in 18.341000080108643 seconds
      Klevel Aiyagari solved in 4110 iterations, with a sup metric of 3.603
4257344395413e-7 and in 1062.454999923706 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (3.6155036820541078, 0.0227765
82982950596, 0.013331541141092652, 1.0165312751567874, 4.626887540540473).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 966 iterations, with a sup metric of 9.946
887757905643e-11 and in 21.18500018119812 seconds
      Klevel Aiyagari solved in 484 iterations, with a sup metric of 7.8105
43393253551e-7 and in 140.95800018310547 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (6.816523149022751, 0.02277658
2982950596, 0.018054062062021623, 1.2772109198420967, 0.8140590207357734).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1076 iterations, with a sup metric of 9.96
9625125449966e-11 and in 23.591000080108643 seconds
      Klevel Aiyagari solved in 7352 iterations, with a sup metric of 6.483
605570217823e-7 and in 1872.9449999332428 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (9.473048849216488, 0.02041532
252248611, 0.018054062062021623, 1.4378589784501454, 2.120969646647211).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1018 iterations, with a sup metric of 9.94
9019386112923e-11 and in 23.062999963760376 seconds
      Klevel Aiyagari solved in 1203 iterations, with a sup metric of 9.076
250132652251e-7 and in 334.3489999771118 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (7.760254744783448, 0.01923469
2292253865, 0.018054062062021623, 1.3382440652136023, 0.27104668625994943).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 992 iterations, with a sup metric of 9.794
831612452981e-11 and in 22.98099994659424 seconds
      Klevel Aiyagari solved in 918 iterations, with a sup metric of 7.7243
71696061309e-7 and in 263.460000038147 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (7.281993571066196, 0.01923469
2292253865, 0.018644377177137746, 1.3079467974241832, 0.2773595637224364).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1005 iterations, with a sup metric of 9.83
3556191551907e-11 and in 22.977999925613403 seconds
      Klevel Aiyagari solved in 4171 iterations, with a sup metric of 9.519
944906581739e-7 and in 1083.4179999828339 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (7.600543953728312, 0.01893953
4734695805, 0.018644377177137746, 1.3282629443309666, 0.07639741252173238).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 998 iterations, with a sup metric of 9.913
847520692798e-11 and in 22.27300000190735 seconds
      Klevel Aiyagari solved in 8396 iterations, with a sup metric of 4.678
9157327952254e-8 and in 2158.2249999046326 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (7.346995370860821, 0.0189395
34734695805, 0.018791955955916775, 1.3121379299553457, 0.1947207748835682).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1001 iterations, with a sup metric of 9.98
0993809222127e-11 and in 23.811999797821045 seconds
      Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.02
1662724534175115 and in 2570.164999961853 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (7.496570681579772, 0.0189395
34734695805, 0.01886574534530629, 1.3216928007947875, 0.03635226122618462).
Equilibrium found in 12251.333999872208 seconds
~~~~



~~~~{.julia}
θ[4] = 0.9;
BE41 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.9
262386991595122 and in 225.9940001964569 seconds
      Klevel Aiyagari solved in 152 iterations, with a sup metric of 0.0 an
d in 265.0460000038147 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.99945971839301, 0.04166666
6666666484, -0.034119631002563044, 2.039083442873658, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 604 iterations, with a sup metric of 9.678
302603788325e-11 and in 13.728000164031982 seconds
      Klevel Aiyagari solved in 3962 iterations, with a sup metric of 1.800
858686211171e-7 and in 1024.7480001449585 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (6.885137452076789, 0.04166666
6666666484, 0.0037735178320517197, 1.2818243439634356, 2.873032017462547).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1226 iterations, with a sup metric of 9.90
3544651024276e-11 and in 26.180000066757202 seconds
      Klevel Aiyagari solved in 3542 iterations, with a sup metric of 2.745
2281682382043e-7 and in 892.9909999370575 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (17.14909937892345, 0.02272009
22493591, 0.0037735178320517197, 1.7803559820243744, 10.05314043983017).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 809 iterations, with a sup metric of 9.835
332548391307e-11 and in 17.224000215530396 seconds
      Klevel Aiyagari solved in 5213 iterations, with a sup metric of 7.232
949406257224e-7 and in 1297.0780000686646 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (9.984796683717672, 0.01324680
504070541, 0.0037735178320517197, 1.4653524518296097, 1.7306991858645162).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 691 iterations, with a sup metric of 9.924
86093309708e-11 and in 15.223000049591064 seconds
      Klevel Aiyagari solved in 2261 iterations, with a sup metric of 4.257
3353160520693e-7 and in 576.4990000724792 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (8.581486791344995, 0.01324680
504070541, 0.008510161436378564, 1.3875942380963602, 0.37310695203595223).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 745 iterations, with a sup metric of 9.993
783578465809e-11 and in 15.55299997329712 seconds
      Klevel Aiyagari solved in 7344 iterations, with a sup metric of 6.360
413849888443e-7 and in 1810.1059999465942 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (9.241290357155155, 0.01087848
3238541986, 0.008510161436378564, 1.4250946696446412, 0.6486381168646069).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 717 iterations, with a sup metric of 9.954
703727999004e-11 and in 15.026000022888184 seconds
      Klevel Aiyagari solved in 1743 iterations, with a sup metric of 7.451
540170254271e-7 and in 435.26199984550476 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (8.8558591319444, 0.0096943223
37460275, 0.008510161436378564, 1.4034050206620154, 0.08529722512396631).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 704 iterations, with a sup metric of 9.864
464800557471e-11 and in 14.687999963760376 seconds
      Klevel Aiyagari solved in 7197 iterations, with a sup metric of 3.441
927912422639e-7 and in 1743.042000055313 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (8.701917094301734, 0.00969432
2337460275, 0.00910224188691942, 1.3945733217046767, 0.15987732689600698).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 711 iterations, with a sup metric of 9.735
146022649133e-11 and in 15.757000207901001 seconds
      Klevel Aiyagari solved in 2394 iterations, with a sup metric of 8.508
991227972875e-7 and in 588.9170000553131 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (8.628198638111101, 0.00969432
2337460275, 0.009398282112189849, 1.3903086427596816, 0.18778598431704197).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 714 iterations, with a sup metric of 9.839
595804805867e-11 and in 14.247000217437744 seconds
      Klevel Aiyagari solved in 1800 iterations, with a sup metric of 5.569
014586624682e-7 and in 438.7020001411438 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (8.931522441306917, 0.0095463
02224825061, 0.009398282112189849, 1.4077098574001536, 0.13829727721171103)
.
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 712 iterations, with a sup metric of 9.941
203416019562e-11 and in 14.505000114440918 seconds
      Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.00
3385001905648678 and in 2405.1100001335144 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (8.796912335428413, 0.0095463
02224825061, 0.009472292168507455, 1.4000349231120965, 0.007680497260697194
).
Equilibrium found in 11477.515000104904 seconds
~~~~



~~~~{.julia}
# Column 2
θ[2] = 3.0;
θ[4] = 0.0;
BE12 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.3
9911539398963214 and in 668.1039998531342 seconds
      Klevel Aiyagari solved in 3464 iterations, with a sup metric of 9.361
849495734544e-7 and in 1502.9139997959137 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.253193082390265, 0.0416666
66666666484, -0.03322105863458137, 2.016957683778331, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 596 iterations, with a sup metric of 9.719
15881109453e-11 and in 40.78699994087219 seconds
      Klevel Aiyagari solved in 2726 iterations, with a sup metric of 9.742
385813312265e-7 and in 693.4099998474121 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (1.9921276086587858, 0.0416666
66666666484, 0.004222804016042556, 0.8202271669185937, 7.684828184947092).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1207 iterations, with a sup metric of 9.95
2216828423843e-11 and in 79.19400000572205 seconds
      Klevel Aiyagari solved in 341 iterations, with a sup metric of 7.5018
05745160768e-7 and in 159.5 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (21.275504681928833, 0.0229447
35341354522, 0.004222804016042556, 1.9240516556507052, 14.203725578096892).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 795 iterations, with a sup metric of 9.934
097988661961e-11 and in 52.34400010108948 seconds
      Klevel Aiyagari solved in 678 iterations, with a sup metric of 7.5430
32907909548e-7 and in 211.28600001335144 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (3.0441291900726757, 0.0229447
35341354522, 0.013583769678698539, 0.9554911802690914, 5.163577369896421).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 956 iterations, with a sup metric of 9.822
009872095805e-11 and in 62.467000007629395 seconds
      Klevel Aiyagari solved in 259 iterations, with a sup metric of 3.7753
214106823677e-7 and in 122.54999995231628 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (4.060127022669038, 0.02294473
5341354522, 0.01826425251002653, 1.059873997515671, 3.544967268638966).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1068 iterations, with a sup metric of 9.82
9648206505226e-11 and in 72.45900011062622 seconds
      Klevel Aiyagari solved in 1497 iterations, with a sup metric of 3.421
8142917529623e-7 and in 431.4910001754761 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (18.964453032854255, 0.0206044
93925690526, 0.01826425251002653, 1.8460288743938877, 11.63396312764264).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1007 iterations, with a sup metric of 9.87
7609841169033e-11 and in 65.94899988174438 seconds
      Klevel Aiyagari solved in 2233 iterations, with a sup metric of 7.149
620460732876e-7 and in 590.0879998207092 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (4.864810462654112, 0.02060449
3925690526, 0.019434373217858528, 1.1311591340950748, 2.6009115160231824).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1036 iterations, with a sup metric of 9.96
7493497242685e-11 and in 69.67899990081787 seconds
      Klevel Aiyagari solved in 1712 iterations, with a sup metric of 8.098
630249991892e-7 and in 480.2410001754761 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (9.54436410788321, 0.020019433
571774527, 0.019434373217858528, 1.4417464594760987, 2.146764920173955).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1021 iterations, with a sup metric of 9.98
0283266486367e-11 and in 68.46900010108948 seconds
      Klevel Aiyagari solved in 3719 iterations, with a sup metric of 7.797
219749028524e-7 and in 964.2590000629425 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (6.545356057638318, 0.02001943
3571774527, 0.019726903394816527, 1.258681802884082, 0.8861765125680234).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1029 iterations, with a sup metric of 9.85
416193088895e-11 and in 67.60800004005432 seconds
      Klevel Aiyagari solved in 908 iterations, with a sup metric of 5.1890
87587546333e-7 and in 292.6119999885559 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (6.736499004024039, 0.0200194
33571774527, 0.019873168483295525, 1.2717926300576057, 0.6780350386234035).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1032 iterations, with a sup metric of 9.99
627047804097e-11 and in 73.23900008201599 seconds
      Klevel Aiyagari solved in 2774 iterations, with a sup metric of 4.599
5206859416335e-7 and in 726.7360000610352 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (8.452105661038464, 0.0199463
01027535026, 0.019873168483295525, 1.3800262403365804, 1.0460469841925448).
Equilibrium found in 6175.09500002861 seconds
~~~~



~~~~{.julia}
θ[4] = 0.3;
BE22 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.3
974028358097712 and in 664.5429999828339 seconds
      Klevel Aiyagari solved in 2243 iterations, with a sup metric of 8.788
398196867841e-7 and in 1205.8259999752045 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.105262560167823, 0.0416666
66666666484, -0.033037532627030936, 2.012520189331083, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 597 iterations, with a sup metric of 9.702
105785436288e-11 and in 39.23399996757507 seconds
      Klevel Aiyagari solved in 4911 iterations, with a sup metric of 1.931
004701652937e-7 and in 1219.3710000514984 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (2.8195004550122795, 0.0416666
66666666484, 0.004314567019817774, 0.9294840781960563, 6.841004395006156).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1210 iterations, with a sup metric of 9.86
6596428764751e-11 and in 78.5239999294281 seconds
      Klevel Aiyagari solved in 3195 iterations, with a sup metric of 8.038
534029870583e-7 and in 836.4289999008179 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (20.664347899995565, 0.0229906
16843242127, 0.004314567019817774, 1.903968626170466, 13.597490712839479).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 797 iterations, with a sup metric of 9.932
67690319044e-11 and in 51.89900016784668 seconds
      Klevel Aiyagari solved in 2026 iterations, with a sup metric of 6.313
212908075197e-7 and in 530.1780002117157 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (4.354839279317453, 0.02299061
6843242127, 0.01365259193152995, 1.0869509371295107, 3.8434448890248216).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 959 iterations, with a sup metric of 9.811
351731059403e-11 and in 63.08800005912781 seconds
      Klevel Aiyagari solved in 3799 iterations, with a sup metric of 5.432
444325621079e-7 and in 980.7709999084473 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (5.7258160264929465, 0.0229906
16843242127, 0.018321604387386038, 1.1995032646501351, 1.872347964562124).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1070 iterations, with a sup metric of 9.97
8506909646967e-11 and in 72.26799988746643 seconds
      Klevel Aiyagari solved in 1005 iterations, with a sup metric of 1.012
081388980181e-7 and in 316.66999983787537 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (17.77506950083131, 0.02065611
0615314083, 0.018321604387386038, 1.803482965783612, 10.450452330338024).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1010 iterations, with a sup metric of 9.98
2770166061528e-11 and in 67.26200008392334 seconds
      Klevel Aiyagari solved in 2962 iterations, with a sup metric of 4.692
473094392841e-7 and in 763.7000000476837 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (10.201357703630178, 0.0194888
5750135006, 0.018321604387386038, 1.476715540785751, 2.742023089991995).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 984 iterations, with a sup metric of 9.798
029054763902e-11 and in 64.36800003051758 seconds
      Klevel Aiyagari solved in 1607 iterations, with a sup metric of 7.257
883114733735e-7 and in 437.88499999046326 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (6.021028111048443, 0.01948885
750135006, 0.01890523094436805, 1.2214098266844544, 1.5071963856292205).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 997 iterations, with a sup metric of 9.813
305723582744e-11 and in 64.74600005149841 seconds
      Klevel Aiyagari solved in 3533 iterations, with a sup metric of 5.434
9546840086e-7 and in 901.7410001754761 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (7.328634547139262, 0.01948885
750135006, 0.019197044222859054, 1.310956488975022, 0.16501518101893975).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1003 iterations, with a sup metric of 9.99
0230864787009e-11 and in 68.08699989318848 seconds
      Klevel Aiyagari solved in 9181 iterations, with a sup metric of 9.660
69593367546e-7 and in 2292.2679998874664 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (8.839923117096175, 0.0193429
5086210456, 0.019197044222859054, 1.4024953486993588, 1.3634632331595755).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1000 iterations, with a sup metric of 9.89
2886509987875e-11 and in 70.18799996376038 seconds
      Klevel Aiyagari solved in 3739 iterations, with a sup metric of 7.832
330204310122e-7 and in 956.3289999961853 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (7.899167895320797, 0.0192699
97542481807, 0.019197044222859054, 1.346819073228042, 0.41412118214838767).
Equilibrium found in 10441.174999952316 seconds
~~~~



~~~~{.julia}
θ[4] = 0.6;
BE32 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.3
9433607637101886 and in 671.8860001564026 seconds
      Klevel Aiyagari solved in 1013 iterations, with a sup metric of 2.953
0376530365803e-7 and in 914.8610000610352 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.075172666058823, 0.0416666
66666666484, -0.03299997618253464, 2.011615446603299, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 596 iterations, with a sup metric of 9.906
031550599437e-11 and in 40.10899996757507 seconds
      Klevel Aiyagari solved in 1989 iterations, with a sup metric of 6.494
524988916149e-7 and in 510.69799995422363 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (5.629436514938243, 0.04166666
6666666484, 0.004333345242065922, 1.1921951405107392, 4.027707495397005).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1210 iterations, with a sup metric of 9.94
7953572009283e-11 and in 79.9909999370575 seconds
      Klevel Aiyagari solved in 3283 iterations, with a sup metric of 1.576
2696063057418e-7 and in 847.2669999599457 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (18.804632181575503, 0.0230000
059543662, 0.004333345242065922, 1.8404131067069682, 11.738781514481115).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 799 iterations, with a sup metric of 9.756
817576089816e-11 and in 52.33999991416931 seconds
      Klevel Aiyagari solved in 1332 iterations, with a sup metric of 9.012
587990368179e-7 and in 369.05200004577637 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (8.327864003117885, 0.01366667
5598216061, 0.004333345242065922, 1.3726887638003984, 0.13150583023470652).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 683 iterations, with a sup metric of 9.776
535137007158e-11 and in 44.63100004196167 seconds
      Klevel Aiyagari solved in 4026 iterations, with a sup metric of 9.908
15458072432e-7 and in 1004.6619999408722 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (7.125690047558909, 0.01366667
5598216061, 0.00900001042014099, 1.2977698128157638, 1.752014604801869).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 736 iterations, with a sup metric of 9.908
163178806717e-11 and in 49.24000000953674 seconds
      Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.01
0378098836341033 and in 2460.1790001392365 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (7.714551277970898, 0.01366667
5598216061, 0.011333343009178525, 1.3354013631661212, 0.811330212128933).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 766 iterations, with a sup metric of 9.894
840502511215e-11 and in 44.43400001525879 seconds
      Klevel Aiyagari solved in 4628 iterations, with a sup metric of 4.330
7004837612224e-7 and in 1017.3269999027252 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (8.147625428117996, 0.01366667
5598216061, 0.012500009303697292, 1.3619186080963313, 0.21083194166234875).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 782 iterations, with a sup metric of 9.869
971506759612e-11 and in 43.71599984169006 seconds
      Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.00
024968214135840647 and in 2145.093999862671 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (8.177059241320409, 0.01366667
5598216061, 0.013083342450956676, 1.3636877708013007, 0.09969776142072995).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 790 iterations, with a sup metric of 9.925
3938401489e-11 and in 47.21199989318848 seconds
      Klevel Aiyagari solved in 2148 iterations, with a sup metric of 8.408
79029755953e-7 and in 493.67399978637695 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (8.325502674211885, 0.01337500
9024586368, 0.013083342450956676, 1.372548631952662, 0.08910596893669798).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 786 iterations, with a sup metric of 9.891
820695884235e-11 and in 44.264999866485596 seconds
      Klevel Aiyagari solved in 133 iterations, with a sup metric of 5.7530
60667120987e-7 and in 70.5569999217987 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (8.161363929466562, 0.0133750
09024586368, 0.013229175737771522, 1.3627448891126515, 0.09517247972380183)
.
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 788 iterations, with a sup metric of 9.907
097364703077e-11 and in 45.05000019073486 seconds
      Klevel Aiyagari solved in 5865 iterations, with a sup metric of 9.086
800410344529e-7 and in 1255.8940000534058 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (8.171229497648023, 0.0133750
09024586368, 0.013302092381178944, 1.3633376895362586, 0.07522697652814969)
.
Equilibrium found in 11089.27999997139 seconds
~~~~



~~~~{.julia}
θ[4] = 0.9;
BE42 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.3
7369815575993925 and in 576.454999923706 seconds
      Klevel Aiyagari solved in 206 iterations, with a sup metric of 0.0 an
d in 620.2200000286102 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.997689938751982, 0.0416666
66666666484, -0.034117552164502074, 2.0390314748848697, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 541 iterations, with a sup metric of 9.976
020010071807e-11 and in 30.945000171661377 seconds
      Klevel Aiyagari solved in 1585 iterations, with a sup metric of 8.244
354554939472e-7 and in 367.7539999485016 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (15.478748999443113, 0.0037745
57251082205, -0.034117552164502074, 1.7158719819642396, 5.720768705621101).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 345 iterations, with a sup metric of 9.687
539659353206e-11 and in 20.317999839782715 seconds
      Klevel Aiyagari solved in 1097 iterations, with a sup metric of 5.969
925645573923e-7 and in 253.7079999446869 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (13.060431074925939, 0.0037745
57251082205, -0.015171497456709935, 1.6140785948845642, 1.505501589957026).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 391 iterations, with a sup metric of 9.654
854693508241e-11 and in 22.263999938964844 seconds
      Klevel Aiyagari solved in 2118 iterations, with a sup metric of 6.705
658907955571e-7 and in 461.4249999523163 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (14.484363226790318, -0.005698
470102813865, -0.015171497456709935, 1.6753430970313887, 2.7140325676426027
).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 360 iterations, with a sup metric of 9.562
484137859428e-11 and in 20.659000158309937 seconds
      Klevel Aiyagari solved in 1346 iterations, with a sup metric of 3.740
646018315619e-7 and in 296.4979999065399 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (14.399363634458341, -0.010434
9837797619, -0.015171497456709935, 1.6717970776427986, 1.3530777602317006).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 358 iterations, with a sup metric of 9.924
86093309708e-11 and in 20.067999839782715 seconds
      Klevel Aiyagari solved in 369 iterations, with a sup metric of 5.7888
92112107197e-7 and in 95.96199989318848 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (13.746539003227145, -0.010434
9837797619, -0.012803240618235918, 1.6441051385158916, 0.02526688749633088)
.
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 360 iterations, with a sup metric of 9.711
698112369049e-11 and in 20.725000143051147 seconds
      Klevel Aiyagari solved in 8081 iterations, with a sup metric of 3.349
229598849335e-7 and in 1709.9690001010895 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (14.1361447828354, -0.01161911
2198998909, -0.012803240618235918, 1.660730372649608, 0.7351470418290909).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 359 iterations, with a sup metric of 9.939
071787812281e-11 and in 20.73799991607666 seconds
      Klevel Aiyagari solved in 9567 iterations, with a sup metric of 8.332
214909107516e-7 and in 1985.851000070572 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (14.129831492958841, -0.012211
176408617413, -0.012803240618235918, 1.6604633251889493, 0.5455043915474675
).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 359 iterations, with a sup metric of 9.771
383702172898e-11 and in 20.001999855041504 seconds
      Klevel Aiyagari solved in 3120 iterations, with a sup metric of 1.629
2566565605223e-7 and in 683.4949998855591 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (13.994743570573297, -0.012507
208513426665, -0.012803240618235918, 1.6547308137932755, 0.3172038627201363
4).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 359 iterations, with a sup metric of 9.606
537787476555e-11 and in 19.627999782562256 seconds
      Klevel Aiyagari solved in 4041 iterations, with a sup metric of 8.696
908641344985e-7 and in 876.8849999904633 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (13.729897542261291, -0.01265
5224565831291, -0.012803240618235918, 1.6433883377024703, 0.005357471526719
948).
Equilibrium found in 7351.773000001907 seconds
~~~~



~~~~{.julia}
# Column 3
θ[2] = 5.0;
θ[4] = 0.0;
BE13 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.2
3917769197896632 and in 561.2580001354218 seconds
      Klevel Aiyagari solved in 301 iterations, with a sup metric of 8.8804
73137476549e-7 and in 625.0080001354218 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.012389028401095, 0.0416666
66666666484, -0.03292136488654712, 2.009725335591777, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 589 iterations, with a sup metric of 9.799
272504551482e-11 and in 35.42799997329712 seconds
      Klevel Aiyagari solved in 1188 iterations, with a sup metric of 9.628
410141919195e-7 and in 279.87999987602234 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (2.700025155111052, 0.04166666
6666666484, 0.004372650890059681, 0.9151080709570352, 6.95009031572262).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1192 iterations, with a sup metric of 9.87
9386198008433e-11 and in 66.1949999332428 seconds
      Klevel Aiyagari solved in 712 iterations, with a sup metric of 7.7608
29804935597e-7 and in 212.32299995422363 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (21.634781120859557, 0.0230196
5877836308, 0.004372650890059681, 1.9356858532269374, 14.571036490953606).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 787 iterations, with a sup metric of 9.954
703727999004e-11 and in 44.89900016784668 seconds
      Klevel Aiyagari solved in 8965 iterations, with a sup metric of 9.556
802916373154e-7 and in 1929.49799990654 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (3.9228802664611706, 0.0230196
5877836308, 0.01369615483421138, 1.0468339359254932, 4.269448909122209).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 947 iterations, with a sup metric of 9.813
660994950624e-11 and in 52.68599987030029 seconds
      Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.00
011604657419313833 and in 2173.4329998493195 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (5.034656352303594, 0.02301965
877836308, 0.01835790680628723, 1.1452205004922325, 2.559126269176886).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1057 iterations, with a sup metric of 9.80
5845024857263e-11 and in 63.83099985122681 seconds
      Klevel Aiyagari solved in 1890 iterations, with a sup metric of 1.693
562437559408e-7 and in 563.936999797821 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (19.741995905832233, 0.0206887
82792325156, 0.01835790680628723, 1.8729265869748246, 12.421092054692107).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 999 iterations, with a sup metric of 9.828
049485349766e-11 and in 73.80999994277954 seconds
      Klevel Aiyagari solved in 1393 iterations, with a sup metric of 8.935
621901100308e-7 and in 424.30099987983704 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (11.502131864781564, 0.0195233
44799306196, 0.01835790680628723, 1.5419140905030924, 4.046835675852914).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 972 iterations, with a sup metric of 9.880
984919163893e-11 and in 69.07599997520447 seconds
      Klevel Aiyagari solved in 1751 iterations, with a sup metric of 7.935
731065398985e-7 and in 501.15499997138977 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (6.95476031775678, 0.019523344
799306196, 0.018940625802796714, 1.2864756021233543, 0.5692565792384698).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 985 iterations, with a sup metric of 9.919
354226894939e-11 and in 69.15699982643127 seconds
      Klevel Aiyagari solved in 6808 iterations, with a sup metric of 7.155
080027944831e-7 and in 1802.106999874115 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (9.53206843118878, 0.019231985
301051455, 0.018940625802796714, 1.4410775367428383, 2.042541149486376).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 979 iterations, with a sup metric of 9.788
08145646326e-11 and in 70.5569999217987 seconds
      Klevel Aiyagari solved in 1793 iterations, with a sup metric of 3.688
7029212446037e-7 and in 548.2690000534058 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (7.990624366757607, 0.0190863
05551924084, 0.018940625802796714, 1.352412049409855, 0.4838847620163813).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 975 iterations, with a sup metric of 9.934
453260029841e-11 and in 69.26300001144409 seconds
      Klevel Aiyagari solved in 2720 iterations, with a sup metric of 1.702
2490038968956e-7 and in 795.7150001525879 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (7.68963343284065, 0.01901346
5677360397, 0.018940625802796714, 1.3338469603522316, 0.17426332441022474).
Equilibrium found in 9855.634999990463 seconds
~~~~



~~~~{.julia}
θ[4] = 0.3;
BE23 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.2
3856594147082433 and in 713.579999923706 seconds
      Klevel Aiyagari solved in 2944 iterations, with a sup metric of 5.517
708068567102e-7 and in 1486.1789999008179 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (23.820299053602138, 0.0416666
66666666484, -0.03267874143917303, 2.00392272090974, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 591 iterations, with a sup metric of 9.643
84128110396e-11 and in 42.617000102996826 seconds
      Klevel Aiyagari solved in 8012 iterations, with a sup metric of 6.888
210744111715e-7 and in 2097.558000087738 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (4.505719564402889, 0.04166666
6666666484, 0.004493962613746728, 1.1003606996464592, 5.122756059235976).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1195 iterations, with a sup metric of 9.98
6145244056388e-11 and in 85.43600010871887 seconds
      Klevel Aiyagari solved in 2033 iterations, with a sup metric of 5.994
17586510177e-7 and in 632.3830001354218 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (20.38669868224152, 0.02308031
4640206608, 0.004493962613746728, 1.894719214868923, 13.329447572179703).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 790 iterations, with a sup metric of 9.826
450764194306e-11 and in 55.92799997329712 seconds
      Klevel Aiyagari solved in 8285 iterations, with a sup metric of 4.074
0801882131187e-7 and in 2235.1749999523163 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (5.723814558605477, 0.02308031
4640206608, 0.013787138626976668, 1.1993523040001897, 2.456100105225504).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 951 iterations, with a sup metric of 9.821
832236411865e-11 and in 70.39800000190735 seconds
      Klevel Aiyagari solved in 2073 iterations, with a sup metric of 5.036
255189459678e-7 and in 608.2820000648499 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (13.433187173261016, 0.0184337
2663359164, 0.013787138626976668, 1.630513645810407, 5.848541958271727).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 863 iterations, with a sup metric of 9.758
771568613156e-11 and in 63.06100010871887 seconds
      Klevel Aiyagari solved in 4927 iterations, with a sup metric of 2.239
805719792203e-8 and in 1367.380000114441 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (6.276231162456988, 0.01843372
663359164, 0.016110432630284152, 1.2397998774977956, 1.5968313706685215).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 904 iterations, with a sup metric of 9.904
432829443977e-11 and in 63.88199996948242 seconds
      Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.00
019741961125689447 and in 12409.96700000763 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (6.631315193593016, 0.01843372
663359164, 0.017272079631937894, 1.2646077980344954, 1.0953321961307028).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 927 iterations, with a sup metric of 9.787
90382077932e-11 and in 99.68400001525879 seconds
      Klevel Aiyagari solved in 328 iterations, with a sup metric of 3.6426
31919071217e-7 and in 218.76799988746643 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (8.117600070440432, 0.01785290
3132764768, 0.017272079631937894, 1.3601096701722264, 0.46249373454862575).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 915 iterations, with a sup metric of 9.915
446241848258e-11 and in 93.80999994277954 seconds
      Klevel Aiyagari solved in 267 iterations, with a sup metric of 5.8858
31017845271e-7 and in 188.2389998435974 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (7.119877084905165, 0.01785290
3132764768, 0.01756249138235133, 1.2973885853505789, 0.5708633541863444).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 921 iterations, with a sup metric of 9.839
595804805867e-11 and in 92.8199999332428 seconds
      Klevel Aiyagari solved in 2097 iterations, with a sup metric of 8.569
759254425941e-7 and in 821.3909997940063 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (7.965242999075432, 0.0177076
9725755805, 0.01756249138235133, 1.3508639893089287, 0.29235353698297395).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 918 iterations, with a sup metric of 9.872
991313386592e-11 and in 94.84800004959106 seconds
      Klevel Aiyagari solved in 4302 iterations, with a sup metric of 5.688
213359350313e-7 and in 1619.9690001010895 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (7.371164677103539, 0.0177076
9725755805, 0.01763509431995469, 1.3136902447900423, 0.3106417696681092).
Equilibrium found in 23685.30499982834 seconds
~~~~



~~~~{.julia}
θ[4] = 0.6;
BE33 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.2
3709711080255147 and in 1114.6979999542236 seconds
      Klevel Aiyagari solved in 302 iterations, with a sup metric of 6.3134
17946377705e-7 and in 1325.7250001430511 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (23.782938586504955, 0.0416666
66666666484, -0.03263117941688375, 2.0027906681660896, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 590 iterations, with a sup metric of 9.868
106332078241e-11 and in 43.21500015258789 seconds
      Klevel Aiyagari solved in 742 iterations, with a sup metric of 5.0607
56631929449e-7 and in 233.25099992752075 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (8.964946963571423, 0.04166666
6666666484, 0.0045177436248913655, 1.4096041018948793, 0.6592958808870737).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 1196 iterations, with a sup metric of 9.89
7149766402435e-11 and in 84.35300016403198 seconds
      Klevel Aiyagari solved in 441 iterations, with a sup metric of 8.4432
8552351218e-7 and in 198.72100019454956 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (19.812177869940157, 0.0230922
05145778925, 0.0045177436248913655, 1.8753208085844992, 12.756198551617349)
.
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 791 iterations, with a sup metric of 9.762
501917975897e-11 and in 55.47000002861023 seconds
      Klevel Aiyagari solved in 508 iterations, with a sup metric of 1.3488
68648159379e-7 and in 185.13899993896484 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (11.204701796633634, 0.0138049
74385335145, 0.0045177436248913655, 1.5274397305611498, 3.027217160837692).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 676 iterations, with a sup metric of 9.818
279522733064e-11 and in 46.32099986076355 seconds
      Klevel Aiyagari solved in 8925 iterations, with a sup metric of 5.064
919356546092e-7 and in 2358.651999950409 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (9.766349706888018, 0.00916135
9005113254, 0.0045177436248913655, 1.4537294601403608, 0.9137343258644535).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 630 iterations, with a sup metric of 9.880
452012112073e-11 and in 63.776000022888184 seconds
      Klevel Aiyagari solved in 8148 iterations, with a sup metric of 3.300
916629186218e-7 and in 2987.2450001239777 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (9.381559503390045, 0.00683955
131500231, 0.0045177436248913655, 1.4328442574789242, 0.15634542755421776).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 609 iterations, with a sup metric of 9.996
62574940885e-11 and in 46.325000047683716 seconds
      Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.00
2523618503539276 and in 2948.4720001220703 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (9.241085956319285, 0.00683955
131500231, 0.005678647469946838, 1.4250833221668355, 0.18017897805979644).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 620 iterations, with a sup metric of 9.709
832937687679e-11 and in 46.681999921798706 seconds
      Klevel Aiyagari solved in 9151 iterations, with a sup metric of 9.081
066926847269e-7 and in 2558.0230000019073 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (9.221622804123498, 0.00683955
131500231, 0.006259099392474574, 1.4240020727979834, 0.10077155704410856).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 625 iterations, with a sup metric of 9.780
354304211869e-11 and in 43.223000049591064 seconds
      Klevel Aiyagari solved in 7418 iterations, with a sup metric of 2.815
933041719877e-7 and in 2082.49799990654 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (9.304152518803821, 0.00654932
5353738442, 0.006259099392474574, 1.4285769204074448, 0.03055706343635123).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 622 iterations, with a sup metric of 9.910
827714065817e-11 and in 46.7189998626709 seconds
      Klevel Aiyagari solved in 5044 iterations, with a sup metric of 8.813
067142211989e-7 and in 1320.3789999485016 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (9.341738429596788, 0.0064042
12373106509, 0.006259099392474574, 1.4306518045415573, 0.043796024448894855
).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 621 iterations, with a sup metric of 9.809
131285010153e-11 and in 42.134000062942505 seconds
      Klevel Aiyagari solved in 2732 iterations, with a sup metric of 3.233
1840647360804e-7 and in 725.3470001220703 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (9.185688930696879, 0.0064042
12373106509, 0.0063316558827905415, 1.4220019707242508, 0.12446628737690091
).
Equilibrium found in 16923.462000131607 seconds
~~~~



~~~~{.julia}
θ[4] = 0.9;
BE43 = EquilibriumAiyagari(u, du, θ);
~~~~~~~~~~~~~


~~~~
Equilibrium(Aiyagari)...
In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666
484, 1.1781242629578268).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.2
2389021877461346 and in 685.6499998569489 seconds
      Klevel Aiyagari solved in 198 iterations, with a sup metric of 4.3393
225000541273e-7 and in 734.481999874115 seconds
In iteration 1 we have (K, r1, r2, w, sup) = (24.999434101429824, 0.0416666
66666666484, -0.03411960091380499, 2.039082690671348, Inf).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 582 iterations, with a sup metric of 9.913
492249324918e-11 and in 39.193000078201294 seconds
      Klevel Aiyagari solved in 4753 iterations, with a sup metric of 4.049
49764341183e-7 and in 1191.5100002288818 seconds
In iteration 2 we have (K, r1, r2, w, sup) = (19.639232063004172, 0.0037735
32876430747, -0.03411960091380499, 1.8694110035923979, 9.881065331605758).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 404 iterations, with a sup metric of 9.822
542779147625e-11 and in 27.39299988746643 seconds
      Klevel Aiyagari solved in 3057 iterations, with a sup metric of 9.591
90094672102e-7 and in 788.1459999084473 seconds
In iteration 3 we have (K, r1, r2, w, sup) = (17.219726716261363, -0.015173
034018687121, -0.03411960091380499, 1.7829921277803247, 2.65325459590359).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 346 iterations, with a sup metric of 9.458
744898438454e-11 and in 23.825000047683716 seconds
      Klevel Aiyagari solved in 274 iterations, with a sup metric of 7.7818
20756561271e-7 and in 92.53600001335144 seconds
In iteration 4 we have (K, r1, r2, w, sup) = (16.56868905017283, -0.0151730
34018687121, -0.024646317466246057, 1.7584244229130161, 2.0760462512575195)
.
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 373 iterations, with a sup metric of 9.913
492249324918e-11 and in 25.93400001525879 seconds
      Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.00
835570777277536 and in 2492.045000076294 seconds
In iteration 5 we have (K, r1, r2, w, sup) = (16.76871552764843, -0.0199096
7574246659, -0.024646317466246057, 1.7660374180304197, 0.3688399972912002).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 359 iterations, with a sup metric of 9.731
593308970332e-11 and in 24.579999923706055 seconds
      Klevel Aiyagari solved in 2372 iterations, with a sup metric of 8.143
44716454096e-7 and in 615.1929998397827 seconds
In iteration 6 we have (K, r1, r2, w, sup) = (16.64867621508576, -0.0199096
7574246659, -0.022277996604356323, 1.7614757495462785, 0.8146403272960292).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 366 iterations, with a sup metric of 9.777
068044058979e-11 and in 25.42899990081787 seconds
      Klevel Aiyagari solved in 435 iterations, with a sup metric of 8.7770
88333620606e-7 and in 135.47599983215332 seconds
In iteration 7 we have (K, r1, r2, w, sup) = (16.613011171024283, -0.019909
67574246659, -0.021093836173411457, 1.7601163717041572, 0.3048906301909753)
.
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 370 iterations, with a sup metric of 9.504
2196335271e-11 and in 25.399999856948853 seconds
      Klevel Aiyagari solved in 381 iterations, with a sup metric of 2.9249
864888896407e-7 and in 121.48199987411499 seconds
In iteration 8 we have (K, r1, r2, w, sup) = (16.600508282927176, -0.019909
67574246659, -0.020501755957939023, 1.7596393803663275, 0.05507801927809908
).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 372 iterations, with a sup metric of 9.413
270163349807e-11 and in 25.54199981689453 seconds
      Klevel Aiyagari solved in 424 iterations, with a sup metric of 6.7929
60576950391e-7 and in 133.12299990653992 seconds
In iteration 9 we have (K, r1, r2, w, sup) = (16.690091197432906, -0.020205
715850202807, -0.020501755957939023, 1.7630519504228341, 0.1631713206603322
4).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 371 iterations, with a sup metric of 9.549
694368615746e-11 and in 25.572999954223633 seconds
      Klevel Aiyagari solved in 708 iterations, with a sup metric of 7.7344
80846197306e-7 and in 202.82299995422363 seconds
In iteration 10 we have (K, r1, r2, w, sup) = (16.789269248666255, -0.02035
3735904070913, -0.020501755957939023, 1.76681639173806, 0.19822071223823556
).
   Klevel(Aiyagari)...
   Kpolicy(Aiyagari)...
      Kpolicy Aiyagari solved in 370 iterations, with a sup metric of 9.822
542779147625e-11 and in 25.788000106811523 seconds
      Klevel Aiyagari solved in 9052 iterations, with a sup metric of 5.077
009573027683e-7 and in 2255.0859999656677 seconds
In iteration 11 we have (K, r1, r2, w, sup) = (16.668194477494875, -0.02042
7745931004968, -0.020501755957939023, 1.7622189017336813, 0.044928422815623
04).
Equilibrium found in 8761.91200017929 seconds
~~~~



~~~~{.julia}
TABLEA = [(AE11[4],AE11[2],AE11[1]) (AE12[4],AE12[2],AE12[1]) (AE13[4],AE13[2],AE13[1]); (AE21[4],AE21[2],AE21[1]) (AE22[4],AE22[2],AE22[1]) (AE23[4],AE23[2],AE23[1]); (AE31[4],AE31[2],AE31[1]) (AE32[4],AE32[2],AE32[1]) (AE33[4],AE33[2],AE33[1]); (AE41[4],AE41[2],AE41[1]) (AE42[4],AE42[2],AE42[1]) (AE43[4],AE43[2],AE43[1])];
TABLEB = [(BE11[4],BE11[2],BE11[1]) (BE12[4],BE12[2],BE12[1]) (BE13[4],BE13[2],BE13[1]); (BE21[4],BE21[2],BE21[1]) (BE22[4],BE22[2],BE22[1]) (BE23[4],BE23[2],BE23[1]); (BE31[4],BE31[2],BE31[1]) (BE32[4],BE32[2],BE32[1]) (BE33[4],BE33[2],BE33[1]); (BE41[4],BE41[2],BE41[1]) (BE42[4],BE42[2],BE42[1]) (BE43[4],BE43[2],BE43[1])];
~~~~~~~~~~~~~




In the following graphs we see the values from our estimation for Table II.A and II.B of Aiyagari in the following order: savings, interest rate, capital

- As the persistency in the shock increases (an increase in $\lambda$) there is a higher level of savings in the economy, a higher level of capital and a lower interest rate.
- As the risk aversion coefficient $\mu$ increases we have more savings, a higher level of capital and lower interest rates.
- As the variance of income shock increases there is an increase in the level of savings, the stationary level of capital, and a reduction in the level of interest rates.

~~~~{.julia}
@show TABLEA[:,1]
~~~~~~~~~~~~~


~~~~
TABLEA[:, 1] = Tuple{Float64,Float64,Float64}[(0.28635738531582133, 0.02057
362399868507, 7.621879051301215), (0.28635738531582133, 0.02057362399868507
, 8.60819618705661), (0.2876273501704559, 0.020129559942517042, 7.499163642
696772), (0.29504207317910247, 0.017613196957564872, 7.558415738249893)]
4-element Array{Tuple{Float64,Float64,Float64},1}:
 (0.28635738531582133, 0.02057362399868507, 7.621879051301215) 
 (0.28635738531582133, 0.02057362399868507, 8.60819618705661)  
 (0.2876273501704559, 0.020129559942517042, 7.499163642696772) 
 (0.29504207317910247, 0.017613196957564872, 7.558415738249893)
~~~~



~~~~{.julia}
@show TABLEA[:,2]
~~~~~~~~~~~~~


~~~~
TABLEA[:, 2] = Tuple{Float64,Float64,Float64}[(0.2863435385161505, 0.020578
48746733848, 8.494109717061006), (0.2876091033181216, 0.020135912485859685,
 7.256470947933149), (0.29147200501491033, 0.018808803262346678, 7.39519741
37898535), (0.3278513698070025, 0.007844684062030302, 9.083593868676724)]
4-element Array{Tuple{Float64,Float64,Float64},1}:
 (0.2863435385161505, 0.02057848746733848, 8.494109717061006)   
 (0.2876091033181216, 0.020135912485859685, 7.256470947933149)  
 (0.29147200501491033, 0.018808803262346678, 7.3951974137898535)
 (0.3278513698070025, 0.007844684062030302, 9.083593868676724)
~~~~



~~~~{.julia}
@show TABLEA[:,3]
~~~~~~~~~~~~~


~~~~
TABLEA[:, 3] = Tuple{Float64,Float64,Float64}[(0.2871441914659308, 0.020298
0413880915, 7.601185820523923), (0.288885529279334, 0.019693467069277208, 6
.746936543755693), (0.2963322930891491, 0.017188192686565414, 7.08710310617
7452), (0.37133342927275315, -0.0024416717438986045, 11.01641623290493)]
4-element Array{Tuple{Float64,Float64,Float64},1}:
 (0.2871441914659308, 0.0202980413880915, 7.601185820523923)     
 (0.288885529279334, 0.019693467069277208, 6.746936543755693)    
 (0.2963322930891491, 0.017188192686565414, 7.087103106177452)   
 (0.37133342927275315, -0.0024416717438986045, 11.01641623290493)
~~~~



~~~~{.julia}
@show TABLEB[:,1]
~~~~~~~~~~~~~


~~~~
TABLEB[:, 1] = Tuple{Float64,Float64,Float64}[(0.2867867339866127, 0.020423
055138053876, 7.536798968000741), (0.28698752177528986, 0.02035279520811463
7, 8.419525648857514), (0.2913041306613413, 0.01886574534530629, 7.49657068
1579772), (0.32188736090229564, 0.009472292168507455, 8.796912335428413)]
4-element Array{Tuple{Float64,Float64,Float64},1}:
 (0.2867867339866127, 0.020423055138053876, 7.536798968000741) 
 (0.28698752177528986, 0.020352795208114637, 8.419525648857514)
 (0.2913041306613413, 0.01886574534530629, 7.496570681579772)  
 (0.32188736090229564, 0.009472292168507455, 8.796912335428413)
~~~~



~~~~{.julia}
@show TABLEB[:,2]
~~~~~~~~~~~~~


~~~~
TABLEB[:, 2] = Tuple{Float64,Float64,Float64}[(0.2881547361324122, 0.019946
301027535026, 8.452105661038464), (0.2901178675629086, 0.019269997542481807
, 7.899167895320797), (0.3086747495687415, 0.013302092381178944, 8.17122949
7648023), (0.427650100758785, -0.012655224565831291, 13.729897542261291)]
4-element Array{Tuple{Float64,Float64,Float64},1}:
 (0.2881547361324122, 0.019946301027535026, 8.452105661038464) 
 (0.2901178675629086, 0.019269997542481807, 7.899167895320797) 
 (0.3086747495687415, 0.013302092381178944, 8.171229497648023) 
 (0.427650100758785, -0.012655224565831291, 13.729897542261291)
~~~~



~~~~{.julia}
@show TABLEB[:,3]
~~~~~~~~~~~~~


~~~~
TABLEB[:, 3] = Tuple{Float64,Float64,Float64}[(0.2908695277251079, 0.019013
465677360397, 7.68963343284065), (0.2949759018578, 0.01763509431995469, 7.3
71164677103539), (0.33359721536096504, 0.0063316558827905415, 9.18568893069
6879), (0.4834465381592677, -0.020427745931004968, 16.668194477494875)]
4-element Array{Tuple{Float64,Float64,Float64},1}:
 (0.2908695277251079, 0.019013465677360397, 7.68963343284065)   
 (0.2949759018578, 0.01763509431995469, 7.371164677103539)      
 (0.33359721536096504, 0.0063316558827905415, 9.185688930696879)
 (0.4834465381592677, -0.020427745931004968, 16.668194477494875)
~~~~




The values do not match properly Aiyagari's estimation. 


As the risk aversion parameter $\sigma$ increases see that there is a lowe concentration of agents at the borrowing constraint. Agents accumulate more assets. 

~~~~{.julia}
plot(ash(vec(AE41[15]); m =20); hist = false, label="sigma = 1", title = "Change of Asset PDF as sigma increases")
plot!(ash(vec(AE42[15]); m =20); hist = false, label="sigma = 3")
plot!(ash(vec(AE43[15]); m =20); hist = false, label="sigma = 5")
~~~~~~~~~~~~~




This is also reflected in the CDF

~~~~{.julia}
plot(AE41[7], [sum(AE41[5][:,1:s]) for s in 1: size(AE41[7])[1]], legend=:bottomright, label="sigma = 1", title = "Change of Asset CDF as sigma increases")
plot!(AE42[7], [sum(AE42[5][:,1:s]) for s in 1: size(AE42[7])[1]], label="sigma = 3")
plot!(AE43[7], [sum(AE43[5][:,1:s]) for s in 1: size(AE43[7])[1]], label="sigma = 5")
~~~~~~~~~~~~~




As the persistency of the shock increases there is going to be a higher concentration of agents at the level of the borrowing constraint and a higher number of agents with high levels of welath

~~~~{.julia}
plot(ash(vec(AE12[15]); m =20); hist = false, label="lambda = 0.0", title = "Change of Asset PDF as persistency of productivity shock increases")
plot!(ash(vec(AE22[15]); m =20); hist = false, label="lambda = 0.3",)
plot!(ash(vec(AE32[15]); m =20); hist = false, label="lambda = 0.6",)
plot!(ash(vec(AE42[15]); m =20); hist = false, label="lambda = 0.9",)
~~~~~~~~~~~~~



~~~~{.julia}
plot(AE12[7], [sum(AE12[5][:,1:s]) for s in 1: size(AE12[7])[1]], legend=:bottomright, label="lambda = 0.0", title = "Change of Asset CDF as mu increases")
plot!(AE22[7], [sum(AE22[5][:,1:s]) for s in 1: size(AE22[7])[1]], label="lambda = 0.3")
plot!(AE32[7], [sum(AE32[5][:,1:s]) for s in 1: size(AE32[7])[1]], label="lambda = 0.6")
plot!(AE42[7], [sum(AE42[5][:,1:s]) for s in 1: size(AE42[7])[1]], label="lambda = 0.9")
~~~~~~~~~~~~~




As the variance of the income shock increases we have that the number of agents with high levels of wealth is going to increase

~~~~{.julia}
plot(ash(vec(AE32[15]); m =20); hist = false, label="sigma_epsilon = 0.2", title = "Change of Asset PDF as variance of income shock changes")
plot!(ash(vec(BE32[15]); m =20); hist = false, label="sigma_epsilon = 0.4",)
~~~~~~~~~~~~~



~~~~{.julia}
plot(AE32[7], [sum(AE32[5][:,1:s]) for s in 1: size(AE32[7])[1]], legend=:bottomright, label="sigma_epsilon = 0.2", title = "Change of Asset CDF as sigma_epsilon increases")
plot!(BE32[7], [sum(BE32[5][:,1:s]) for s in 1: size(BE32[7])[1]], label="sigma_epsilon = 0.4")
~~~~~~~~~~~~~


