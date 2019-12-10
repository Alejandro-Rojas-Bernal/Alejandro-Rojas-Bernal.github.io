# __Aiyagari - Uninsured Idiosyncratic Risk and Aggregate Saving - The Quarterly Journal of Economic - (1994)__

# Replication made by Alejandro Rojas-Bernal. Commentaries or suggestions to `alejandro.rojas@alumni.ubc.ca`

## __1. The model__

### __1.1 Consumer Problem__ 

The individual problem is to maximize

<img src="https://render.githubusercontent.com/render/math?math=E_0\left\{\sum_{t=0}^{\infty}\beta^t U(C_t)\right\}">
subject to 
<img src="https://render.githubusercontent.com/render/math?math=c_t + a_{t+1} = w y_t + (1 + r) a_t">
<img src="https://render.githubusercontent.com/render/math?math=c_t \ge 0">
<img src="https://render.githubusercontent.com/render/math?math= a_t > - b">
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


```julia
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
```

## __3. Model Computation__

### __3.1 Environment Set-up__


```julia
using LinearAlgebra, Statistics, Distributions, Expectations, NLsolve, Roots, Random, Plots, Parameters, BenchmarkTools, ProgressMeter, LaTeXStrings, Profile, BenchmarkTools, Roots, NLsolve, ForwardDiff, KernelDensity, AverageShiftedHistograms
```

    ┌ Info: Recompiling stale cache file C:\Users\laroj\.julia\compiled\v1.2\Distributions\xILW0.ji for Distributions [31c24e10-a181-5473-b8eb-7969acd0382f]
    └ @ Base loading.jl:1240
    ┌ Info: Recompiling stale cache file C:\Users\laroj\.julia\compiled\v1.2\Expectations\r1e2K.ji for Expectations [2fe49d83-0758-5602-8f54-1f90ad0d522b]
    └ @ Base loading.jl:1240
    ┌ Info: Recompiling stale cache file C:\Users\laroj\.julia\compiled\v1.2\NLsolve\KFCNP.ji for NLsolve [2774e3e8-f4cf-5e23-947b-6d7e65073b56]
    └ @ Base loading.jl:1240
    ┌ Info: Recompiling stale cache file C:\Users\laroj\.julia\compiled\v1.2\Roots\o0Xsi.ji for Roots [f2b01f46-fcfa-551c-844a-d8ac1e96c665]
    └ @ Base loading.jl:1240
    ┌ Info: Recompiling stale cache file C:\Users\laroj\.julia\compiled\v1.2\Plots\ld3vC.ji for Plots [91a5bcdd-55d7-5caf-9e0b-520d859cae80]
    └ @ Base loading.jl:1240
    ┌ Info: Recompiling stale cache file C:\Users\laroj\.julia\compiled\v1.2\BenchmarkTools\ZXPQo.ji for BenchmarkTools [6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf]
    └ @ Base loading.jl:1240
    ┌ Info: Recompiling stale cache file C:\Users\laroj\.julia\compiled\v1.2\ProgressMeter\3V8n6.ji for ProgressMeter [92933f4c-e287-5a05-a399-4b506db050ca]
    └ @ Base loading.jl:1240
    ┌ Info: Recompiling stale cache file C:\Users\laroj\.julia\compiled\v1.2\LaTeXStrings\H4HGh.ji for LaTeXStrings [b964fa9f-0449-5b57-a5c2-d3ea65f4040f]
    └ @ Base loading.jl:1240
    ┌ Info: Precompiling KernelDensity [5ab0869b-81aa-558d-bb23-cbf5423bbe9b]
    └ @ Base loading.jl:1242
    ┌ Info: Precompiling AverageShiftedHistograms [77b51b56-6f8f-5c3a-9cb4-d71f9594ea6e]
    └ @ Base loading.jl:1242
    

### __3.2 Tauchen Method function__

Given a $\lambda$, $\sigma_\epsilon$ and a number of markov states the following function produces a vector $Y$ with the log productivity endowment and a matrix $M$ with the markov process.

__Steps__

1. Define $\sigma_y = \left(\frac{\sigma_\epsilon^2}{1-\lambda^2}\right)$
2. Establish the max and the min values $y_N = m \times \sigma_y$ and $y_1 = -Y_N$ (following Tauchen(1986) $m=3$).
3. Define $P\left(y_1 \lvert y_j\right) = \Phi\left(\frac{y_1 -\lambda y_j + \frac{w}{2}}{\sigma_\epsilon}\right)$, $P\left(y_N \lvert y_j\right) = 1-\Phi\left(\frac{y_N -\lambda y_j + \frac{w}{2}}{\sigma_\epsilon}\right)$, and $P\left(y_k \lvert y_j\right) = \Phi\left(\frac{y_k -\lambda y_j + \frac{w}{2}}{\sigma_\epsilon}\right) -  \Phi\left(\frac{y_k -\lambda y_j - \frac{w}{2}}{\sigma_\epsilon}\right)$.


```julia
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
```




    TauchenAR1 (generic function with 1 method)



For example with $\lambda = 0.5$, $\sigma_\epsilon=1$, and $N=5$ we obtain:

The log productivity process:


```julia
@show TauchenAR1(0.5, 1, 5)[1]
```

    (TauchenAR1(0.5, 1, 5))[1] = [-3.4641016151377544, -1.7320508075688772, 0.0, 1.7320508075688772, 3.4641016151377544]
    




    5-element Array{Float64,1}:
     -3.4641016151377544
     -1.7320508075688772
      0.0               
      1.7320508075688772
      3.4641016151377544



The Markov process:


```julia
@show TauchenAR1(0.5, 1, 5)[2]
```

    (TauchenAR1(0.5, 1, 5))[2] = [0.1932381153856163 0.6135237692287674 0.18855073115589882 0.00467993306182124 7.451167896244115e-6; 0.04163225833177518 0.4583677416682248 0.4583677416682248 0.04136625557920559 0.00026600275256960515; 0.004687384229717438 0.18855073115589888 0.6135237692287674 0.18855073115589882 0.004687384229717484; 0.0002660027525696245 0.041366255579205556 0.4583677416682248 0.4583677416682248 0.041632258331775196; 7.451167896213828e-6 0.004679933061821224 0.18855073115589888 0.6135237692287674 0.1932381153856163]
    




    5×5 Array{Float64,2}:
     0.193238     0.613524    0.188551  0.00467993  7.45117e-6 
     0.0416323    0.458368    0.458368  0.0413663   0.000266003
     0.00468738   0.188551    0.613524  0.188551    0.00468738 
     0.000266003  0.0413663   0.458368  0.458368    0.0416323  
     7.45117e-6   0.00467993  0.188551  0.613524    0.193238   



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


```julia
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
```




    KpolicyAiyagari (generic function with 1 method)




```julia
VFI = KpolicyAiyagari(u, du, Kr(0.015, δ, α), θ);
```

       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 849 iterations, with a sup metric of 9.912071163853398e-11 and in 18.919999837875366 seconds
    

#### __3.3.1 Policy Function__

In general, as the productivity shock is bigger households are going to save more


```julia
plot(VFI[2],[VFI[1][s,:] for s in 1:N], title ="Policy Function",xlabel="Capital at t", ylabel="Capital at t+1")
```




![svg](Aiyagari%281994%29_files/Aiyagari%281994%29_36_0.svg)



#### __3.3.2 Value Function__

Concavity of value function and monotonicity with respect to the productivity shock


```julia
plot(VFI[2],[VFI[3][s,:] for s in 1:N], title ="Value Function",xlabel="Capital at t", ylabel="Value Function")
```




![svg](Aiyagari%281994%29_files/Aiyagari%281994%29_39_0.svg)



#### __3.3.3 Non-Uniform grid__


```julia
plot(collect(1:1:size(VFI[2])[1]),VFI[2], title ="Capital Grid",xlabel="Position", ylabel="Value on Grid")
```




![svg](Aiyagari%281994%29_files/Aiyagari%281994%29_41_0.svg)



### __3.4 Convergence to Stationary Distribution__

Using the results from `KpolicyAiyagari` we follow the following algorithm:

1. Randomly assign an initial combination of state variables $\left\{y_i,a_i\right\}_{i=1}^{pop}$ to each of the pop households. In our simulations we are going to use $pop = 10000$. From this initial assignment we define an initial probability mass function for the state combination $f_0\left(y_i,a_i\right)$.
2. Now we iterate with idiosyncratic shocks at least 100 times until $sup\lvert E_{n+1}\left[a\right]-E_{n}\left[a\right], Var_{n+1}\left[a\right]-Var_{n}\left[a\right], Skw_{n+1}\left[a\right]-Skw_{n}\left[a\right] \lvert< 1E^{-6}$.


```julia
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
```




    KlevelAiyagari (generic function with 1 method)




```julia
KS = KlevelAiyagari(u, du, Kr(0.015, δ, α), θ, pop);
```

       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 849 iterations, with a sup metric of 9.912071163853398e-11 and in 16.39299988746643 seconds
          Klevel Aiyagari solved in 863 iterations, with a sup metric of 3.985122598923412e-7 and in 154.03199982643127 seconds
    

#### __3.4.1 Unconditional CDF for asset holdings__

The unconditional distribution for asset holdings is:


```julia
plot(KS[5], [sum(KS[3][:,1:s]) for s in 1: size(KS[5])[1]])
```




![svg](Aiyagari%281994%29_files/Aiyagari%281994%29_48_0.svg)




```julia
a = sum(KS[3][:,1]) * 100
println("The probability mass function show us that $a% of the households are hand-to-mouth consumers at the borrowing constraint")
```

    The probability mass function show us that 9.23% of the households are hand-to-mouth consumers at the borrowing constraint
    

#### __3.4.2 Unconditional PDF for asset holdings__

The density function for asset holdings is given by


```julia
plot(ash(vec(KS[10]); m =20); hist = false)
```




![svg](Aiyagari%281994%29_files/Aiyagari%281994%29_52_0.svg)



#### __3.4.3 Conditional CDF for asset holdings__

The conditional distribution for asset holdings given idiosyncratic productivity is:


```julia
plot(KS[5],[sum(KS[3][p,1:s])/sum(KS[3][p,:]) for s in 1:size(KS[5])[1], p in 1:size(KS[8])[1]], legend=:bottomright)
```




![svg](Aiyagari%281994%29_files/Aiyagari%281994%29_55_0.svg)



We can see that as the idiosyncratic productivity shock increases the probability of being a hand-to-mouth consumer diminishes.  

#### __3.4.4 Conditional PDF for asset holdings__

The conditional PDF for asset holdings are


```julia
plot(ash(vec(KS[10][KS[11] .== KS[8][1]]); m=30),  label = "y1" ; hist = false)
plot!(ash(vec(KS[10][KS[11] .== KS[8][3]]); m=30),  label = "y3"  ; hist = false)
plot!(ash(vec(KS[10][KS[11] .== KS[8][5]]); m=30),  label = "y5"  ; hist = false)
plot!(ash(vec(KS[10][KS[11] .== KS[8][7]]); m=30),  label = "y7"  ; hist = false)
```




![svg](Aiyagari%281994%29_files/Aiyagari%281994%29_59_0.svg)



Again, as productivity increases the probability of being a hand-to-mouth consumer decreases

#### __3.4.5 Supply and Demand in Bewley models__

Let's look at the behaviour of the demand and supply function under $\sigma =5$, $\lambda = 0.6$, and $\sigma_\epsilon=0.2$


```julia
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

plot([Supply1 Demand1], r_range)
```

       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 525 iterations, with a sup metric of 9.838174719334347e-11 and in 28.5 seconds
          Klevel Aiyagari solved in 1043 iterations, with a sup metric of 7.789821441754257e-7 and in 192.4100000858307 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (1, 0.0, 2.208041498118361, 10.48683784717663)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 536 iterations, with a sup metric of 9.906653275493227e-11 and in 29.7810001373291 seconds
          Klevel Aiyagari solved in 573 iterations, with a sup metric of 1.695323149304264e-7 and in 119.64400005340576 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (2, 0.0008620689655172414, 2.333170515892558, 10.312674983554945)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 548 iterations, with a sup metric of 9.773781783906088e-11 and in 33.47200012207031 seconds
          Klevel Aiyagari solved in 953 iterations, with a sup metric of 2.314010633929421e-7 and in 179.25699996948242 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (3, 0.0017241379310344827, 2.349848749492232, 10.143205831695491)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 560 iterations, with a sup metric of 9.840039894015717e-11 and in 31.73199987411499 seconds
          Klevel Aiyagari solved in 134 iterations, with a sup metric of 8.564402001325823e-7 and in 52.365999937057495 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (4, 0.002586206896551724, 2.4066713279994043, 9.978256362290475)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 573 iterations, with a sup metric of 9.738609918485963e-11 and in 31.08400011062622 seconds
          Klevel Aiyagari solved in 255 iterations, with a sup metric of 6.575872672138022e-7 and in 70.20900011062622 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (5, 0.0034482758620689655, 2.4940016154210602, 9.817660724606434)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 586 iterations, with a sup metric of 9.852119120523639e-11 and in 31.683000087738037 seconds
          Klevel Aiyagari solved in 863 iterations, with a sup metric of 6.440588713382164e-7 and in 165.5920000076294 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (6, 0.004310344827586207, 2.5004806706830025, 9.661260782666488)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 600 iterations, with a sup metric of 9.831602199028566e-11 and in 32.56500005722046 seconds
          Klevel Aiyagari solved in 664 iterations, with a sup metric of 9.23856192936945e-7 and in 135.89800000190735 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (7, 0.005172413793103448, 2.6083079511559526, 9.508905682151791)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 615 iterations, with a sup metric of 9.701661696226438e-11 and in 34.36400008201599 seconds
          Klevel Aiyagari solved in 516 iterations, with a sup metric of 3.1133917330857294e-7 and in 114.95500016212463 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (8, 0.00603448275862069, 2.6214185555958647, 9.360451445700878)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 630 iterations, with a sup metric of 9.819967061730495e-11 and in 42.03999996185303 seconds
          Klevel Aiyagari solved in 1728 iterations, with a sup metric of 7.692311980930955e-7 and in 343.3199999332428 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (9, 0.006896551724137931, 2.7620383862273106, 9.215760594482523)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 646 iterations, with a sup metric of 9.862866079402011e-11 and in 37.45300006866455 seconds
          Klevel Aiyagari solved in 158 iterations, with a sup metric of 4.882033908379827e-7 and in 61.497000217437744 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (10, 0.007758620689655172, 2.7853886254058096, 9.074701794095665)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 663 iterations, with a sup metric of 9.85469483794077e-11 and in 37.978999853134155 seconds
          Klevel Aiyagari solved in 282 iterations, with a sup metric of 3.9358948733317364e-7 and in 79.57800006866455 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (11, 0.008620689655172414, 2.8279904007475043, 8.937149523011778)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 681 iterations, with a sup metric of 9.820677604466255e-11 and in 36.956000089645386 seconds
          Klevel Aiyagari solved in 295 iterations, with a sup metric of 5.698931603709984e-7 and in 81.05900001525879 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (12, 0.009482758620689655, 2.910175371327155, 8.80298376192179)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 700 iterations, with a sup metric of 9.78621628178189e-11 and in 37.88000011444092 seconds
          Klevel Aiyagari solved in 793 iterations, with a sup metric of 7.594046625814267e-7 and in 160.6210000514984 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (13, 0.010344827586206896, 3.037967966162476, 8.672089702483182)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 720 iterations, with a sup metric of 9.775735776429428e-11 and in 41.13700008392334 seconds
          Klevel Aiyagari solved in 803 iterations, with a sup metric of 2.3948873459178565e-7 and in 167.68700003623962 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (14, 0.011206896551724138, 3.1486964408089895, 8.54435747408426)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 741 iterations, with a sup metric of 9.814637991212294e-11 and in 40.09299993515015 seconds
          Klevel Aiyagari solved in 1551 iterations, with a sup metric of 2.9817091800375005e-7 and in 281.6630001068115 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (15, 0.01206896551724138, 3.248402015258355, 8.419681887353196)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 763 iterations, with a sup metric of 9.92868010030179e-11 and in 41.36699986457825 seconds
          Klevel Aiyagari solved in 599 iterations, with a sup metric of 5.719413058604359e-7 and in 133.97300004959106 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (16, 0.01293103448275862, 3.390319165936541, 8.297962193240288)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 787 iterations, with a sup metric of 9.874767670225992e-11 and in 42.47000002861023 seconds
          Klevel Aiyagari solved in 699 iterations, with a sup metric of 8.582429912652706e-7 and in 149.26600003242493 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (17, 0.013793103448275862, 3.588726411657493, 8.179101856593729)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 812 iterations, with a sup metric of 9.964296054931765e-11 and in 43.9229998588562 seconds
          Klevel Aiyagari solved in 143 iterations, with a sup metric of 6.288987454556887e-7 and in 65.61999988555908 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (18, 0.014655172413793103, 3.749742327887296, 8.063008343233282)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 839 iterations, with a sup metric of 9.977085824175447e-11 and in 45.36199998855591 seconds
          Klevel Aiyagari solved in 1306 iterations, with a sup metric of 8.071593062732745e-7 and in 247.59899997711182 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (19, 0.015517241379310345, 4.21423211566162, 7.949592919602913)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 868 iterations, with a sup metric of 9.981349080590007e-11 and in 48.194000005722046 seconds
          Klevel Aiyagari solved in 907 iterations, with a sup metric of 8.908809610643868e-7 and in 188.8989999294281 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (20, 0.016379310344827588, 5.170031451822685, 7.8387704641537015)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 900 iterations, with a sup metric of 9.869438599707792e-11 and in 48.95599985122681 seconds
          Klevel Aiyagari solved in 4059 iterations, with a sup metric of 8.076801735659768e-7 and in 667.0299999713898 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (21, 0.017241379310344827, 7.899233647649724, 7.730459289672512)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 934 iterations, with a sup metric of 9.823608593251265e-11 and in 50.670000076293945 seconds
          Klevel Aiyagari solved in 552 iterations, with a sup metric of 6.549659041072281e-7 and in 135.7829999923706 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (22, 0.01810344827586207, 8.79388477967258, 7.624580975830904)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 970 iterations, with a sup metric of 9.92468329741314e-11 and in 53.09000015258789 seconds
          Klevel Aiyagari solved in 307 iterations, with a sup metric of 8.647309616104317e-8 and in 99.65200018882751 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (23, 0.01896551724137931, 9.507630327981484, 7.521060211282674)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1010 iterations, with a sup metric of 9.809042467168183e-11 and in 54.9539999961853 seconds
          Klevel Aiyagari solved in 290 iterations, with a sup metric of 5.139959313204951e-7 and in 100.11899995803833 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (24, 0.019827586206896553, 10.694366238290039, 7.419824644687936)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1052 iterations, with a sup metric of 9.950085200216563e-11 and in 58.04700016975403 seconds
          Klevel Aiyagari solved in 576 iterations, with a sup metric of 1.0665227488963477e-7 and in 157.2920000553131 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (25, 0.020689655172413793, 12.546642873474351, 7.320804744087232)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1099 iterations, with a sup metric of 9.821121693676105e-11 and in 68.81299996376038 seconds
          Klevel Aiyagari solved in 449 iterations, with a sup metric of 6.487134350842582e-7 and in 136.1930000782013 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (26, 0.021551724137931036, 13.430336315328459, 7.223933664090859)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1149 iterations, with a sup metric of 9.908873721542477e-11 and in 62.68499994277954 seconds
          Klevel Aiyagari solved in 692 iterations, with a sup metric of 9.084963457713672e-7 and in 170.3159999847412 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (27, 0.022413793103448276, 14.704736907740788, 7.129147120387139)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1204 iterations, with a sup metric of 9.958256441677804e-11 and in 68.03800010681152 seconds
          Klevel Aiyagari solved in 1104 iterations, with a sup metric of 6.823303601718955e-7 and in 234.98800015449524 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (28, 0.02327586206896552, 16.8353664502482, 7.036383271108701)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1266 iterations, with a sup metric of 9.906031550599437e-11 and in 68.33899998664856 seconds
          Klevel Aiyagari solved in 2331 iterations, with a sup metric of 1.7482468355692765e-7 and in 428.8440001010895 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (29, 0.02413793103448276, 22.393682882273453, 6.945582604628445)
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1333 iterations, with a sup metric of 9.960032798517204e-11 and in 74.13800001144409 seconds
          Klevel Aiyagari solved in 1607 iterations, with a sup metric of 7.313640960892309e-7 and in 338.66100001335144 seconds
    (r, r_range[r], Supply1[r], Demand1[r]) = (30, 0.025, 23.526260662218945, 6.856687833386916)
    




![svg](Aiyagari%281994%29_files/Aiyagari%281994%29_63_1.svg)



### __3.5 Convergence to Equilibrium__

__Steps for Convergence to Equilibrium__

1. We start with an interest rate $r_1=\frac{1-\beta}{\beta}-\epsilon$ and find the demand for capital $K(r_1)$.
2. We evaluate $KlevelAiyagari(u, du, Kr(r_1, δ, α), θ, pop)$ and obtain the supply of capital $A(r_1)$, with this we establish the interest rate $r_2=Min\left\{f_K\left(A(r_1),1,\alpha\right)-\delta, \frac{1-\beta}{\beta}\right\}$. By definition $r_1$ and $r_2$ are at opposite sides of the equilibrium. Without loss of generality less assume $r_1>r_2$.
3. We define $r =\frac{r_1+r_2}{2}$ and run $KlevelAiyagari(u, du, Kr(r_1, δ, α), θ, pop)$ in order to find $A(r)$. If $A(r)> K(r)$ we replace $r_1 =r$, if $A(r) < K(r)$ we replace $r_2 =r$.
4. If $\lvert A(r) - K(r)\lvert<0.01$ or if $\lvert r_1 -r_2 \lvert < 1E^{-4}$ we declare $(A(r), r)$ as the equilibrium. Otherwise repeat 3. 


```julia
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
```




    EquilibriumAiyagari (generic function with 1 method)



To have an idea about the way in which this algorithm solves this problem let's run non silently one time


```julia
θ[5] = 0.4;
θ[2] = 3.0;
θ[4] = 0.6;
E = EquilibriumAiyagari(u, du, θ)
```

      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.39433607637101886 and in 629.8780000209808 seconds
          Klevel Aiyagari solved in 636 iterations, with a sup metric of 2.907807884373856e-7 and in 758.6510000228882 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.080694329923823, 0.041666666666666484, -0.03300687376558956, 2.0117815261481993, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 596 iterations, with a sup metric of 9.885958718314214e-11 and in 38.878000020980835 seconds
          Klevel Aiyagari solved in 942 iterations, with a sup metric of 3.322094215339875e-7 and in 224.04699993133545 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (5.60958020920355, 0.041666666666666484, 0.0043298964505384605, 1.1906795766627767, 4.0481809061881044).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1210 iterations, with a sup metric of 9.92734783267224e-11 and in 81.83999991416931 seconds
          Klevel Aiyagari solved in 6129 iterations, with a sup metric of 2.3469146813541142e-8 and in 1306.7649998664856 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (18.87106382645733, 0.022998281558602472, 0.0043298964505384605, 1.842751070021662, 11.80502832041246).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 799 iterations, with a sup metric of 9.736922379488533e-11 and in 52.117000102996826 seconds
          Klevel Aiyagari solved in 1680 iterations, with a sup metric of 5.958535889969842e-7 and in 392.6300001144409 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (8.304870830730687, 0.013664089004570466, 0.0043298964505384605, 1.3713231623553628, 0.10815898687194014).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 683 iterations, with a sup metric of 9.756817576089816e-11 and in 44.46000003814697 seconds
          Klevel Aiyagari solved in 1270 iterations, with a sup metric of 9.373712117869105e-7 and in 303.1410000324249 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (7.215533937921305, 0.013664089004570466, 0.008996992727554463, 1.3036368389516617, 1.66264106809587).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 736 iterations, with a sup metric of 9.888090346521494e-11 and in 48.8970000743866 seconds
          Klevel Aiyagari solved in 4287 iterations, with a sup metric of 8.558106024994449e-7 and in 843.8720002174377 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (7.6666560555014875, 0.013664089004570466, 0.011330540866062465, 1.3324107466280122, 0.8596341653822694).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 766 iterations, with a sup metric of 9.874412398858112e-11 and in 42.89800000190735 seconds
          Klevel Aiyagari solved in 5446 iterations, with a sup metric of 6.325338910666351e-7 and in 993.8079998493195 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (8.024550218783864, 0.013664089004570466, 0.012497314935316466, 1.354476347937818, 0.33428758345430154).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 782 iterations, with a sup metric of 9.849543403106509e-11 and in 53.972999811172485 seconds
          Klevel Aiyagari solved in 8175 iterations, with a sup metric of 9.655458670628187e-7 and in 1830.2919998168945 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (8.17479923458003, 0.013664089004570466, 0.013080701969943466, 1.3635520743621872, 0.10232463384893187).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 790 iterations, with a sup metric of 9.905143372179737e-11 and in 53.28699994087219 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.01982179731368907 and in 2222.7200000286102 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (8.265175706047433, 0.013372395487256966, 0.013080701969943466, 1.3689598991550478, 0.028418778212682128).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 786 iterations, with a sup metric of 9.871747863599012e-11 and in 54.13599991798401 seconds
          Klevel Aiyagari solved in 8030 iterations, with a sup metric of 5.727642168292251e-7 and in 1767.1859998703003 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (8.187324769849921, 0.013372395487256966, 0.013226548728600216, 1.3643038367887221, 0.06957517193028373).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 788 iterations, with a sup metric of 9.886846896733914e-11 and in 54.39699983596802 seconds
          Klevel Aiyagari solved in 8737 iterations, with a sup metric of 5.761698056050072e-7 and in 3449.4879999160767 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (8.295533162450775, 0.013299472107928591, 0.013226548728600216, 1.37076789237227, 0.048714813571944404).
    Equilibrium found in 14092.661000013351 seconds
    




    (8.295533162450775, 0.013299472107928591, 1.37076789237227, 0.30868341855872705, [0.00010000000000000002 0.0 … 0.0 0.0; 0.0004000000000000001 0.0 … 0.0 0.00010000000000000002; … ; 0.0 0.0 … 0.0005000000000000001 0.0009000000000000002; 0.0 0.0 … 0.00010000000000000002 0.0005000000000000001], 25.0, [4.0960000000000004e-16, 5.2428800000000005e-14, 8.957952000000001e-13, 6.710886400000001e-12, 3.2000000000000006e-11, 1.1466178560000002e-10, 3.373232128e-10, 8.589934592000001e-10, 1.959104102399999e-9, 4.096000000000001e-9  …  19.3410142982354, 19.90982807496902, 20.492920872550275, 21.090590156950757, 21.703138331168, 22.3308727964269, 22.97410601388485, 23.633155566842625, 24.308344223463116, 25.0], [4.0960000000000004e-16 4.0960000000000004e-16 … 22.97410601388485 23.633155566842625; 4.0960000000000004e-16 4.0960000000000004e-16 … 22.97410601388485 23.633155566842625; … ; 2.2293220545737324 2.2293220545737324 … 25.0 25.0; 4.391392213728001 4.391392213728001 … 25.0 25.0], [-0.7467732229553574 -0.7467732229535029 … 12.97556553921805 13.045485092353381; 4.235778945201329 4.2357789452017425 … 13.064048527315531 13.130327022627062; … ; 10.310312343943217 10.310312343943233 … 13.603084316438183 13.617168828983365; 11.272980856037767 11.272980856037778 … 13.698788295757929 13.70187753421016], [0.22313016014842982, 0.36787944117144233, 0.6065306597126334, 1.0, 1.6487212707001282, 2.718281828459045, 4.4816890703380645], 8.295533162450775, [5.446807380113248, 24.080694329923823, 5.60958020920355, 18.87106382645733, 8.304870830730687, 7.215533937921305, 7.6666560555014875, 8.024550218783864, 8.17479923458003, 8.265175706047433, 8.187324769849921, 8.295533162450775], [0.041666666666666484, -0.03300687376558956, 0.0043298964505384605, 0.022998281558602472, 0.013664089004570466, 0.008996992727554463, 0.011330540866062465, 0.012497314935316466, 0.013080701969943466, 0.013372395487256966, 0.013226548728600216, 0.013299472107928591], [1.1781242629578268, 2.0117815261481993, 1.1906795766627767, 1.842751070021662, 1.3713231623553628, 1.3036368389516617, 1.3324107466280122, 1.354476347937818, 1.3635520743621872, 1.3689598991550478, 1.3643038367887221, 1.37076789237227], [12.721576923511934; 8.41891958739923; … ; 10.216889909248; 24.308344223463116], [1.0, 1.0, 0.6065306597126334, 1.6487212707001282, 1.6487212707001282, 1.0, 1.0, 1.0, 1.0, 1.6487212707001282  …  1.0, 1.0, 1.0, 1.6487212707001282, 1.0, 1.6487212707001282, 1.0, 1.6487212707001282, 1.0, 1.0])



We can see the way in which the algorithm converges to the equilibrium valu of $K$ and $r$


```julia
plot(1:size(E[12])[1], E[12], title ="Convergence Path of K",xlabel="Algorithm iteration", ylabel="Level of per capita aggregate Capital", legend=:bottomright, label="K")
```




![svg](Aiyagari%281994%29_files/Aiyagari%281994%29_70_0.svg)




```julia
plot(1:size(E[13])[1], E[13], title ="Convergence Path of r",xlabel="Algorithm iteration", ylabel="Level of interest rate", legend=:bottomright, label="K")
```




![svg](Aiyagari%281994%29_files/Aiyagari%281994%29_71_0.svg)



Now we estimate the final results in table A and B of Aiyagari and compare our results with his estimations (the following line of commands takes around one day to run):


```julia
#TABLE A
θ[5] = 0.2;
# Column 1
θ[2] = 1.0;
θ[4] = 0.0;
AE11 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.3;
AE21 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.6;
AE31 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.9;
AE41 = EquilibriumAiyagari(u, du, θ);
# Column 2
θ[2] = 3.0;
θ[4] = 0.0;
AE12 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.3;
AE22 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.6;
AE32 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.9;
AE42 = EquilibriumAiyagari(u, du, θ);
# Column 3
θ[2] = 5.0;
θ[4] = 0.0;
AE13 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.3;
AE23 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.6;
AE33 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.9;
AE43 = EquilibriumAiyagari(u, du, θ);
#TABLE B
θ[5] = 0.4;
# Column 1
θ[2] = 1.0;
θ[4] = 0.0;
BE11 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.3;
BE21 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.6;
BE31 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.9;
BE41 = EquilibriumAiyagari(u, du, θ);
# Column 2
θ[2] = 3.0;
θ[4] = 0.0;
BE12 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.3;
BE22 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.6;
BE32 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.9;
BE42 = EquilibriumAiyagari(u, du, θ);
# Column 3
θ[2] = 5.0;
θ[4] = 0.0;
BE13 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.3;
BE23 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.6;
BE33 = EquilibriumAiyagari(u, du, θ);
θ[4] = 0.9;
BE43 = EquilibriumAiyagari(u, du, θ);
```

      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.8027649081013806 and in 160.04999995231628 seconds
          Klevel Aiyagari solved in 102 iterations, with a sup metric of NaN and in 175.74899983406067 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (25.0, 0.041666666666666484, -0.034120265586010584, 2.0390993072884185, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 597 iterations, with a sup metric of 9.832135106080386e-11 and in 9.809000015258789 seconds
          Klevel Aiyagari solved in 533 iterations, with a sup metric of 9.401574527009033e-7 and in 96.45100021362305 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (0.2414743994162356, 0.041666666666666484, 0.00377320054032795, 0.3837192332898275, 9.516752818858862).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1208 iterations, with a sup metric of 9.880452012112073e-11 and in 19.460999965667725 seconds
          Klevel Aiyagari solved in 355 iterations, with a sup metric of 9.543583657724584e-7 and in 74.19900012016296 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (24.198881591098406, 0.022719933603497215, 0.00377320054032795, 2.0153305113622246, 17.10290552800088).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 794 iterations, with a sup metric of 9.738698736327933e-11 and in 12.619999885559082 seconds
          Klevel Aiyagari solved in 429 iterations, with a sup metric of 9.418154812081397e-7 and in 82.39699983596802 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (0.4093378014474179, 0.022719933603497215, 0.013246567071912583, 0.46401224442472655, 7.844792610134647).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 951 iterations, with a sup metric of 9.847767046267109e-11 and in 15.240000009536743 seconds
          Klevel Aiyagari solved in 247 iterations, with a sup metric of 2.5682176364481075e-7 and in 55.25 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (0.7058676305501225, 0.022719933603497215, 0.017983250337704898, 0.5645733188201312, 6.933332793068765).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1056 iterations, with a sup metric of 9.958611713045684e-11 and in 16.83999991416931 seconds
          Klevel Aiyagari solved in 553 iterations, with a sup metric of 5.734931902318106e-7 and in 103.92899990081787 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (2.3400288033308834, 0.022719933603497215, 0.020351591970601057, 0.8691597515818907, 5.019347177903276).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1130 iterations, with a sup metric of 9.92521620446496e-11 and in 18.24500012397766 seconds
          Klevel Aiyagari solved in 1339 iterations, with a sup metric of 5.807874158649295e-7 and in 225.97399997711182 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (23.354771242174056, 0.021535762787049134, 0.020351591970601057, 1.989734803027559, 16.129063128825706).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1091 iterations, with a sup metric of 9.885425811262394e-11 and in 17.544999837875366 seconds
          Klevel Aiyagari solved in 546 iterations, with a sup metric of 8.284774144959595e-7 and in 103.68499994277954 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (11.2733831742326, 0.020943677378825097, 0.020351591970601057, 1.5308037261786274, 3.9813433927841677).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1073 iterations, with a sup metric of 9.865175343293231e-11 and in 19.05299997329712 seconds
          Klevel Aiyagari solved in 1479 iterations, with a sup metric of 7.375115730623423e-7 and in 248.79399991035461 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (9.340643840229095, 0.020647634674713075, 0.020351591970601057, 1.4305914546832352, 2.015062841656711).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1065 iterations, with a sup metric of 9.798384326131782e-11 and in 17.187000036239624 seconds
          Klevel Aiyagari solved in 625 iterations, with a sup metric of 9.69531075232981e-7 and in 115.37999987602234 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (5.682473991268011, 0.020647634674713075, 0.020499613322657068, 1.1962266081524515, 1.6599726114687643).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1069 iterations, with a sup metric of 9.827161306930066e-11 and in 16.937999963760376 seconds
          Klevel Aiyagari solved in 1056 iterations, with a sup metric of 9.306193200916106e-7 and in 182.47699999809265 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (7.781328567534022, 0.02057362399868507, 0.020499613322657068, 1.3395512234617855, 0.44732271777444677).
    Equilibrium found in 1464.3600001335144 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.8033331071692373 and in 173.83400011062622 seconds
          Klevel Aiyagari solved in 102 iterations, with a sup metric of NaN and in 190.3989999294281 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (25.0, 0.041666666666666484, -0.034120265586010584, 2.0390993072884185, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 597 iterations, with a sup metric of 9.78417347141658e-11 and in 9.496000051498413 seconds
          Klevel Aiyagari solved in 291 iterations, with a sup metric of 5.134962805980756e-7 and in 58.01900005340576 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (0.28708334509750955, 0.041666666666666484, 0.00377320054032795, 0.40837848462508197, 9.471143873177589).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1208 iterations, with a sup metric of 9.870149142443552e-11 and in 18.441999912261963 seconds
          Klevel Aiyagari solved in 272 iterations, with a sup metric of 7.671524443997689e-7 and in 62.08999991416931 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (24.106016890186687, 0.022719933603497215, 0.00377320054032795, 2.012542861231348, 17.01004082708916).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 793 iterations, with a sup metric of 9.983835980165168e-11 and in 12.707000017166138 seconds
          Klevel Aiyagari solved in 425 iterations, with a sup metric of 1.7079062270191705e-7 and in 79.33500003814697 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (0.5448345884562763, 0.022719933603497215, 0.013246567071912583, 0.514322182218116, 7.709295823125788).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 951 iterations, with a sup metric of 9.86268844371807e-11 and in 15.270999908447266 seconds
          Klevel Aiyagari solved in 243 iterations, with a sup metric of 9.248656858047178e-7 and in 55.51899981498718 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (1.039630836581983, 0.022719933603497215, 0.017983250337704898, 0.649017592267647, 6.599569587036905).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1057 iterations, with a sup metric of 9.899636665977596e-11 and in 16.924999952316284 seconds
          Klevel Aiyagari solved in 487 iterations, with a sup metric of 7.216486826731322e-7 and in 92.50300002098083 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (5.361237277524327, 0.022719933603497215, 0.020351591970601057, 1.1714274167823264, 1.9981387037098326).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1130 iterations, with a sup metric of 9.907452636070957e-11 and in 17.999000072479248 seconds
          Klevel Aiyagari solved in 1253 iterations, with a sup metric of 9.497867241998422e-7 and in 216.0149998664856 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (23.27243922376123, 0.021535762787049134, 0.020351591970601057, 1.9872067776249702, 16.04673111041288).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1091 iterations, with a sup metric of 9.837819447966467e-11 and in 17.27400016784668 seconds
          Klevel Aiyagari solved in 925 iterations, with a sup metric of 4.317907025671655e-7 and in 162.89800000190735 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (12.485269474068643, 0.020943677378825097, 0.020351591970601057, 1.588119671716558, 5.19322969262021).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1073 iterations, with a sup metric of 9.973533110496646e-11 and in 16.95799994468689 seconds
          Klevel Aiyagari solved in 594 iterations, with a sup metric of 5.461912469384256e-7 and in 108.85399985313416 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (9.269230809170939, 0.020647634674713075, 0.020351591970601057, 1.426644297230995, 1.9436498105985542).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1065 iterations, with a sup metric of 9.912071163853398e-11 and in 16.978000164031982 seconds
          Klevel Aiyagari solved in 591 iterations, with a sup metric of 9.321133025015229e-7 and in 110.80400013923645 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (7.370889928223056, 0.020499613322657068, 0.020351591970601057, 1.3136726168942883, 0.028443325486280813).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1061 iterations, with a sup metric of 9.902478836920636e-11 and in 17.14800000190735 seconds
          Klevel Aiyagari solved in 606 iterations, with a sup metric of 4.080444201102047e-7 and in 113.92899990081787 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (6.504934850431803, 0.020499613322657068, 0.020425602646629064, 1.2558779530052597, 0.845968448830285).
    Equilibrium found in 1250.4370000362396 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.8060872231908434 and in 158.9650001525879 seconds
          Klevel Aiyagari solved in 102 iterations, with a sup metric of NaN and in 175.5460000038147 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (25.0, 0.041666666666666484, -0.034120265586010584, 2.0390993072884185, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 597 iterations, with a sup metric of 9.771738973540778e-11 and in 9.150999784469604 seconds
          Klevel Aiyagari solved in 276 iterations, with a sup metric of 6.113116531620028e-7 and in 53.473999977111816 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (0.406526571081952, 0.041666666666666484, 0.00377320054032795, 0.4628624941837032, 9.351700647193145).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1208 iterations, with a sup metric of 9.874412398858112e-11 and in 19.319999933242798 seconds
          Klevel Aiyagari solved in 997 iterations, with a sup metric of 6.529923385155533e-7 and in 175.81299996376038 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (23.85609973828709, 0.022719933603497215, 0.00377320054032795, 2.0050064452891645, 16.76012367518956).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 794 iterations, with a sup metric of 9.773870601748058e-11 and in 12.320000171661377 seconds
          Klevel Aiyagari solved in 299 iterations, with a sup metric of 8.252401954929646e-7 and in 59.42700004577637 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (0.9071588923108426, 0.022719933603497215, 0.013246567071912583, 0.6179395386957154, 7.3469715192712215).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 952 iterations, with a sup metric of 9.842082704381028e-11 and in 14.918999910354614 seconds
          Klevel Aiyagari solved in 766 iterations, with a sup metric of 6.651251506074216e-7 and in 139.4909999370575 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (2.0086382335670745, 0.022719933603497215, 0.017983250337704898, 0.8226679819628843, 5.630562190051814).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1060 iterations, with a sup metric of 9.827516578297946e-11 and in 16.29699993133545 seconds
          Klevel Aiyagari solved in 867 iterations, with a sup metric of 9.108835052062126e-7 and in 155.95099997520447 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (9.711504149337838, 0.020351591970601057, 0.017983250337704898, 1.4507851915612782, 2.352128168103678).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1002 iterations, with a sup metric of 9.908518450174597e-11 and in 15.749000072479248 seconds
          Klevel Aiyagari solved in 1428 iterations, with a sup metric of 7.932954232063444e-7 and in 243.46600008010864 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (2.6625757838262722, 0.020351591970601057, 0.01916742115415298, 0.9105183149283386, 4.83457187281113).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1029 iterations, with a sup metric of 9.972112025025126e-11 and in 15.818000078201294 seconds
          Klevel Aiyagari solved in 535 iterations, with a sup metric of 9.887847489850696e-7 and in 100.18899989128113 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (3.15662478531881, 0.020351591970601057, 0.019759506562377016, 0.9680554635216568, 4.2711132024511675).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1044 iterations, with a sup metric of 9.900702480081236e-11 and in 16.317999839782715 seconds
          Klevel Aiyagari solved in 778 iterations, with a sup metric of 5.252895653482321e-7 and in 138.65700006484985 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (7.052503845879921, 0.020351591970601057, 0.020055549266489038, 1.2929554865251556, 0.34092356063439766).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1052 iterations, with a sup metric of 9.841727433013148e-11 and in 16.259000062942505 seconds
          Klevel Aiyagari solved in 493 iterations, with a sup metric of 9.936514514110955e-7 and in 92.70000004768372 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (8.057450560318088, 0.020203570618545046, 0.020055549266489038, 1.3564729188784521, 0.6810810904987461).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1048 iterations, with a sup metric of 9.866241157396871e-11 and in 16.738999843597412 seconds
          Klevel Aiyagari solved in 311 iterations, with a sup metric of 8.583218839190723e-7 and in 68.81699991226196 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (7.123473249835518, 0.020203570618545046, 0.020129559942517042, 1.2976244536078079, 0.2614171111021424).
    Equilibrium found in 1403.5929999351501 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.8322313018379646 and in 157.55599999427795 seconds
          Klevel Aiyagari solved in 104 iterations, with a sup metric of 1.402545069408302e-7 and in 174.71900010108948 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.99973308727964, 0.041666666666666484, -0.03411995208861712, 2.0390914698954203, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 597 iterations, with a sup metric of 9.869793871075672e-11 and in 9.712000131607056 seconds
          Klevel Aiyagari solved in 9332 iterations, with a sup metric of 7.187217088208672e-7 and in 1486.3830001354218 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (1.427323068868431, 0.041666666666666484, 0.0037733572890246807, 0.7274581211763235, 8.330875620262612).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1210 iterations, with a sup metric of 9.968204039978446e-11 and in 19.211999893188477 seconds
          Klevel Aiyagari solved in 686 iterations, with a sup metric of 3.381906005394578e-7 and in 129.73799991607666 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (16.848189402786126, 0.02272001197784558, 0.0037733572890246807, 1.7690460532953107, 9.752221799310565).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 796 iterations, with a sup metric of 9.888623253573314e-11 and in 12.652999877929688 seconds
          Klevel Aiyagari solved in 3509 iterations, with a sup metric of 9.809380572812836e-7 and in 559.1489999294281 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (4.167318782970711, 0.02272001197784558, 0.01324668463343513, 1.0698635573441622, 4.086795368519266).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 959 iterations, with a sup metric of 9.7855945568881e-11 and in 16.55900001525879 seconds
          Klevel Aiyagari solved in 233 iterations, with a sup metric of 4.760278480355873e-7 and in 59.39800000190735 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (7.984048440613982, 0.017983348305640356, 0.01324668463343513, 1.3520112730006917, 0.3448599513648203).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 870 iterations, with a sup metric of 9.773160059012298e-11 and in 13.812000036239624 seconds
          Klevel Aiyagari solved in 1917 iterations, with a sup metric of 2.626056486166521e-7 and in 316.63700008392334 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (6.271702391993937, 0.017983348305640356, 0.015615016469537744, 1.2394777440827869, 1.6651923295728155).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 912 iterations, with a sup metric of 9.848122317634989e-11 and in 14.321000099182129 seconds
          Klevel Aiyagari solved in 476 iterations, with a sup metric of 7.236784310140321e-7 and in 90.57200002670288 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (6.8818018875501625, 0.017983348305640356, 0.01679918238758905, 1.2816007525673232, 0.9039066797976743).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 935 iterations, with a sup metric of 9.794121069717221e-11 and in 14.652999877929688 seconds
          Klevel Aiyagari solved in 4161 iterations, with a sup metric of 9.627126365607494e-7 and in 672.547000169754 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (7.4639425471903476, 0.017983348305640356, 0.017391265346614705, 1.3196189929138373, 0.2479353428346709).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 946 iterations, with a sup metric of 9.984546522900928e-11 and in 14.940000057220459 seconds
          Klevel Aiyagari solved in 1253 iterations, with a sup metric of 9.68116252308181e-7 and in 212.614000082016 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (7.848225101370894, 0.01768730682612753, 0.017391265346614705, 1.3436857032289, 0.1728330317880804).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 940 iterations, with a sup metric of 9.99662574940885e-11 and in 14.41100001335144 seconds
          Klevel Aiyagari solved in 3160 iterations, with a sup metric of 9.470055410280002e-7 and in 517.9749999046326 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (7.61053549843296, 0.01768730682612753, 0.01753928608637112, 1.328891280295328, 0.08306401058349788).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 943 iterations, with a sup metric of 9.988454507947608e-11 and in 14.230999946594238 seconds
          Klevel Aiyagari solved in 124 iterations, with a sup metric of 8.220281359053305e-7 and in 34.20799994468689 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (7.487291804160057, 0.01768730682612753, 0.017613296456249324, 1.3211036344639364, 0.19719514139256855).
    Equilibrium found in 4253.981999874115 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.3972162672821469 and in 535.2969999313354 seconds
          Klevel Aiyagari solved in 147 iterations, with a sup metric of 2.473867798773093e-27 and in 559.231999874115 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.985821056580996, 0.041666666666666484, -0.03410360436747148, 2.038682894962495, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 587 iterations, with a sup metric of 9.85966863709109e-11 and in 31.413999795913696 seconds
          Klevel Aiyagari solved in 149 iterations, with a sup metric of 7.555352134159309e-7 and in 53.5479998588562 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (0.5696473033195438, 0.041666666666666484, 0.0037815311495975022, 0.5226345909874983, 9.187063886758317).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1188 iterations, with a sup metric of 9.989875593419129e-11 and in 63.456000089645386 seconds
          Klevel Aiyagari solved in 1806 iterations, with a sup metric of 5.761639572077877e-7 and in 347.03000020980835 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (23.358239457495262, 0.02272409890813199, 0.0037815311495975022, 1.9898411701707575, 16.262712968871092).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 782 iterations, with a sup metric of 9.93303217455832e-11 and in 42.79299998283386 seconds
          Klevel Aiyagari solved in 128 iterations, with a sup metric of 5.06814041022563e-7 and in 62.412999868392944 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (0.835517022120575, 0.02272409890813199, 0.013252815028864747, 0.599906935463337, 7.417749298930058).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 939 iterations, with a sup metric of 9.779022036582319e-11 and in 52.33899998664856 seconds
          Klevel Aiyagari solved in 245 iterations, with a sup metric of 3.868535935704374e-7 and in 91.04500007629395 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (1.2753083597788477, 0.02272409890813199, 0.017988456968498368, 0.6985563556607896, 6.363257838917844).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1045 iterations, with a sup metric of 9.912781706589158e-11 and in 55.86899995803833 seconds
          Klevel Aiyagari solved in 615 iterations, with a sup metric of 4.359029608896272e-7 and in 150.643000125885 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (6.096724377257418, 0.02272409890813199, 0.02035627793831518, 1.2269157348701145, 1.2621146833851622).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1112 iterations, with a sup metric of 9.876544027065393e-11 and in 60.164000034332275 seconds
          Klevel Aiyagari solved in 2495 iterations, with a sup metric of 1.543449294569746e-7 and in 452.0739998817444 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (12.279819880751312, 0.021540188423223586, 0.02035627793831518, 1.578661773169485, 5.054603844168733).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1077 iterations, with a sup metric of 9.890754881780595e-11 and in 56.704999923706055 seconds
          Klevel Aiyagari solved in 936 iterations, with a sup metric of 5.226028737426201e-7 and in 207.0239999294281 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (10.229182079865684, 0.020948233180769382, 0.02035627793831518, 1.4781642731150808, 2.937656495556573).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1061 iterations, with a sup metric of 9.853451388153189e-11 and in 57.125 seconds
          Klevel Aiyagari solved in 913 iterations, with a sup metric of 7.467399420377938e-7 and in 196.06700015068054 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (8.906720740515011, 0.020652255559542282, 0.02035627793831518, 1.4063013538324711, 1.5816652242832534).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1053 iterations, with a sup metric of 9.869793871075672e-11 and in 56.18899989128113 seconds
          Klevel Aiyagari solved in 613 iterations, with a sup metric of 2.6644003179477774e-7 and in 152.42700004577637 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (7.127149488806728, 0.020652255559542282, 0.02050426674892873, 1.2978654950694237, 0.2147659317398407).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1057 iterations, with a sup metric of 9.85629355909623e-11 and in 57.212000131607056 seconds
          Klevel Aiyagari solved in 920 iterations, with a sup metric of 2.9049333269532656e-7 and in 199.7149999141693 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (8.388679608415359, 0.020578261154235507, 0.02050426674892873, 1.3762891074875776, 1.0552020861204099).
    Equilibrium found in 2471.2380001544952 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.39704988096491434 and in 532.0700001716614 seconds
          Klevel Aiyagari solved in 109 iterations, with a sup metric of 1.025203632425227e-26 and in 549.9690001010895 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.97780275690752, 0.041666666666666484, -0.03409417545695323, 2.0384473436967587, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 587 iterations, with a sup metric of 9.776712772691099e-11 and in 32.02500009536743 seconds
          Klevel Aiyagari solved in 732 iterations, with a sup metric of 2.552518820590225e-7 and in 146.49399995803833 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (0.816046823996515, 0.041666666666666484, 0.003786245604856627, 0.5948362161988636, 8.939806587118802).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1188 iterations, with a sup metric of 9.962164426724485e-11 and in 62.8030002117157 seconds
          Klevel Aiyagari solved in 1240 iterations, with a sup metric of 2.2437874179051596e-7 and in 255.76200008392334 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (23.20690123396441, 0.022726456135761554, 0.003786245604856627, 1.9851903224738632, 16.111629147659183).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 782 iterations, with a sup metric of 9.901413022816996e-11 and in 42.11199998855591 seconds
          Klevel Aiyagari solved in 322 iterations, with a sup metric of 5.649647679748856e-7 and in 87.98300004005432 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (1.3029665567431163, 0.022726456135761554, 0.01325635087030909, 0.7039729158962681, 6.949810824250668).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 939 iterations, with a sup metric of 9.816858437261544e-11 and in 46.04100012779236 seconds
          Klevel Aiyagari solved in 463 iterations, with a sup metric of 7.385976732045797e-7 and in 113.51800012588501 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (1.883099477427036, 0.022726456135761554, 0.01799140350303532, 0.8037747307253793, 5.755107839202457).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1046 iterations, with a sup metric of 9.85593828772835e-11 and in 50.670000076293945 seconds
          Klevel Aiyagari solved in 860 iterations, with a sup metric of 5.841583480825514e-7 and in 174.27600002288818 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (8.319408152532173, 0.020358929819398436, 0.01799140350303532, 1.3721868381146063, 0.9608729173237371).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 988 iterations, with a sup metric of 9.973533110496646e-11 and in 47.84600019454956 seconds
          Klevel Aiyagari solved in 1620 iterations, with a sup metric of 2.9545174084414586e-7 and in 278.2240002155304 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (3.345602718775152, 0.020358929819398436, 0.01917516666121688, 0.9885320320541348, 4.1506300803341585).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1016 iterations, with a sup metric of 9.918466048475238e-11 and in 51.48099994659424 seconds
          Klevel Aiyagari solved in 728 iterations, with a sup metric of 2.526140376872568e-7 and in 154.37199997901917 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (5.986309881128861, 0.020358929819398436, 0.019767048240307658, 1.2188697082213173, 1.440550806444219).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1031 iterations, with a sup metric of 9.839240533437987e-11 and in 50.39800000190735 seconds
          Klevel Aiyagari solved in 902 iterations, with a sup metric of 9.053161868400225e-7 and in 180.78800010681152 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (7.307097807690343, 0.020358929819398436, 0.020062989029853047, 1.3095682746014097, 0.08547069920971229).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1038 iterations, with a sup metric of 9.942979772858962e-11 and in 50.454999923706055 seconds
          Klevel Aiyagari solved in 1141 iterations, with a sup metric of 4.1801141317184473e-7 and in 208.7779998779297 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (8.037020750519504, 0.02021095942462574, 0.020062989029853047, 1.35523374280262, 0.6615010728794939).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1034 iterations, with a sup metric of 9.985612337004568e-11 and in 50.032999992370605 seconds
          Klevel Aiyagari solved in 546 iterations, with a sup metric of 5.856060044245598e-7 and in 126.32800006866455 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (6.9737677252374475, 0.02021095942462574, 0.020136974227239396, 1.2877402375858946, 0.4102682974926193).
    Equilibrium found in 2276.5469999313354 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.3963668504738962 and in 484.1329998970032 seconds
          Klevel Aiyagari solved in 147 iterations, with a sup metric of 4.930380657631324e-28 and in 504.00099992752075 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.969361295811407, 0.041666666666666484, -0.034084243575404974, 2.03819930904211, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 586 iterations, with a sup metric of 9.944578494014422e-11 and in 28.06599998474121 seconds
          Klevel Aiyagari solved in 307 iterations, with a sup metric of 6.623045327801125e-7 and in 70.9359998703003 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (1.470548889981021, 0.041666666666666484, 0.003791211545630755, 0.7353135725738125, 8.284401118986658).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1186 iterations, with a sup metric of 9.972467296393006e-11 and in 57.22100019454956 seconds
          Klevel Aiyagari solved in 989 iterations, with a sup metric of 6.725525702310289e-7 and in 197.11100006103516 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (18.81795130952468, 0.02272893910614862, 0.003791211545630755, 1.8408822768416988, 11.72294718007014).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 782 iterations, with a sup metric of 9.876188755697513e-11 and in 37.77900004386902 seconds
          Klevel Aiyagari solved in 1897 iterations, with a sup metric of 7.784228068877207e-7 and in 304.9960000514984 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (2.471551404624145, 0.02272893910614862, 0.013260075325889688, 0.8864393918379776, 5.78071100591233).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 939 iterations, with a sup metric of 9.99662574940885e-11 and in 45.15499997138977 seconds
          Klevel Aiyagari solved in 795 iterations, with a sup metric of 1.7916612941164435e-7 and in 155.97099995613098 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (3.7748706949884006, 0.02272893910614862, 0.017994507216019154, 1.0324397636480966, 3.8629586254541457).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1048 iterations, with a sup metric of 9.990941407522769e-11 and in 50.938000202178955 seconds
          Klevel Aiyagari solved in 1374 iterations, with a sup metric of 1.9518878311685642e-7 and in 241.40300011634827 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (11.746742262473106, 0.020361723161083888, 0.017994507216019154, 1.5536394931028759, 4.388527037561943).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 991 iterations, with a sup metric of 9.833200920184026e-11 and in 47.97000002861023 seconds
          Klevel Aiyagari solved in 607 iterations, with a sup metric of 9.942234587798134e-7 and in 133.5090000629425 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (8.193772845456053, 0.01917811518855152, 0.017994507216019154, 1.364690552917813, 0.6978882623830547).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 964 iterations, with a sup metric of 9.960920976936904e-11 and in 46.63199996948242 seconds
          Klevel Aiyagari solved in 276 iterations, with a sup metric of 9.51763197444154e-8 and in 84.93300008773804 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (6.767729993691518, 0.01917811518855152, 0.018586311202285336, 1.2739120994873177, 0.7985810973582916).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 977 iterations, with a sup metric of 9.971401482289366e-11 and in 47.424999952316284 seconds
          Klevel Aiyagari solved in 636 iterations, with a sup metric of 9.018280547199886e-7 and in 136.06299996376038 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (7.742545135779955, 0.018882213195418426, 0.018586311202285336, 1.3371438231266237, 0.21158230970188008).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 971 iterations, with a sup metric of 9.847411774899228e-11 and in 47.031999826431274 seconds
          Klevel Aiyagari solved in 2136 iterations, with a sup metric of 9.150907758567748e-7 and in 343.6949999332428 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (7.32341894815469, 0.018882213195418426, 0.01873426219885188, 1.3106205419214123, 0.22518407744251867).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 974 iterations, with a sup metric of 9.906209186283377e-11 and in 47.11299991607666 seconds
          Klevel Aiyagari solved in 357 iterations, with a sup metric of 9.289437165492417e-7 and in 96.22499990463257 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (7.3050439106855745, 0.018882213195418426, 0.018808237697135154, 1.3094357478798222, 0.23473055453419178).
    Equilibrium found in 2268.8959999084473 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.3903775074072655 and in 484.5499999523163 seconds
          Klevel Aiyagari solved in 201 iterations, with a sup metric of 0.0 and in 512.4300000667572 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.99811257858877, 0.041666666666666484, -0.03411804863230137, 2.039043885537894, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 579 iterations, with a sup metric of 9.701395242700528e-11 and in 27.950000047683716 seconds
          Klevel Aiyagari solved in 1528 iterations, with a sup metric of 7.39130359012882e-7 and in 244.03800010681152 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (8.312108163444659, 0.041666666666666484, 0.0037743090171825575, 1.3717532598368398, 1.4459173087475978).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1184 iterations, with a sup metric of 9.880096740744193e-11 and in 57.40700006484985 seconds
          Klevel Aiyagari solved in 403 iterations, with a sup metric of 4.495107525715011e-7 and in 114.49699997901917 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (17.26473783105525, 0.02272048784192452, 0.0037743090171825575, 1.784668546644092, 10.168821591354568).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 777 iterations, with a sup metric of 9.904788100811857e-11 and in 37.89800000190735 seconds
          Klevel Aiyagari solved in 709 iterations, with a sup metric of 9.524451136227183e-7 and in 135.05299997329712 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (12.073690932353452, 0.013247398429553539, 0.0037743090171825575, 1.569070269607455, 3.819675505835578).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 663 iterations, with a sup metric of 9.843148518484668e-11 and in 31.786999940872192 seconds
          Klevel Aiyagari solved in 715 iterations, with a sup metric of 5.764600774024145e-7 and in 131.885999917984 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (9.297914309643232, 0.008510853723368048, 0.0037743090171825575, 1.4282320289445045, 0.34343000084797204).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 618 iterations, with a sup metric of 9.818457158417004e-11 and in 30.005000114440918 seconds
          Klevel Aiyagari solved in 742 iterations, with a sup metric of 5.988239598612593e-7 and in 133.33500003814697 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (8.761490562633483, 0.008510853723368048, 0.006142581370275303, 1.3980028381232972, 0.5806138551240085).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 640 iterations, with a sup metric of 9.732303851706092e-11 and in 31.16600012779236 seconds
          Klevel Aiyagari solved in 2013 iterations, with a sup metric of 8.26669067185718e-7 and in 315.8589999675751 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (9.005046862673188, 0.008510853723368048, 0.0073267175468216756, 1.411870700704203, 0.1398804366926516).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 651 iterations, with a sup metric of 9.880629647796013e-11 and in 31.483999967575073 seconds
          Klevel Aiyagari solved in 1239 iterations, with a sup metric of 3.942010972698131e-7 and in 205.05100011825562 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (9.216715918526402, 0.007918785635094861, 0.0073267175468216756, 1.423729246854937, 0.1678317052304994).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 645 iterations, with a sup metric of 9.953637913895363e-11 and in 31.355000019073486 seconds
          Klevel Aiyagari solved in 1765 iterations, with a sup metric of 5.764108796323681e-7 and in 283.34299993515015 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (9.079365227034122, 0.007918785635094861, 0.007622751590958269, 1.4160544368313157, 0.01733265828378805).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 648 iterations, with a sup metric of 9.912426435221278e-11 and in 36.13000011444092 seconds
          Klevel Aiyagari solved in 981 iterations, with a sup metric of 7.346512205777635e-7 and in 191.54500007629395 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (8.946097088521775, 0.007918785635094861, 0.007770768613026565, 1.4085363930152497, 0.12664230517024144).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 650 iterations, with a sup metric of 9.733547301493672e-11 and in 36.75 seconds
          Klevel Aiyagari solved in 646 iterations, with a sup metric of 5.579359007306699e-7 and in 141.72099995613098 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (9.01764311370713, 0.007918785635094861, 0.007844777124060714, 1.4125813551291675, 0.043155814638002).
    Equilibrium found in 2408.7870001792908 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.23896124740485902 and in 555.0629999637604 seconds
          Klevel Aiyagari solved in 205 iterations, with a sup metric of 1.932906433017784e-27 and in 586.7669999599457 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.937708161925894, 0.041666666666666484, -0.03404695268824053, 2.0372687697988106, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 579 iterations, with a sup metric of 9.872813677702652e-11 and in 32.105000019073486 seconds
          Klevel Aiyagari solved in 561 iterations, with a sup metric of 2.959416760679406e-7 and in 119.17699980735779 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (0.8261038834538115, 0.041666666666666484, 0.0038098569892129758, 0.5974649778569334, 8.925455377479945).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1171 iterations, with a sup metric of 9.85771464456775e-11 and in 63.52900004386902 seconds
          Klevel Aiyagari solved in 1196 iterations, with a sup metric of 6.443605532726277e-7 and in 250.8400001525879 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (21.10116862183749, 0.02273826182793973, 0.0038098569892129758, 1.91836091237186, 14.007170432445857).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 773 iterations, with a sup metric of 9.849365767422569e-11 and in 41.35199999809265 seconds
          Klevel Aiyagari solved in 779 iterations, with a sup metric of 4.962886115569296e-7 and in 166.75500011444092 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (1.2044685513908615, 0.02273826182793973, 0.013274059408576353, 0.6843312009889968, 7.045860787960682).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 928 iterations, with a sup metric of 9.898215580506076e-11 and in 48.89199995994568 seconds
          Klevel Aiyagari solved in 149 iterations, with a sup metric of 5.348097629409342e-7 and in 72.12100005149841 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (1.5777477210658017, 0.02273826182793973, 0.018006160618258042, 0.7541773689631258, 6.0586626241541275).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1034 iterations, with a sup metric of 9.881162554847833e-11 and in 54.824000120162964 seconds
          Klevel Aiyagari solved in 662 iterations, with a sup metric of 8.145654167710393e-7 and in 160.17600011825562 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (7.254188291002458, 0.02273826182793973, 0.020372211223098888, 1.3061466793140428, 0.10282560617333925).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1098 iterations, with a sup metric of 9.820944057992165e-11 and in 58.9539999961853 seconds
          Klevel Aiyagari solved in 1194 iterations, with a sup metric of 9.814231848702786e-7 and in 226.81200003623962 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (11.133097088962362, 0.021555236525519307, 0.020372211223098888, 1.523918469796002, 3.9095538067617195).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1065 iterations, with a sup metric of 9.825384950090665e-11 and in 52.17900013923645 seconds
          Klevel Aiyagari solved in 1003 iterations, with a sup metric of 6.232496745165196e-7 and in 188.5900001525879 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (9.738694180132015, 0.020963723874309097, 0.020372211223098888, 1.4522461576589751, 2.448916530467778).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1049 iterations, with a sup metric of 9.909584264278237e-11 and in 49.76800012588501 seconds
          Klevel Aiyagari solved in 785 iterations, with a sup metric of 6.980267652726821e-7 and in 158.60700011253357 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (8.491518048701133, 0.020667967548703994, 0.020372211223098888, 1.3823394276313823, 1.1682488215851805).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1041 iterations, with a sup metric of 9.987566329527908e-11 and in 49.03499984741211 seconds
          Klevel Aiyagari solved in 655 iterations, with a sup metric of 7.013655700032976e-7 and in 139.510999917984 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (7.713338044791395, 0.02052008938590144, 0.020372211223098888, 1.335325754812396, 0.3732282850599118).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1038 iterations, with a sup metric of 9.829648206505226e-11 and in 52.687000036239624 seconds
          Klevel Aiyagari solved in 646 iterations, with a sup metric of 6.11087260912515e-7 and in 149.99499988555908 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (7.261558854805246, 0.02052008938590144, 0.020446150304500164, 1.3066242802718473, 0.08699500219313983).
    Equilibrium found in 2219.390000104904 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.23887223437350258 and in 520.5299999713898 seconds
          Klevel Aiyagari solved in 156 iterations, with a sup metric of 8.034511084033538e-7 and in 544.8000001907349 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.976417403966664, 0.041666666666666484, -0.03409254588433786, 2.0384066416821778, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 579 iterations, with a sup metric of 9.650058530041861e-11 and in 31.124000072479248 seconds
          Klevel Aiyagari solved in 1590 iterations, with a sup metric of 4.4739519616404655e-7 and in 279.41299986839294 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (1.200703444144259, 0.041666666666666484, 0.003787060391164311, 0.6835603229371057, 8.55500173193181).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1170 iterations, with a sup metric of 9.868550421288091e-11 and in 62.55200004577637 seconds
          Klevel Aiyagari solved in 2272 iterations, with a sup metric of 8.775178269282353e-7 and in 437.75600004196167 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (22.597579496896426, 0.022726863528915398, 0.003787060391164311, 1.9662659846744817, 15.502351376726697).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 772 iterations, with a sup metric of 9.947420664957463e-11 and in 43.38300013542175 seconds
          Klevel Aiyagari solved in 265 iterations, with a sup metric of 4.1792402628955474e-7 and in 90.98600006103516 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (1.7483336516842465, 0.022726863528915398, 0.013256961960039854, 0.7825727622954471, 6.504359231961642).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 928 iterations, with a sup metric of 9.798029054763902e-11 and in 53.36899995803833 seconds
          Klevel Aiyagari solved in 338 iterations, with a sup metric of 3.97015775982429e-7 and in 109.90199995040894 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (3.0131291916822525, 0.022726863528915398, 0.017991912744477624, 0.9519768009916391, 4.6250161031513155).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1034 iterations, with a sup metric of 9.841372161645268e-11 and in 59.740999937057495 seconds
          Klevel Aiyagari solved in 694 iterations, with a sup metric of 7.46740835403549e-8 and in 168.97399997711182 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (8.631518209689226, 0.02035938813669651, 0.017991912744477624, 1.390501183290882, 1.2730354817081544).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 978 iterations, with a sup metric of 9.819167701152764e-11 and in 53.17300009727478 seconds
          Klevel Aiyagari solved in 385 iterations, with a sup metric of 7.734473412947226e-7 and in 113.70300006866455 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (5.6285564306832985, 0.02035938813669651, 0.01917565044058707, 1.1921280392157667, 1.867619233090391).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1005 iterations, with a sup metric of 9.872280770650832e-11 and in 54.74799990653992 seconds
          Klevel Aiyagari solved in 1108 iterations, with a sup metric of 7.985157482611675e-7 and in 228.80999994277954 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (7.681087634809557, 0.01976751928864179, 0.01917565044058707, 1.3333131217121423, 0.25428173720178204).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 991 iterations, with a sup metric of 9.904255193760036e-11 and in 52.99399995803833 seconds
          Klevel Aiyagari solved in 534 iterations, with a sup metric of 4.5494820110088323e-7 and in 142.08500003814697 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (6.5061336671425165, 0.01976751928864179, 0.01947158486461443, 1.2559612701088407, 0.9552249018387888).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 998 iterations, with a sup metric of 9.880274376428133e-11 and in 56.07200002670288 seconds
          Klevel Aiyagari solved in 697 iterations, with a sup metric of 9.293228793415656e-7 and in 165.23600006103516 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (7.100003394461464, 0.01976751928864179, 0.01961955207662811, 1.296083716284116, 0.34404596076739846).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1001 iterations, with a sup metric of 9.976730552807567e-11 and in 52.87100005149841 seconds
          Klevel Aiyagari solved in 764 iterations, with a sup metric of 9.761105713468504e-7 and in 169.65499997138977 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (7.31831417229061, 0.01976751928864179, 0.019693535682634947, 1.3102915848943784, 0.11710525633951185).
    Equilibrium found in 2451.3629999160767 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.23849623078513105 and in 544.7519998550415 seconds
          Klevel Aiyagari solved in 317 iterations, with a sup metric of 7.411741896726676e-7 and in 595.3150000572205 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.954204081932517, 0.041666666666666484, -0.03406639638316048, 2.0377538113301457, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 578 iterations, with a sup metric of 9.870149142443552e-11 and in 31.63599991798401 seconds
          Klevel Aiyagari solved in 853 iterations, with a sup metric of 8.901522544786714e-7 and in 167.50699996948242 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (2.479717212824293, 0.041666666666666484, 0.0038001351417530013, 0.8874926200918349, 7.273609763502703).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1169 iterations, with a sup metric of 9.883116547371174e-11 and in 62.21799993515015 seconds
          Klevel Aiyagari solved in 452 iterations, with a sup metric of 8.826427775802012e-7 and in 133.08699989318848 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (14.855920468341566, 0.022733400904209743, 0.0038001351417530013, 1.6906893526850453, 7.761397804881558).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 772 iterations, with a sup metric of 9.954526092315064e-11 and in 41.32099986076355 seconds
          Klevel Aiyagari solved in 338 iterations, with a sup metric of 9.495493357236111e-7 and in 99.39800000190735 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (3.44467334266194, 0.022733400904209743, 0.013266768022981372, 0.9989718947686513, 4.806663819070465).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 930 iterations, with a sup metric of 9.780976029105659e-11 and in 56.05800008773804 seconds
          Klevel Aiyagari solved in 457 iterations, with a sup metric of 5.500936494292275e-7 and in 135.2920000553131 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (8.515391577646945, 0.018000084463595557, 0.013266768022981372, 1.383737269947938, 0.8782414240812741).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 843 iterations, with a sup metric of 9.9253938401489e-11 and in 46.1710000038147 seconds
          Klevel Aiyagari solved in 1046 iterations, with a sup metric of 3.523070637874082e-7 and in 211.72900009155273 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (4.3825432816674486, 0.018000084463595557, 0.015633426243288465, 1.08943521909269, 3.551964256054376).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 884 iterations, with a sup metric of 9.919709498262819e-11 and in 47.2739999294281 seconds
          Klevel Aiyagari solved in 589 iterations, with a sup metric of 5.929601578706972e-7 and in 139.81699991226196 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (7.227930917320517, 0.018000084463595557, 0.01681675535344201, 1.3044427143882311, 0.5555696933942613).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 906 iterations, with a sup metric of 9.950795742952323e-11 and in 51.76400017738342 seconds
          Klevel Aiyagari solved in 2119 iterations, with a sup metric of 7.935359005929697e-7 and in 947.7430000305176 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (7.825499511476005, 0.017408419908518782, 0.01681675535344201, 1.3422837072074785, 0.11574360391779592).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 895 iterations, with a sup metric of 9.906031550599437e-11 and in 45.37999987602234 seconds
          Klevel Aiyagari solved in 665 iterations, with a sup metric of 6.514875361252776e-8 and in 141.02899980545044 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (7.607731923835442, 0.017408419908518782, 0.0171125876309804, 1.3287150258596003, 0.13875242096041873).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 901 iterations, with a sup metric of 9.805134482121503e-11 and in 43.96499991416931 seconds
          Klevel Aiyagari solved in 359 iterations, with a sup metric of 5.158083998901818e-7 and in 94.52600002288818 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (7.716337753143767, 0.017408419908518782, 0.01726050376974959, 1.335512681949516, 0.011746589335954738).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 903 iterations, with a sup metric of 9.993783578465809e-11 and in 43.931999921798706 seconds
          Klevel Aiyagari solved in 709 iterations, with a sup metric of 2.9411730503420235e-7 and in 144.70899987220764 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (7.782617629469243, 0.017334461839134188, 0.01726050376974959, 1.3396311072838165, 0.06370642616138777).
    Equilibrium found in 2810.1900000572205 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.2348562130555365 and in 505.4340000152588 seconds
          Klevel Aiyagari solved in 245 iterations, with a sup metric of 0.0 and in 542.1440000534058 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.99836934155363, 0.041666666666666484, -0.03411835024064995, 2.0390514252160674, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 567 iterations, with a sup metric of 9.684697488410166e-11 and in 28.51800012588501 seconds
          Klevel Aiyagari solved in 454 iterations, with a sup metric of 5.548844743541452e-7 and in 94.87100005149841 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (13.451447076610345, 0.003774158213008267, -0.03411835024064995, 1.6313111951076014, 3.693394158012058).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 376 iterations, with a sup metric of 9.542588941258145e-11 and in 18.91600012779236 seconds
          Klevel Aiyagari solved in 216 iterations, with a sup metric of 7.363396675911468e-8 and in 52.01100015640259 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (9.734227027638408, 0.003774158213008267, -0.015172096013820842, 1.4520063094135267, 4.831915774505038).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 450 iterations, with a sup metric of 9.943335044226842e-11 and in 22.59599995613098 seconds
          Klevel Aiyagari solved in 665 iterations, with a sup metric of 9.744283765108796e-7 and in 120.13499999046326 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (10.584529509636601, 0.003774158213008267, -0.005698968900406287, 1.4964483608584818, 1.1859246131342172).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 501 iterations, with a sup metric of 9.726619509820011e-11 and in 25.075000047683716 seconds
          Klevel Aiyagari solved in 204 iterations, with a sup metric of 9.752752495053855e-7 and in 55.29299998283386 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (11.262676244343762, -0.0009624053436990102, -0.005698968900406287, 1.5302801682493334, 0.5756351870445329).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 474 iterations, with a sup metric of 9.86091208687867e-11 and in 23.763999938964844 seconds
          Klevel Aiyagari solved in 399 iterations, with a sup metric of 8.578838477333229e-7 and in 82.73899984359741 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (10.814279251567479, -0.0009624053436990102, -0.003330687122052649, 1.5080616735542325, 0.39303249695381126).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 487 iterations, with a sup metric of 9.838174719334347e-11 and in 24.381999969482422 seconds
          Klevel Aiyagari solved in 1527 iterations, with a sup metric of 7.744533723383758e-7 and in 260.5789999961853 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (10.986073033154334, -0.0021465462328758298, -0.003330687122052649, 1.516642659330874, 0.04396585981264778).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 481 iterations, with a sup metric of 9.598011274647433e-11 and in 25.664999961853027 seconds
          Klevel Aiyagari solved in 1989 iterations, with a sup metric of 5.376155140713485e-7 and in 321.6859998703003 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (11.141419228044176, -0.0027386166774642394, -0.003330687122052649, 1.5243284655428215, 0.06801170881565177).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 477 iterations, with a sup metric of 9.934097988661961e-11 and in 24.53000020980835 seconds
          Klevel Aiyagari solved in 221 iterations, with a sup metric of 9.577524750215282e-7 and in 57.611000061035156 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (10.872941079875401, -0.0027386166774642394, -0.003034651899758444, 1.5110015381734656, 0.2670886061876594).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 479 iterations, with a sup metric of 9.762146646608016e-11 and in 24.51799988746643 seconds
          Klevel Aiyagari solved in 204 iterations, with a sup metric of 4.1693847271186686e-7 and in 55.257999897003174 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (10.872541559370838, -0.0027386166774642394, -0.0028866342886113418, 1.5109815503938309, 0.23409512022713486).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 480 iterations, with a sup metric of 9.679013146524085e-11 and in 25.592000007629395 seconds
          Klevel Aiyagari solved in 2265 iterations, with a sup metric of 6.080754892109426e-7 and in 361.6400001049042 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (10.947757836550716, -0.0027386166774642394, -0.0028126254830377903, 1.5147363222817174, 0.14224385204923884).
    Equilibrium found in 2004.0209999084473 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.822026036295938 and in 142.98799991607666 seconds
          Klevel Aiyagari solved in 116 iterations, with a sup metric of 1.1961925379398978e-7 and in 162.7369999885559 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.56288640933595, 0.041666666666666484, -0.033599390547751184, 2.026191820976974, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 604 iterations, with a sup metric of 9.86197790098231e-11 and in 9.223000049591064 seconds
          Klevel Aiyagari solved in 3758 iterations, with a sup metric of 4.303965885655583e-7 and in 573.2380001544952 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (0.9091959582255952, 0.041666666666666484, 0.00403363805945765, 0.6184387201049848, 8.801818167741828).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1222 iterations, with a sup metric of 9.99911264898401e-11 and in 17.7739999294281 seconds
          Klevel Aiyagari solved in 319 iterations, with a sup metric of 5.884853921103691e-7 and in 64.0310001373291 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (22.23385054871018, 0.022850152363062065, 0.00403363805945765, 1.9548131845709478, 15.151907341912567).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 804 iterations, with a sup metric of 9.992007221626409e-11 and in 11.59500002861023 seconds
          Klevel Aiyagari solved in 311 iterations, with a sup metric of 9.358276482271446e-7 and in 57.56599998474121 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (1.3922601237555696, 0.022850152363062065, 0.013441895211259857, 0.7209735170441024, 6.834926475482712).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 965 iterations, with a sup metric of 9.963940783563885e-11 and in 14.281999826431274 seconds
          Klevel Aiyagari solved in 1151 iterations, with a sup metric of 4.280863684089346e-7 and in 189.72399997711182 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (2.265799484150559, 0.022850152363062065, 0.018146023787160962, 0.8591316221921075, 5.353614114335487).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1075 iterations, with a sup metric of 9.987388693843968e-11 and in 15.447999954223633 seconds
          Klevel Aiyagari solved in 1068 iterations, with a sup metric of 8.027850327232351e-7 and in 175.2039999961853 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (8.757054472037401, 0.020498088075111512, 0.018146023787160962, 1.3977479771472034, 1.414433750682477).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1016 iterations, with a sup metric of 9.99698102077673e-11 and in 14.771000146865845 seconds
          Klevel Aiyagari solved in 212 iterations, with a sup metric of 8.90608840134544e-7 and in 46.30200004577637 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (3.314487158373338, 0.020498088075111512, 0.019322055931136235, 0.9852123713328098, 4.164430471394873).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1044 iterations, with a sup metric of 9.945466672434122e-11 and in 15.118000030517578 seconds
          Klevel Aiyagari solved in 1459 iterations, with a sup metric of 5.039129618144938e-7 and in 235.64400005340576 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (3.848479257622234, 0.020498088075111512, 0.019910072003123874, 1.0396426032602875, 3.561776031331104).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1059 iterations, with a sup metric of 9.847411774899228e-11 and in 15.188999891281128 seconds
          Klevel Aiyagari solved in 8862 iterations, with a sup metric of 4.939794416517099e-7 and in 1391.0279998779297 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (4.23490331476706, 0.020498088075111512, 0.020204080039117694, 1.076077705973637, 3.1414075610530245).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1067 iterations, with a sup metric of 9.871570227915072e-11 and in 16.253000020980835 seconds
          Klevel Aiyagari solved in 3960 iterations, with a sup metric of 3.1397324392802244e-7 and in 644.3050000667572 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (5.379716979747923, 0.020498088075111512, 0.020351084057114603, 1.1728794261920095, 1.9797172023341174).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1071 iterations, with a sup metric of 9.920597676682519e-11 and in 15.993000030517578 seconds
          Klevel Aiyagari solved in 1856 iterations, with a sup metric of 8.21841914997745e-7 and in 305.9940001964569 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (7.314869959386822, 0.020498088075111512, 0.020424586066113058, 1.3100695532835493, 0.03614960881268914).
    Equilibrium found in 3845.8040001392365 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.8224149234592915 and in 162.7960000038147 seconds
          Klevel Aiyagari solved in 106 iterations, with a sup metric of 7.139748049720357e-7 and in 180.63000011444092 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.783107582425004, 0.041666666666666484, -0.033863694624852676, 2.0327129236762476, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 602 iterations, with a sup metric of 9.738698736327933e-11 and in 9.603000164031982 seconds
          Klevel Aiyagari solved in 9790 iterations, with a sup metric of 4.076119234823574e-7 and in 1444.2310001850128 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (1.177073977713259, 0.041666666666666484, 0.003901486020906904, 0.6786866820760319, 8.55785023371774).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1218 iterations, with a sup metric of 9.880096740744193e-11 and in 15.467000007629395 seconds
          Klevel Aiyagari solved in 711 iterations, with a sup metric of 6.105891915608614e-7 and in 108.10199999809265 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (21.395237960333564, 0.022784076343786694, 0.003901486020906904, 1.9279427684013382, 14.306179850143462).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 802 iterations, with a sup metric of 9.78452874278446e-11 and in 9.993000030517578 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.0003638054521986366 and in 1886.4219999313354 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (1.8731492525653903, 0.022784076343786694, 0.013342781182346799, 0.8022431766996247, 6.36769120603125).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 963 iterations, with a sup metric of 9.809397738536063e-11 and in 12.671000003814697 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.0015513413610306816 and in 1422.1740000247955 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (3.7407534532970255, 0.022784076343786694, 0.018063428763066748, 1.0290707752480714, 3.8886899212897177).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1072 iterations, with a sup metric of 9.884359997158754e-11 and in 14.993000030517578 seconds
          Klevel Aiyagari solved in 6606 iterations, with a sup metric of 8.861182855539079e-7 and in 960.4630000591278 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (10.852420335907347, 0.02042375255342672, 0.018063428763066748, 1.5099742886340457, 3.501305434460366).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1014 iterations, with a sup metric of 9.878675655272673e-11 and in 14.689000129699707 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.00020775704762483627 and in 1357.8559999465942 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (4.51832198903405, 0.02042375255342672, 0.019243590658246733, 1.1014676782678923, 2.969836882816119).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1042 iterations, with a sup metric of 9.846701232163468e-11 and in 13.428999900817871 seconds
          Klevel Aiyagari solved in 477 iterations, with a sup metric of 9.871605521748406e-7 and in 76.65199995040894 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (5.153628302538672, 0.02042375255342672, 0.019833671605836725, 1.1548901766116466, 2.2654896695811964).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1056 iterations, with a sup metric of 9.958611713045684e-11 and in 13.450000047683716 seconds
          Klevel Aiyagari solved in 319 iterations, with a sup metric of 8.480427715723753e-7 and in 55.699000120162964 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (5.495953435031165, 0.02042375255342672, 0.020128712079631725, 1.1819401072593645, 1.8890346343526874).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1064 iterations, with a sup metric of 9.85771464456775e-11 and in 13.470999956130981 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.0010088820668232453 and in 1336.0250000953674 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (6.689392823119541, 0.02042375255342672, 0.020276232316529225, 1.2685838726015897, 0.6786267386022375).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1068 iterations, with a sup metric of 9.85558301636047e-11 and in 14.990000009536743 seconds
          Klevel Aiyagari solved in 1800 iterations, with a sup metric of 2.2053254962709215e-7 and in 255.82299995422363 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (8.982858261616679, 0.020349992434977975, 0.020276232316529225, 1.4106173167433962, 1.6232989900595074).
    Equilibrium found in 9084.108000040054 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.8315887066764844 and in 128.84600019454956 seconds
          Klevel Aiyagari solved in 104 iterations, with a sup metric of 5.142718154488737e-7 and in 143.7900002002716 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.802150187180594, 0.041666666666666484, -0.033886368191602394, 2.033275060956846, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 602 iterations, with a sup metric of 9.86091208687867e-11 and in 8.810999870300293 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.0026669806254234004 and in 1400.7579998970032 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (2.0494826494707032, 0.041666666666666484, 0.003890149237532045, 0.8286514650976886, 7.687497206352618).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1219 iterations, with a sup metric of 9.921308219418279e-11 and in 19.691999912261963 seconds
          Klevel Aiyagari solved in 1965 iterations, with a sup metric of 6.281408417433661e-7 and in 313.5649998188019 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (19.287723043215856, 0.022778407952099265, 0.003890149237532045, 1.85729598562197, 12.198054028608885).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 803 iterations, with a sup metric of 9.969269854082086e-11 and in 10.60699987411499 seconds
          Klevel Aiyagari solved in 4834 iterations, with a sup metric of 3.7898801777497743e-7 and in 684.2839999198914 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (3.6532059110169635, 0.022778407952099265, 0.013334278594815655, 1.0203347282018014, 4.588807586969837).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 966 iterations, with a sup metric of 9.968914582714206e-11 and in 12.467000007629395 seconds
          Klevel Aiyagari solved in 9421 iterations, with a sup metric of 9.486923061187239e-7 and in 1385.9400000572205 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (6.833289815592229, 0.022778407952099265, 0.01805634327345746, 1.2783409952550502, 0.7970149808337945).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1076 iterations, with a sup metric of 9.991296678890649e-11 and in 14.059000015258789 seconds
          Klevel Aiyagari solved in 302 iterations, with a sup metric of 6.202561505743439e-7 and in 55.43600010871887 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (9.530066050457304, 0.020417375612778364, 0.01805634327345746, 1.4409685487758168, 2.1782217175312937).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1018 iterations, with a sup metric of 9.970335668185726e-11 and in 13.836999893188477 seconds
          Klevel Aiyagari solved in 3555 iterations, with a sup metric of 9.173448849352604e-7 and in 537.0649998188019 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (7.739067936376419, 0.019236859443117912, 0.01805634327345746, 1.3369276065939957, 0.25011542402988685).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 992 iterations, with a sup metric of 9.816147894525784e-11 and in 15.79800009727478 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.041125424184939495 and in 1821.5169999599457 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (7.340041945296718, 0.019236859443117912, 0.018646601358287686, 1.311690729113929, 0.21904487798356875).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1005 iterations, with a sup metric of 9.85522774499259e-11 and in 17.792999982833862 seconds
          Klevel Aiyagari solved in 4547 iterations, with a sup metric of 5.386754323708453e-7 and in 1000.0220000743866 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (7.508736623506385, 0.019236859443117912, 0.0189417304007028, 1.3224645758344609, 0.015149025354540768).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1012 iterations, with a sup metric of 9.794831612452981e-11 and in 23.17799997329712 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.0024955759157913792 and in 2021.2669999599457 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (7.643573232885893, 0.019089294921910357, 0.0189417304007028, 1.3309651654858314, 0.13718747930490416).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1008 iterations, with a sup metric of 9.93125581771892e-11 and in 17.127000093460083 seconds
          Klevel Aiyagari solved in 1091 iterations, with a sup metric of 4.1449416472281574e-7 and in 221.22399997711182 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (7.564773259064907, 0.019015512661306577, 0.0189417304007028, 1.326009096923741, 0.04964591171965349).
    Equilibrium found in 9584.93499994278 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.9262386991595122 and in 201.68899989128113 seconds
          Klevel Aiyagari solved in 156 iterations, with a sup metric of 0.0 and in 229.5609998703003 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.999087385643822, 0.041666666666666484, -0.03411919366901435, 2.039072509852641, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 604 iterations, with a sup metric of 9.679368417891965e-11 and in 10.688999891281128 seconds
          Klevel Aiyagari solved in 3443 iterations, with a sup metric of 3.915231375323573e-7 and in 660.287999868393 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (6.945886501330253, 0.041666666666666484, 0.003773736498826067, 1.2858844357300387, 2.8122431700507464).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1226 iterations, with a sup metric of 9.905676279231557e-11 and in 21.817999839782715 seconds
          Klevel Aiyagari solved in 3013 iterations, with a sup metric of 4.4068138889068157e-7 and in 579.6099998950958 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (17.137264759191822, 0.022720201582746277, 0.003773736498826067, 1.7799135788122613, 10.04131762134634).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 809 iterations, with a sup metric of 9.836043091127067e-11 and in 13.56000018119812 seconds
          Klevel Aiyagari solved in 5480 iterations, with a sup metric of 7.696476058811114e-7 and in 954.0400002002716 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (9.995941827141454, 0.013246969040786172, 0.003773736498826067, 1.4659410731303855, 1.7418670121969324).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 691 iterations, with a sup metric of 9.9262820185686e-11 and in 10.950999975204468 seconds
          Klevel Aiyagari solved in 5830 iterations, with a sup metric of 5.577359002602773e-7 and in 1070.2490000724792 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (8.616878373145566, 0.013246969040786172, 0.00851035276980612, 1.3896516923186824, 0.33768512461971767).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 745 iterations, with a sup metric of 9.99591520667309e-11 and in 12.417999982833862 seconds
          Klevel Aiyagari solved in 1448 iterations, with a sup metric of 5.8500258132201e-7 and in 313.3340001106262 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (9.142387911247216, 0.010878660905296145, 0.00851035276980612, 1.4195851446198293, 0.5497619186577651).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 717 iterations, with a sup metric of 9.955058999366884e-11 and in 12.907999992370605 seconds
          Klevel Aiyagari solved in 2901 iterations, with a sup metric of 4.4865667721170123e-7 and in 556.2850000858307 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (8.938468175332506, 0.009694506837551133, 0.00851035276980612, 1.4081038610678178, 0.1679344573992303).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 704 iterations, with a sup metric of 9.865175343293231e-11 and in 12.851999998092651 seconds
          Klevel Aiyagari solved in 6651 iterations, with a sup metric of 7.156396646851967e-7 and in 1553.6070001125336 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (8.737416088965789, 0.009694506837551133, 0.009102429803678626, 1.3966187248558037, 0.1243491299085644).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 711 iterations, with a sup metric of 9.736211836752773e-11 and in 19.717999935150146 seconds
          Klevel Aiyagari solved in 1609 iterations, with a sup metric of 4.943941196284524e-7 and in 447.5680000782013 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (8.80830852430827, 0.009694506837551133, 0.009398468320614878, 1.4006875888228556, 0.00764740618447135).
    Equilibrium found in 6364.573999881744 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.39911539398963214 and in 873.4130001068115 seconds
          Klevel Aiyagari solved in 406 iterations, with a sup metric of 7.967971556399476e-7 and in 980.2549998760223 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.259352518295497, 0.041666666666666484, -0.033228660355968526, 2.017142073204425, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 596 iterations, with a sup metric of 9.697487257653847e-11 and in 51.30999994277954 seconds
          Klevel Aiyagari solved in 1461 iterations, with a sup metric of 8.23560060921275e-7 and in 440.7079999446869 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (2.0117893751618645, 0.041666666666666484, 0.004219003155348979, 0.8231323639859242, 7.665848813939435).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1207 iterations, with a sup metric of 9.9298347322474e-11 and in 117.86099982261658 seconds
          Klevel Aiyagari solved in 2305 iterations, with a sup metric of 2.532697212860918e-7 and in 756.9450001716614 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (21.882373074449745, 0.02294283491100773, 0.004219003155348979, 1.943631659023228, 14.810389981593733).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 795 iterations, with a sup metric of 9.912071163853398e-11 and in 89.7720000743866 seconds
          Klevel Aiyagari solved in 677 iterations, with a sup metric of 3.227827025339258e-7 and in 275.5039999485016 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (3.0584171691622486, 0.02294283491100773, 0.013580919033178354, 0.9571032573311155, 5.149680053084586).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 956 iterations, with a sup metric of 9.800160682971182e-11 and in 82.67300009727478 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.0007098608401750831 and in 2802.819000005722 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (4.078023322997062, 0.02294283491100773, 0.01826187697209304, 1.0615534543461362, 3.5273582477182117).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1068 iterations, with a sup metric of 9.806555567593023e-11 and in 91.79300022125244 seconds
          Klevel Aiyagari solved in 2420 iterations, with a sup metric of 1.2682516669599037e-7 and in 3903.0640001296997 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (19.07301819195026, 0.020602355941550386, 0.01826187697209304, 1.8498263703182958, 11.74228486915343).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1007 iterations, with a sup metric of 9.85522774499259e-11 and in 65.07100009918213 seconds
          Klevel Aiyagari solved in 8976 iterations, with a sup metric of 1.399338500831751e-7 and in 2493.292999982834 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (4.839302317533891, 0.020602355941550386, 0.01943211645682171, 1.1290203382021167, 2.626684421832386).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1036 iterations, with a sup metric of 9.944400858330482e-11 and in 118.39800000190735 seconds
          Klevel Aiyagari solved in 3418 iterations, with a sup metric of 7.452342983359306e-7 and in 1058.7620000839233 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (10.258922179880019, 0.02001723619918605, 0.01943211645682171, 1.4797099656890509, 2.861069045595915).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1021 iterations, with a sup metric of 9.956835356206284e-11 and in 63.854000091552734 seconds
          Klevel Aiyagari solved in 3623 iterations, with a sup metric of 5.698924074505478e-7 and in 886.0800001621246 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (6.435154856069484, 0.02001723619918605, 0.01972467632800388, 1.2510112488380296, 0.9966370315914821).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1029 iterations, with a sup metric of 9.831779834712506e-11 and in 75.5939998626709 seconds
          Klevel Aiyagari solved in 656 iterations, with a sup metric of 7.188358484617896e-7 and in 223.7260000705719 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (6.330178708685356, 0.02001723619918605, 0.019870956263594967, 1.2436258007008116, 1.0846119570003232).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1032 iterations, with a sup metric of 9.972822567760886e-11 and in 74.87600016593933 seconds
          Klevel Aiyagari solved in 1899 iterations, with a sup metric of 3.87815425813451e-7 and in 517.954999923706 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (8.182377022951979, 0.01994409623139051, 0.019870956263594967, 1.364006969010208, 0.7760630635352292).
    Equilibrium found in 14339.21899986267 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.3974028358097712 and in 755.2369999885559 seconds
          Klevel Aiyagari solved in 1067 iterations, with a sup metric of 8.733343310378954e-7 and in 1012.1619999408722 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.098611303622032, 0.041666666666666484, -0.033029237539194234, 2.012320261649279, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 597 iterations, with a sup metric of 9.725731331400311e-11 and in 46.454999923706055 seconds
          Klevel Aiyagari solved in 6912 iterations, with a sup metric of 8.236653792626367e-7 and in 5066.120000123978 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (2.8495161160826945, 0.041666666666666484, 0.004318714563736125, 0.9330342287395442, 6.810246260607842).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1210 iterations, with a sup metric of 9.891465424516355e-11 and in 112.26200008392334 seconds
          Klevel Aiyagari solved in 2052 iterations, with a sup metric of 4.354233221387482e-7 and in 706.4520001411438 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (20.637559005721076, 0.022992690615201303, 0.004318714563736125, 1.9030796796671516, 13.570924148773493).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 797 iterations, with a sup metric of 9.956835356206284e-11 and in 58.6560001373291 seconds
          Klevel Aiyagari solved in 682 iterations, with a sup metric of 1.5473924670444182e-7 and in 207.15300011634827 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (4.4129504709686795, 0.022992690615201303, 0.013655702589468714, 1.092150359632001, 3.7849082391059365).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 959 iterations, with a sup metric of 9.835510184075247e-11 and in 64.47099995613098 seconds
          Klevel Aiyagari solved in 2563 iterations, with a sup metric of 9.645711877596442e-7 and in 634.5160000324249 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (5.782562629734435, 0.022992690615201303, 0.01832419660233501, 1.2037693971335772, 1.8152883672770992).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1071 iterations, with a sup metric of 9.801937039810582e-11 and in 67.51500010490417 seconds
          Klevel Aiyagari solved in 2041 iterations, with a sup metric of 6.917711993015243e-7 and in 999.3980000019073 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (17.61084442210415, 0.020658443608768156, 0.01832419660233501, 1.7974666403446895, 10.286492507744429).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1011 iterations, with a sup metric of 9.795542155188741e-11 and in 97.57799983024597 seconds
          Klevel Aiyagari solved in 1242 iterations, with a sup metric of 8.403642834814804e-7 and in 428.4079999923706 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (10.102896531861166, 0.01949132010555158, 0.01832419660233501, 1.4715685575806345, 2.643850405400527).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 984 iterations, with a sup metric of 9.822187507779745e-11 and in 91.75999999046326 seconds
          Klevel Aiyagari solved in 201 iterations, with a sup metric of 8.033038405087468e-7 and in 146.3180000782013 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (6.013170318146984, 0.01949132010555158, 0.018907758353943295, 1.2208357428799295, 1.5147536022309414).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 997 iterations, with a sup metric of 9.837819447966467e-11 and in 85.9359998703003 seconds
          Klevel Aiyagari solved in 1799 iterations, with a sup metric of 2.4969105642408912e-8 and in 538.5640001296997 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (7.352809260664024, 0.01949132010555158, 0.019199539229747438, 1.3125116351024952, 0.14054597621588005).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1004 iterations, with a sup metric of 9.800515954339062e-11 and in 68.26700019836426 seconds
          Klevel Aiyagari solved in 1203 iterations, with a sup metric of 5.937705134504308e-8 and in 328.35700011253357 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (8.886551248236088, 0.01934542966764951, 0.019199539229747438, 1.4051540643497327, 1.4103828434883123).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1000 iterations, with a sup metric of 9.917400234371598e-11 and in 67.51600003242493 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 6.17920591273977e-5 and in 2065.8529999256134 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (7.850419364071882, 0.019272484448698474, 0.019199539229747438, 1.3438209349314605, 0.3656656336138129).
    Equilibrium found in 12133.390000104904 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.39433607637101886 and in 570.1000001430511 seconds
          Klevel Aiyagari solved in 1446 iterations, with a sup metric of 6.326681384324761e-7 and in 805.675999879837 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (24.09475290466072, 0.041666666666666484, -0.033024423831424475, 2.012204267263815, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 596 iterations, with a sup metric of 9.834799641339487e-11 and in 33.19700002670288 seconds
          Klevel Aiyagari solved in 5516 iterations, with a sup metric of 8.200273154249345e-7 and in 940.6280000209808 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (5.6645386164838625, 0.041666666666666484, 0.0043211214176210044, 1.194866015459915, 3.994792939731995).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1210 iterations, with a sup metric of 9.875122941593872e-11 and in 66.92700004577637 seconds
          Klevel Aiyagari solved in 623 iterations, with a sup metric of 9.905689920851565e-7 and in 166.04099988937378 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (18.83661849681184, 0.022993894042143744, 0.0043211214176210044, 1.8415394763860156, 11.77011265465488).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 798 iterations, with a sup metric of 9.953993185263244e-11 and in 44.257999897003174 seconds
          Klevel Aiyagari solved in 2361 iterations, with a sup metric of 8.079966349680533e-7 and in 421.4500000476837 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (8.350422563366799, 0.013657507729882374, 0.0043211214176210044, 1.3740262098521865, 0.1528107336244986).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 683 iterations, with a sup metric of 9.706191406166909e-11 and in 39.45900011062622 seconds
          Klevel Aiyagari solved in 258 iterations, with a sup metric of 5.984778251975344e-7 and in 84.15700006484985 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (7.133414304317242, 0.013657507729882374, 0.00898931457375169, 1.2982760795101376, 1.7459576439905753).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 736 iterations, with a sup metric of 9.836753633862827e-11 and in 43.79200005531311 seconds
          Klevel Aiyagari solved in 683 iterations, with a sup metric of 1.2721819411048646e-7 and in 153.2720000743866 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (7.6579778564730105, 0.013657507729882374, 0.011323411151817033, 1.3318675943153475, 0.869352475368844).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 766 iterations, with a sup metric of 9.823253321883385e-11 and in 41.65299987792969 seconds
          Klevel Aiyagari solved in 2017 iterations, with a sup metric of 5.205268022719577e-7 and in 367.9189999103546 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (7.950054524068124, 0.013657507729882374, 0.012490459440849703, 1.3499361036456627, 0.4097513706026561).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 782 iterations, with a sup metric of 9.798561961815722e-11 and in 41.62700009346008 seconds
          Klevel Aiyagari solved in 2727 iterations, with a sup metric of 1.4555194655492662e-7 and in 480.09299993515015 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (8.228746532612716, 0.013657507729882374, 0.013073983585366038, 1.3667846749032817, 0.04931090144825312).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 790 iterations, with a sup metric of 9.85380665952107e-11 and in 42.348999977111816 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.00941407732659068 and in 1759.1259999275208 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (8.247113478894166, 0.013365745657624207, 0.013073983585366038, 1.367882153204844, 0.009439891811650725).
    Equilibrium found in 5178.422999858856 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.37369815575993925 and in 651.2840001583099 seconds
          Klevel Aiyagari solved in 213 iterations, with a sup metric of NaN and in 694.2130000591278 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (25.0, 0.041666666666666484, -0.034120265586010584, 2.0390993072884185, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 541 iterations, with a sup metric of 9.970335668185726e-11 and in 34.984999895095825 seconds
          Klevel Aiyagari solved in 3999 iterations, with a sup metric of 2.6227877037375607e-7 and in 799.300999879837 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (15.363060031110415, 0.00377320054032795, -0.034120265586010584, 1.7112440689310324, 5.604832812835317).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 345 iterations, with a sup metric of 9.679013146524085e-11 and in 18.707000017166138 seconds
          Klevel Aiyagari solved in 4766 iterations, with a sup metric of 5.334481587416318e-7 and in 778.4409999847412 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (13.012022558780084, 0.00377320054032795, -0.015173532522841317, 1.6119223043029012, 1.5546245832271008).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 391 iterations, with a sup metric of 9.640643838793039e-11 and in 20.652999877929688 seconds
          Klevel Aiyagari solved in 4739 iterations, with a sup metric of 3.2711581867163503e-7 and in 765.3889999389648 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (14.573566818912735, -0.005700165991256684, -0.015173532522841317, 1.6790502058959265, 2.8028163805660693).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 360 iterations, with a sup metric of 9.562484137859428e-11 and in 19.819000005722046 seconds
          Klevel Aiyagari solved in 5140 iterations, with a sup metric of 9.728474573795602e-7 and in 853.2920000553131 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (14.285788065154733, -0.010436849257049, -0.015173532522841317, 1.6670379515804188, 1.2389555263758183).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 358 iterations, with a sup metric of 9.922018762154039e-11 and in 19.76800012588501 seconds
          Klevel Aiyagari solved in 2163 iterations, with a sup metric of 3.737027346167493e-7 and in 362.2810001373291 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (13.802013804989937, -0.012805190889945159, -0.015173532522841317, 1.6464906115628037, 0.02958335470649942).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 352 iterations, with a sup metric of 9.794121069717221e-11 and in 18.43499994277954 seconds
          Klevel Aiyagari solved in 6269 iterations, with a sup metric of 4.252509059988571e-7 and in 1014.8489999771118 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (13.176891183859773, -0.012805190889945159, -0.013989361706393238, 1.619245278578596, 0.9835207820242129).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 356 iterations, with a sup metric of 9.480061180511257e-11 and in 18.780999898910522 seconds
          Klevel Aiyagari solved in 3285 iterations, with a sup metric of 2.1084073528304066e-7 and in 538.5749998092651 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (13.411598289808241, -0.012805190889945159, -0.013397276298169197, 1.6295697989340046, 0.5526133798704631).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 357 iterations, with a sup metric of 9.811174095375463e-11 and in 19.788000106811523 seconds
          Klevel Aiyagari solved in 1775 iterations, with a sup metric of 8.639937707996333e-7 and in 299.49699997901917 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (13.38228899868718, -0.012805190889945159, -0.013101233594057179, 1.6282868652354363, 0.48548837949883605).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 358 iterations, with a sup metric of 9.657696864451282e-11 and in 18.746999979019165 seconds
          Klevel Aiyagari solved in 2330 iterations, with a sup metric of 9.579622876141343e-7 and in 389.10599994659424 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (13.510132174139464, -0.012805190889945159, -0.012953212242001168, 1.6338697388909993, 0.30983688833333467).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 358 iterations, with a sup metric of 9.811174095375463e-11 and in 19.55299997329712 seconds
          Klevel Aiyagari solved in 2510 iterations, with a sup metric of 4.499056433218068e-7 and in 414.54900002479553 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (13.742291171769967, -0.012805190889945159, -0.012879201565973164, 1.6439222236743052, 0.05387500407780266).
    Equilibrium found in 6909.569000005722 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.23917769197896632 and in 535.433000087738 seconds
          Klevel Aiyagari solved in 651 iterations, with a sup metric of 7.681712511915909e-7 and in 637.6749999523163 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (23.99263923235371, 0.041666666666666484, -0.03289656646152811, 2.0091301110999225, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 589 iterations, with a sup metric of 9.870149142443552e-11 and in 32.372000217437744 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.0010246737066777437 and in 1625.1280000209808 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (2.7409854419130366, 0.041666666666666484, 0.004385050102569187, 0.9200817074457679, 6.906914572668608).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1192 iterations, with a sup metric of 9.951683921372023e-11 and in 63.004000186920166 seconds
          Klevel Aiyagari solved in 1073 iterations, with a sup metric of 7.340106465140767e-7 and in 230.48800015449524 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (21.615014458700426, 0.023025858384617834, 0.004385050102569187, 1.935048991383469, 14.551933977808158).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 788 iterations, with a sup metric of 9.757528118825576e-11 and in 43.29100012779236 seconds
          Klevel Aiyagari solved in 8637 iterations, with a sup metric of 6.76070646475391e-7 and in 1432.2409999370575 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (3.9216509142367912, 0.023025858384617834, 0.01370545424359351, 1.0467158236247567, 4.269407962818862).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 947 iterations, with a sup metric of 9.885781082630274e-11 and in 52.89800000190735 seconds
          Klevel Aiyagari solved in 6557 iterations, with a sup metric of 9.901311681894525e-7 and in 1102.313000202179 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (5.089222487864757, 0.023025858384617834, 0.018365656314105673, 1.149673421957591, 2.5036253756159965).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1057 iterations, with a sup metric of 9.877965112536913e-11 and in 56.46399998664856 seconds
          Klevel Aiyagari solved in 6090 iterations, with a sup metric of 4.580700810161979e-7 and in 1014.8259999752045 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (19.860554116948006, 0.020695757349361753, 0.018365656314105673, 1.8769679808026303, 12.540442551341622).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 999 iterations, with a sup metric of 9.900702480081236e-11 and in 53.276000022888184 seconds
          Klevel Aiyagari solved in 2359 iterations, with a sup metric of 6.1951985161938e-7 and in 423.00999999046326 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (12.341872805572478, 0.01953070683173371, 0.018365656314105673, 1.5815289927307907, 4.887438238150823).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 972 iterations, with a sup metric of 9.953637913895363e-11 and in 53.85799980163574 seconds
          Klevel Aiyagari solved in 342 iterations, with a sup metric of 9.347219523264353e-7 and in 111.03499984741211 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (5.624232861741066, 0.01953070683173371, 0.018948181572919692, 1.191798294722079, 1.8988863349341694).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 985 iterations, with a sup metric of 9.992184857310349e-11 and in 54.12299990653992 seconds
          Klevel Aiyagari solved in 999 iterations, with a sup metric of 4.478880687396125e-7 and in 210.67399978637695 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (8.02890451089733, 0.019239444202326702, 0.018948181572919692, 1.3547408904013438, 0.5402567695853886).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 979 iterations, with a sup metric of 9.86037917982685e-11 and in 54.03200006484985 seconds
          Klevel Aiyagari solved in 3478 iterations, with a sup metric of 2.1518887963006085e-7 and in 609.0279998779297 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (8.372795426286888, 0.0190938128876232, 0.018948181572919692, 1.3753503643653031, 0.8669444115432334).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 976 iterations, with a sup metric of 9.7900354489866e-11 and in 53.157999992370605 seconds
          Klevel Aiyagari solved in 592 iterations, with a sup metric of 7.496038434103734e-7 and in 147.4579999446869 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (6.283985728337159, 0.0190938128876232, 0.019020997230271446, 1.2403511178759672, 1.230491242552473).
    Equilibrium found in 7543.914999961853 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.23856594147082433 and in 536.8199999332428 seconds
          Klevel Aiyagari solved in 616 iterations, with a sup metric of 8.250845707505773e-7 and in 632.8219997882843 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (23.836633688931993, 0.041666666666666484, -0.032699497950689806, 2.0044173167146386, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 590 iterations, with a sup metric of 9.940315237599862e-11 and in 31.54200005531311 seconds
          Klevel Aiyagari solved in 2968 iterations, with a sup metric of 5.119105624793537e-7 and in 502.5859999656677 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (4.487936933088371, 0.041666666666666484, 0.004483584357988339, 1.098795323311916, 5.142386871985045).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1195 iterations, with a sup metric of 9.92521620446496e-11 and in 66.63899993896484 seconds
          Klevel Aiyagari solved in 1410 iterations, with a sup metric of 7.528889281574737e-7 and in 286.1889998912811 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (20.41121175261619, 0.02307512551232741, 0.004483584357988339, 1.8955390587176482, 13.353405502904758).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 790 iterations, with a sup metric of 9.767120445758337e-11 and in 43.88100004196167 seconds
          Klevel Aiyagari solved in 960 iterations, with a sup metric of 6.73971926492859e-7 and in 195.7630000114441 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (5.810285963808373, 0.02307512551232741, 0.013779354935157875, 1.205843860930125, 2.370689558199518).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 950 iterations, with a sup metric of 9.980638537854247e-11 and in 51.32799983024597 seconds
          Klevel Aiyagari solved in 1285 iterations, with a sup metric of 8.022699018178367e-7 and in 252.64699983596802 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (13.138718386306675, 0.018427240223742643, 0.013779354935157875, 1.617554996359675, 5.553292168817014).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 862 iterations, with a sup metric of 9.943690315594722e-11 and in 45.83799982070923 seconds
          Klevel Aiyagari solved in 5725 iterations, with a sup metric of 8.120901997352564e-7 and in 951.5899999141693 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (6.246395188356264, 0.018427240223742643, 0.01610329757945026, 1.2376748857177342, 1.6275806828742194).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 904 iterations, with a sup metric of 9.844569603956188e-11 and in 49.561999797821045 seconds
          Klevel Aiyagari solved in 1875 iterations, with a sup metric of 6.080467114513664e-7 and in 346.3589999675751 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (6.6620706573889645, 0.018427240223742643, 0.01726526890159645, 1.2667161230824553, 1.0654221193361453).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 926 iterations, with a sup metric of 9.955591906418704e-11 and in 48.157999992370605 seconds
          Klevel Aiyagari solved in 6551 iterations, with a sup metric of 3.6444815853883344e-7 and in 1068.7909998893738 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (8.718981644141165, 0.01784625456266955, 0.01726526890159645, 1.3955572227567066, 1.0630625458586431).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 915 iterations, with a sup metric of 9.85505010930865e-11 and in 48.241000175476074 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.008084567359917095 and in 1596.3700001239777 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (7.152739927726035, 0.01784625456266955, 0.017555761732133, 1.2995411930661211, 0.5388294763255272).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 921 iterations, with a sup metric of 9.779554943634139e-11 and in 49.01300001144409 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.0009629878879059676 and in 1591.366000175476 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (7.8190796765990385, 0.017701008147401276, 0.017555761732133, 1.3418871803066852, 0.14536937805717987).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 918 iterations, with a sup metric of 9.812950452214864e-11 and in 48.51800012588501 seconds
          Klevel Aiyagari solved in 841 iterations, with a sup metric of 7.027623801638534e-7 and in 177.85899996757507 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (7.321187650760802, 0.017701008147401276, 0.017628384939767136, 1.3104767728731483, 0.3614436898832212).
    Equilibrium found in 7602.384999990463 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.23709711080255147 and in 544.5859999656677 seconds
          Klevel Aiyagari solved in 782 iterations, with a sup metric of 4.3291370539939125e-7 and in 671.2160000801086 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (23.79593344159311, 0.041666666666666484, -0.03264773650420593, 2.0031845519194578, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 590 iterations, with a sup metric of 9.820144697414435e-11 and in 32.54099988937378 seconds
          Klevel Aiyagari solved in 529 iterations, with a sup metric of 5.340945334893535e-7 and in 116.27900004386902 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (8.903686073909753, 0.041666666666666484, 0.004509465081230277, 1.4061288410013508, 0.7220299211749399).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 1196 iterations, with a sup metric of 9.849010496054689e-11 and in 64.00200009346008 seconds
          Klevel Aiyagari solved in 233 iterations, with a sup metric of 5.904830120222484e-7 and in 100.70499992370605 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (19.678944513860046, 0.02308806587394838, 0.004509465081230277, 1.8707709715746144, 12.622522507500179).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 790 iterations, with a sup metric of 9.982059623325767e-11 and in 44.931999921798706 seconds
          Klevel Aiyagari solved in 3202 iterations, with a sup metric of 2.599705721262793e-7 and in 559.680999994278 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (11.325346679049446, 0.013798765477589328, 0.004509465081230277, 1.533340178654642, 3.1470162466584544).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 676 iterations, with a sup metric of 9.770673159437138e-11 and in 37.11400008201599 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.0001402857984192395 and in 1636.7540001869202 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (9.91362050508442, 0.009154115279409802, 0.004509465081230277, 1.4615833883582412, 1.0598812383544853).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 630 iterations, with a sup metric of 9.832579195290236e-11 and in 34.90899991989136 seconds
          Klevel Aiyagari solved in 2931 iterations, with a sup metric of 8.121123643283746e-7 and in 505.4029998779297 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (9.445110583743682, 0.006831790180320039, 0.004509465081230277, 1.436330924128553, 0.21860809857695251).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 609 iterations, with a sup metric of 9.948131207693223e-11 and in 33.48300004005432 seconds
          Klevel Aiyagari solved in 2249 iterations, with a sup metric of 6.47550053188452e-7 and in 397.08200001716614 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (9.217265607842524, 0.006831790180320039, 0.005670627630775158, 1.4237598145514858, 0.20537740640137336).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 620 iterations, with a sup metric of 9.662670663601602e-11 and in 34.204999923706055 seconds
          Klevel Aiyagari solved in 10001 iterations, with a sup metric of 0.0006709740038270203 and in 1670.24799990654 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (9.274836248369157, 0.006831790180320039, 0.006251208905547598, 1.426954824777601, 0.048890705052057726).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 625 iterations, with a sup metric of 9.733014394441852e-11 and in 34.496999979019165 seconds
          Klevel Aiyagari solved in 5001 iterations, with a sup metric of 5.724105519645607e-7 and in 849.0810000896454 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (9.188375543566778, 0.006831790180320039, 0.006541499542933819, 1.4221516823110298, 0.08653025224483812).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 627 iterations, with a sup metric of 9.947154211431553e-11 and in 1239.8580000400543 seconds
          Klevel Aiyagari solved in 3078 iterations, with a sup metric of 6.675659235606234e-7 and in 2014.3519999980927 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (9.426780439302645, 0.006686644861626929, 0.006541499542933819, 1.43532680379417, 0.17612821820398317).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 626 iterations, with a sup metric of 9.838885262070107e-11 and in 44.954999923706055 seconds
          Klevel Aiyagari solved in 4724 iterations, with a sup metric of 2.455287497080271e-7 and in 915.3559999465942 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (9.312470320658493, 0.006614072202280373, 0.006541499542933819, 1.429036556031592, 0.04970433079994585).
    Equilibrium found in 9436.242999792099 seconds
      
      
       Equilibrium(Aiyagari)...
    In iteration 0 we have (K0, r0, w0) = (5.446807380113248, 0.041666666666666484, 1.1781242629578268).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 10001 iterations, with a sup metric of 0.22389021877461346 and in 623.4530000686646 seconds
          Klevel Aiyagari solved in 199 iterations, with a sup metric of NaN and in 656.8480000495911 seconds
    In iteration 1 we have (K, r1, r2, w, sup) = (25.0, 0.041666666666666484, -0.034120265586010584, 2.0390993072884185, Inf).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 582 iterations, with a sup metric of 9.913492249324918e-11 and in 31.697999954223633 seconds
          Klevel Aiyagari solved in 5661 iterations, with a sup metric of 6.106242931158477e-7 and in 1129.555999994278 seconds
    In iteration 2 we have (K, r1, r2, w, sup) = (19.68508616435288, 0.00377320054032795, -0.034120265586010584, 1.8709811378620398, 9.926858946077784).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 404 iterations, with a sup metric of 9.822542779147625e-11 and in 24.776000022888184 seconds
          Klevel Aiyagari solved in 3267 iterations, with a sup metric of 9.937986091164434e-7 and in 668.7560000419617 seconds
    In iteration 3 we have (K, r1, r2, w, sup) = (17.213604227779573, -0.015173532522841317, -0.034120265586010584, 1.7827638818368794, 2.6469570857723888).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 346 iterations, with a sup metric of 9.367795428261161e-11 and in 21.95900011062622 seconds
          Klevel Aiyagari solved in 1023 iterations, with a sup metric of 6.04252851870776e-7 and in 225.57500004768372 seconds
    In iteration 4 we have (K, r1, r2, w, sup) = (16.637076333952756, -0.015173532522841317, -0.02464689905442595, 1.7610338232375091, 2.0079650588913545).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 373 iterations, with a sup metric of 9.913492249324918e-11 and in 21.281999826431274 seconds
          Klevel Aiyagari solved in 3444 iterations, with a sup metric of 4.7522495959745123e-7 and in 700.2699999809265 seconds
    In iteration 5 we have (K, r1, r2, w, sup) = (16.9606814075275, -0.019910215788633633, -0.02464689905442595, 1.7732891657636418, 0.5605755782407407).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 359 iterations, with a sup metric of 9.731593308970332e-11 and in 23.871999979019165 seconds
          Klevel Aiyagari solved in 6287 iterations, with a sup metric of 9.217304025225204e-7 and in 1262.5220000743866 seconds
    In iteration 6 we have (K, r1, r2, w, sup) = (16.65978453234661, -0.019910215788633633, -0.02227855742152979, 1.7618987638324666, 0.8037971236715542).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 366 iterations, with a sup metric of 9.822542779147625e-11 and in 24.169000148773193 seconds
          Klevel Aiyagari solved in 1764 iterations, with a sup metric of 6.983191738681073e-7 and in 316.6419999599457 seconds
    In iteration 7 we have (K, r1, r2, w, sup) = (16.80502135331148, -0.019910215788633633, -0.021094386605081714, 1.7674129740504398, 0.1131274578257262).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 370 iterations, with a sup metric of 9.549694368615746e-11 and in 20.372999906539917 seconds
          Klevel Aiyagari solved in 1927 iterations, with a sup metric of 6.460807834524917e-7 and in 405.8489999771118 seconds
    In iteration 8 we have (K, r1, r2, w, sup) = (16.6929832467795, -0.020502301196857673, -0.021094386605081714, 1.7631619245489751, 0.03715845590540212).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 368 iterations, with a sup metric of 9.640643838793039e-11 and in 29.32800006866455 seconds
          Klevel Aiyagari solved in 5702 iterations, with a sup metric of 8.108377282096664e-7 and in 1380.138000011444 seconds
    In iteration 9 we have (K, r1, r2, w, sup) = (16.804289050403348, -0.02079834390096969, -0.021094386605081714, 1.7673852473439076, 0.018142600231708883).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 367 iterations, with a sup metric of 9.731593308970332e-11 and in 30.27400016784668 seconds
          Klevel Aiyagari solved in 336 iterations, with a sup metric of 7.863377961218284e-7 and in 117.67300009727478 seconds
    In iteration 10 we have (K, r1, r2, w, sup) = (16.730348592954094, -0.02079834390096969, -0.020946365253025703, 1.7645816976473492, 0.1215870733329325).
       Klevel(Aiyagari)...
       Kpolicy(Aiyagari)...
          Kpolicy Aiyagari solved in 368 iterations, with a sup metric of 9.390532795805484e-11 and in 29.930000066757202 seconds
          Klevel Aiyagari solved in 3553 iterations, with a sup metric of 4.905663709369869e-7 and in 859.4839999675751 seconds
    In iteration 11 we have (K, r1, r2, w, sup) = (16.67688170907274, -0.02079834390096969, -0.020872354576997695, 1.7625494864558204, 0.14210659432182382).
    Equilibrium found in 7723.407999992371 seconds
    


```julia
TABLEA = [(AE11[4],AE11[2],AE11[1]) (AE12[4],AE12[2],AE12[1]) (AE13[4],AE13[2],AE13[1]); (AE21[4],AE21[2],AE21[1]) (AE22[4],AE22[2],AE22[1]) (AE23[4],AE23[2],AE23[1]); (AE31[4],AE31[2],AE31[1]) (AE32[4],AE32[2],AE32[1]) (AE33[4],AE33[2],AE33[1]); (AE41[4],AE41[2],AE41[1]) (AE42[4],AE42[2],AE42[1]) (AE43[4],AE43[2],AE43[1])];
TABLEB = [(BE11[4],BE11[2],BE11[1]) (BE12[4],BE12[2],BE12[1]) (BE13[4],BE13[2],BE13[1]); (BE21[4],BE21[2],BE21[1]) (BE22[4],BE22[2],BE22[1]) (BE23[4],BE23[2],BE23[1]); (BE31[4],BE31[2],BE31[1]) (BE32[4],BE32[2],BE32[1]) (BE33[4],BE33[2],BE33[1]); (BE41[4],BE41[2],BE41[1]) (BE42[4],BE42[2],BE42[1]) (BE43[4],BE43[2],BE43[1])];
```

In the following graphs we see the values from our estimation for Table II.A and II.B of Aiyagari in the following order: savings, interest rate, capital

- As the persistency in the shock increases (an increase in $\lambda$) there is a higher level of savings in the economy, a higher level of capital and a lower interest rate.
- As the risk aversion coefficient $\mu$ increases we have more savings, a higher level of capital and lower interest rates.
- As the variance of income shock increases there is an increase in the level of savings, the stationary level of capital, and a reduction in the level of interest rates.


```julia
@show TABLEA[:,1]
```

    TABLEA[:, 1] = Tuple{Float64,Float64,Float64}[(0.28635738531582133, 0.02057362399868507, 7.781328567534022), (0.28677945903236973, 0.020425602646629064, 6.504934850431803), (0.2876273501704559, 0.020129559942517042, 7.123473249835518), (0.2950417724383304, 0.017613296456249324, 7.487291804160057)]
    TABLEA[:, 2] = Tuple{Float64,Float64,Float64}[(0.28634418282331964, 0.020578261154235507, 8.388679608415359), (0.28760605383027227, 0.020136974227239396, 6.9737677252374475), (0.29147367336190255, 0.018808237697135154, 7.3050439106855745), (0.3278510224839727, 0.007844777124060714, 9.01764311370713)]
    TABLEA[:, 3] = Tuple{Float64,Float64,Float64}[(0.2867207943031512, 0.020446150304500164, 7.261558854805246), (0.2888853304559496, 0.019693535682634947, 7.31831417229061), (0.2958869803749272, 0.017334461839134188, 7.782617629469243), (0.37311801548154866, -0.0028126254830377903, 10.947757836550716)]
    




    4-element Array{Tuple{Float64,Float64,Float64},1}:
     (0.2867207943031512, 0.020446150304500164, 7.261558854805246)    
     (0.2888853304559496, 0.019693535682634947, 7.31831417229061)     
     (0.2958869803749272, 0.017334461839134188, 7.782617629469243)    
     (0.37311801548154866, -0.0028126254830377903, 10.947757836550716)




```julia
@show TABLEA[:,2]
```

    TABLEA[:, 2] = Tuple{Float64,Float64,Float64}[(0.28634418282331964, 0.020578261154235507, 8.388679608415359), (0.28760605383027227, 0.020136974227239396, 6.9737677252374475), (0.29147367336190255, 0.018808237697135154, 7.3050439106855745), (0.3278510224839727, 0.007844777124060714, 9.01764311370713)]
    




    4-element Array{Tuple{Float64,Float64,Float64},1}:
     (0.28634418282331964, 0.020578261154235507, 8.388679608415359) 
     (0.28760605383027227, 0.020136974227239396, 6.9737677252374475)
     (0.29147367336190255, 0.018808237697135154, 7.3050439106855745)
     (0.3278510224839727, 0.007844777124060714, 9.01764311370713)   




```julia
@show TABLEA[:,3]
```

    TABLEA[:, 3] = Tuple{Float64,Float64,Float64}[(0.2867207943031512, 0.020446150304500164, 7.261558854805246), (0.2888853304559496, 0.019693535682634947, 7.31831417229061), (0.2958869803749272, 0.017334461839134188, 7.782617629469243), (0.37311801548154866, -0.0028126254830377903, 10.947757836550716)]
    




    4-element Array{Tuple{Float64,Float64,Float64},1}:
     (0.2867207943031512, 0.020446150304500164, 7.261558854805246)    
     (0.2888853304559496, 0.019693535682634947, 7.31831417229061)     
     (0.2958869803749272, 0.017334461839134188, 7.782617629469243)    
     (0.37311801548154866, -0.0028126254830377903, 10.947757836550716)




```julia
@show TABLEB[:,1]
```

    TABLEB[:, 1] = Tuple{Float64,Float64,Float64}[(0.286782362050663, 0.020424586066113058, 7.314869959386822), (0.28699553733061844, 0.020349992434977975, 8.982858261616679), (0.29086351447286407, 0.019015512661306577, 7.564773259064907), (0.3221531704180087, 0.009398468320614878, 8.80830852430827)]
    




    4-element Array{Tuple{Float64,Float64,Float64},1}:
     (0.286782362050663, 0.020424586066113058, 7.314869959386822)  
     (0.28699553733061844, 0.020349992434977975, 8.982858261616679)
     (0.29086351447286407, 0.019015512661306577, 7.564773259064907)
     (0.3221531704180087, 0.009398468320614878, 8.80830852430827)  




```julia
@show TABLEB[:,2]
```

    TABLEB[:, 2] = Tuple{Float64,Float64,Float64}[(0.2881610929106033, 0.01994409623139051, 8.182377022951979), (0.2901105997290026, 0.019272484448698474, 7.850419364071882), (0.3084643066592186, 0.013365745657624207, 8.247113478894166), (0.42907713662416536, -0.012879201565973164, 13.742291171769967)]
    




    4-element Array{Tuple{Float64,Float64,Float64},1}:
     (0.2881610929106033, 0.01994409623139051, 8.182377022951979)    
     (0.2901105997290026, 0.019272484448698474, 7.850419364071882)   
     (0.3084643066592186, 0.013365745657624207, 8.247113478894166)   
     (0.42907713662416536, -0.012879201565973164, 13.742291171769967)




```julia
@show TABLEB[:,3]
```

    TABLEB[:, 3] = Tuple{Float64,Float64,Float64}[(0.2908474041422361, 0.019020997230271446, 6.283985728337159), (0.2949961736821567, 0.017628384939767136, 7.321187650760802), (0.33250947874543824, 0.006614072202280373, 9.312470320658493), (0.4870818006359508, -0.020872354576997695, 16.67688170907274)]
    




    4-element Array{Tuple{Float64,Float64,Float64},1}:
     (0.2908474041422361, 0.019020997230271446, 6.283985728337159) 
     (0.2949961736821567, 0.017628384939767136, 7.321187650760802) 
     (0.33250947874543824, 0.006614072202280373, 9.312470320658493)
     (0.4870818006359508, -0.020872354576997695, 16.67688170907274)



The values do not match properly Aiyagari's estimation. 

As the risk aversion parameter $\sigma$ increases see that there is a lowe concentration of agents at the borrowing constraint. Agents accumulate more assets. 


```julia
plot(ash(vec(AE41[15]); m =20); hist = false, label="sigma = 1", title = "Change of Asset PDF as sigma increases")
plot!(ash(vec(AE42[15]); m =20); hist = false, label="sigma = 3")
plot!(ash(vec(AE43[15]); m =20); hist = false, label="sigma = 5")
```


    UndefVarError: AE41 not defined

    

    Stacktrace:

     [1] top-level scope at In[2]:1


This is also reflected in the CDF


```julia
plot(AE41[7], [sum(AE41[5][:,1:s]) for s in 1: size(AE41[7])[1]], legend=:bottomright, label="sigma = 1", title = "Change of Asset CDF as sigma increases")
plot!(AE42[7], [sum(AE42[5][:,1:s]) for s in 1: size(AE42[7])[1]], label="sigma = 3")
plot!(AE43[7], [sum(AE43[5][:,1:s]) for s in 1: size(AE43[7])[1]], label="sigma = 5")
```




![svg](Aiyagari%281994%29_files/Aiyagari%281994%29_86_0.svg)



As the persistency of the shock increases there is going to be a higher concentration of agents at the level of the borrowing constraint and a higher number of agents with high levels of welath


```julia
plot(ash(vec(AE12[15]); m =20); hist = false, label="lambda = 0.0", title = "Change of Asset PDF as persistency of productivity shock increases")
plot!(ash(vec(AE22[15]); m =20); hist = false, label="lambda = 0.3",)
plot!(ash(vec(AE32[15]); m =20); hist = false, label="lambda = 0.6",)
plot!(ash(vec(AE42[15]); m =20); hist = false, label="lambda = 0.9",)
```




![svg](Aiyagari%281994%29_files/Aiyagari%281994%29_88_0.svg)




```julia
plot(AE12[7], [sum(AE12[5][:,1:s]) for s in 1: size(AE12[7])[1]], legend=:bottomright, label="lambda = 0.0", title = "Change of Asset CDF as lambda increases")
plot!(AE22[7], [sum(AE22[5][:,1:s]) for s in 1: size(AE22[7])[1]], label="lambda = 0.3")
plot!(AE32[7], [sum(AE32[5][:,1:s]) for s in 1: size(AE32[7])[1]], label="lambda = 0.6")
plot!(AE42[7], [sum(AE42[5][:,1:s]) for s in 1: size(AE42[7])[1]], label="lambda = 0.9")
```


    UndefVarError: AE12 not defined

    

    Stacktrace:

     [1] top-level scope at In[1]:1


As the variance of the income shock increases we have that the number of agents with high levels of wealth is going to increase


```julia
plot(ash(vec(AE32[15]); m =20); hist = false, label="sigma_epsilon = 0.2", title = "Change of Asset PDF as variance of income shock changes")
plot!(ash(vec(BE32[15]); m =20); hist = false, label="sigma_epsilon = 0.4",)
```




![svg](Aiyagari%281994%29_files/Aiyagari%281994%29_91_0.svg)




```julia
plot(AE32[7], [sum(AE32[5][:,1:s]) for s in 1: size(AE32[7])[1]], legend=:bottomright, label="sigma_epsilon = 0.2", title = "Change of Asset CDF as sigma_epsilon increases")
plot!(BE32[7], [sum(BE32[5][:,1:s]) for s in 1: size(BE32[7])[1]], label="sigma_epsilon = 0.4")
```




![svg](Aiyagari%281994%29_files/Aiyagari%281994%29_92_0.svg)


