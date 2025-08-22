# Fix BNode Calibration Issues
# Addresses: 0.5% coverage instead of 50% (severe under-coverage)

using Turing, MCMCChains, CSV, DataFrames, Statistics, Random
using BSON, Plots

println("🔧 Fixing BNode Calibration Issues")
println("=" ^ 50)

# Load data
train_data = CSV.read("data/training_roadmap.csv", DataFrame)
val_data = CSV.read("data/validation_roadmap.csv", DataFrame)

# IMPROVED BNode Model with Better Calibration
@model function improved_bnode_model(Y, scenarios, width=5)
    # BROADER PRIORS for better calibration
    σ ~ Gamma(2.0, 0.5)  # Broader observation noise prior
    α ~ Normal(0.0, 2.0)  # Broader neural network priors
    β ~ Normal(0.0, 2.0)
    
    # Neural network parameters with proper regularization
    nn_params ~ MvNormal(zeros(width*8), 1.0)  # 8 = 4 inputs × 2 outputs
    
    # IMPROVED LIKELIHOOD with Student-t for robustness
    for sid in unique(scenarios)
        sdf = Y[scenarios .== sid, :]
        Yhat = solve_scenario_improved(nn_params, width, sdf)
        
        if Yhat !== nothing
            Y_obs = Matrix(sdf[:, [:x1, :x2]])
            Y_vec = vec(Y_obs)
            Yhat_vec = vec(Yhat)
            
            # Student-t likelihood for better calibration
            Y_vec ~ MvTDist(3.0, Yhat_vec, σ)  # Degrees of freedom = 3
        end
    end
end

# IMPROVED SOLVER with better numerical stability
function solve_scenario_improved(θ, width::Int, sdf::DataFrame)
    T = Vector{Float64}(sdf.time)
    Y = Matrix(sdf[:, [:x1, :x2]])
    
    function rhs!(du, x, p, t)
        # More stable neural network implementation
        x1, x2 = x
        u = interpolate_control(t, sdf)
        d = interpolate_demand(t, sdf)
        Pgen = interpolate_generation(t, sdf)
        Pload = interpolate_load(t, sdf)
        
        # Improved neural network with skip connections
        du[1] = neural_ode1_improved(x1, x2, u, d, θ, width)
        du[2] = neural_ode2_improved(x1, x2, Pgen, Pload, θ, width)
    end
    
    prob = ODEProblem(rhs!, Y[1, :], (minimum(T), maximum(T)))
    sol = solve(prob, Rodas5(); saveat=T, abstol=1e-8, reltol=1e-8, maxiters=10000)
    
    return sol.u !== nothing ? hcat([u for u in sol.u]...) : nothing
end

# IMPROVED NEURAL NETWORKS with skip connections
function neural_ode1_improved(x1, x2, u, d, θ, width)
    # Skip connection + neural network
    linear_term = 0.1 * u - 0.1 * d  # Physics-inspired linear term
    nn_term = neural_net1(x1, x2, u, d, θ, width)
    return linear_term + nn_term
end

function neural_ode2_improved(x1, x2, Pgen, Pload, θ, width)
    # Skip connection + neural network
    linear_term = -0.1 * x2 + 0.5 * (Pgen - Pload)  # Physics-inspired
    nn_term = neural_net2(x1, x2, Pgen, Pload, θ, width)
    return linear_term + nn_term
end

# CALIBRATION DIAGNOSTICS
function compute_calibration_metrics(chains, test_data)
    # Extract posterior samples
    σ_samples = chains[:σ]
    nn_samples = chains[:nn_params]
    
    # Compute predictive intervals
    coverage_50 = 0.0
    coverage_90 = 0.0
    nll_values = Float64[]
    
    for i in 1:min(100, length(σ_samples))  # Sample 100 posterior draws
        σ = σ_samples[i]
        θ = nn_samples[i, :]
        
        # Generate predictions
        Y_pred = solve_scenario_improved(θ, 5, test_data)
        if Y_pred !== nothing
            Y_true = Matrix(test_data[:, [:x1, :x2]])
            
            # Compute coverage
            residuals = abs.(Y_true - Y_pred)
            coverage_50 += mean(residuals .< 0.674 * σ) / 100
            coverage_90 += mean(residuals .< 1.645 * σ) / 100
            
            # Compute NLL
            nll = -logpdf(MvNormal(Y_pred, σ), Y_true)
            push!(nll_values, nll)
        end
    end
    
    return coverage_50, coverage_90, mean(nll_values)
end

# MAIN EXECUTION
println("📊 Training improved BNode model...")

# Prepare data
scenarios = train_data.scenario
Y_matrix = Matrix(train_data[:, [:x1, :x2, :u, :d, :Pgen, :Pload]])

# Run MCMC with improved settings
chain = sample(improved_bnode_model(Y_matrix, scenarios), NUTS(0.65), 1000)

println("✅ Improved BNode training completed")
println("📊 Computing calibration metrics...")

# Test calibration
coverage_50, coverage_90, mean_nll = compute_calibration_metrics(chain, val_data)

println("📈 IMPROVED CALIBRATION RESULTS:")
println("  50% Coverage: $(round(coverage_50, digits=3)) (target: 0.5)")
println("  90% Coverage: $(round(coverage_90, digits=3)) (target: 0.9)")
println("  Mean NLL: $(round(mean_nll, digits=3))")

# Save improved model
BSON.bson("checkpoints/improved_bnode_posterior.bson", 
          chain=chain, 
          calibration_metrics=(coverage_50=coverage_50, coverage_90=coverage_90, mean_nll=mean_nll))

println("💾 Saved improved BNode model to checkpoints/improved_bnode_posterior.bson")
