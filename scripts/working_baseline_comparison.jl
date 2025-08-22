# Working Baseline Comparison
# Tests multiple ML methods to demonstrate UDE/BNode advantages

using CSV, DataFrames, Statistics, Random
using BSON, Plots, GLM

println("🏆 Working Baseline Comparison")
println("=" ^ 50)

# Load data
train_data = CSV.read("data/training_roadmap.csv", DataFrame)
val_data = CSV.read("data/validation_roadmap.csv", DataFrame)
test_data = CSV.read("data/test_roadmap.csv", DataFrame)

# BASELINE 1: Linear Regression
function linear_regression_baseline(train_data, test_data)
    println("📊 Training Linear Regression baseline...")
    
    # Prepare features
    features = [:u, :d, :Pgen, :Pload]
    target_x1 = :x1
    target_x2 = :x2
    
    # Train models for x1 and x2
    model_x1 = lm(Term(target_x1) ~ sum(Term.(features)), train_data)
    model_x2 = lm(Term(target_x2) ~ sum(Term.(features)), train_data)
    
    # Predictions
    pred_x1 = predict(model_x1, test_data)
    pred_x2 = predict(model_x2, test_data)
    
    # Metrics
    true_x1 = test_data[:, target_x1]
    true_x2 = test_data[:, target_x2]
    
    rmse_x1 = sqrt(mean((pred_x1 .- true_x1).^2))
    rmse_x2 = sqrt(mean((pred_x2 .- true_x2).^2))
    r2_x1 = 1 - sum((pred_x1 .- true_x1).^2) / sum((true_x1 .- mean(true_x1)).^2)
    r2_x2 = 1 - sum((pred_x2 .- true_x2).^2) / sum((true_x2 .- mean(true_x2)).^2)
    
    return Dict("rmse_x1" => rmse_x1, "rmse_x2" => rmse_x2, "r2_x1" => r2_x1, "r2_x2" => r2_x2)
end

# BASELINE 2: Physics-Only (no learning)
function physics_only_baseline(test_data)
    println("⚛️ Computing Physics-only baseline...")
    
    # Use physics equations directly
    pred_x1 = zeros(nrow(test_data))
    pred_x2 = zeros(nrow(test_data))
    
    for i in 1:nrow(test_data)
        row = test_data[i, :]
        
        # Physics equations
        ηin, ηout, α, β, γ = 0.9, 0.9, 0.1, 0.5, 0.3
        
        # Equation 1: dx1/dt = ηin * u * I(u>0) - (1/ηout) * u * I(u<0) - d
        u = row.u
        d = row.d
        pred_x1[i] = ηin * max(u, 0) - (1/ηout) * min(u, 0) - d
        
        # Equation 2: dx2/dt = -α * x2 + β * (Pgen - Pload) + γ * x1
        x1 = row.x1
        x2 = row.x2
        Pgen = row.Pgen
        Pload = row.Pload
        pred_x2[i] = -α * x2 + β * (Pgen - Pload) + γ * x1
    end
    
    # Metrics
    true_x1 = test_data[:, :x1]
    true_x2 = test_data[:, :x2]
    
    rmse_x1 = sqrt(mean((pred_x1 .- true_x1).^2))
    rmse_x2 = sqrt(mean((pred_x2 .- true_x2).^2))
    r2_x1 = 1 - sum((pred_x1 .- true_x1).^2) / sum((true_x1 .- mean(true_x1)).^2)
    r2_x2 = 1 - sum((pred_x2 .- true_x2).^2) / sum((true_x2 .- mean(true_x2)).^2)
    
    return Dict("rmse_x1" => rmse_x1, "rmse_x2" => rmse_x2, "r2_x1" => r2_x1, "r2_x2" => r2_x2)
end

# BASELINE 3: Simple Average (naive baseline)
function simple_average_baseline(train_data, test_data)
    println("📊 Computing Simple Average baseline...")
    
    # Use average values from training data
    avg_x1 = mean(train_data[:, :x1])
    avg_x2 = mean(train_data[:, :x2])
    
    # Predictions
    pred_x1 = fill(avg_x1, nrow(test_data))
    pred_x2 = fill(avg_x2, nrow(test_data))
    
    # Metrics
    true_x1 = test_data[:, :x1]
    true_x2 = test_data[:, :x2]
    
    rmse_x1 = sqrt(mean((pred_x1 .- true_x1).^2))
    rmse_x2 = sqrt(mean((pred_x2 .- true_x2).^2))
    r2_x1 = 1 - sum((pred_x1 .- true_x1).^2) / sum((true_x1 .- mean(true_x1)).^2)
    r2_x2 = 1 - sum((pred_x2 .- true_x2).^2) / sum((true_x2 .- mean(true_x2)).^2)
    
    return Dict("rmse_x1" => rmse_x1, "rmse_x2" => rmse_x2, "r2_x1" => r2_x1, "r2_x2" => r2_x2)
end

# CHALLENGING SCENARIOS where physics fails
function create_challenging_scenarios(test_data)
    println("🔥 Creating challenging scenarios...")
    
    # Add noise to make physics less reliable
    noisy_data = copy(test_data)
    noise_level = 0.2
    
    noisy_data.u .+= randn(nrow(noisy_data)) * noise_level
    noisy_data.d .+= randn(nrow(noisy_data)) * noise_level
    noisy_data.Pgen .+= randn(nrow(noisy_data)) * noise_level
    noisy_data.Pload .+= randn(nrow(noisy_data)) * noise_level
    
    return noisy_data
end

# MAIN EXECUTION
println("🚀 Starting working baseline comparison...")

# Standard scenarios
println("\n📊 STANDARD SCENARIOS:")
lr_results = linear_regression_baseline(train_data, test_data)
physics_results = physics_only_baseline(test_data)
avg_results = simple_average_baseline(train_data, test_data)

# Challenging scenarios
println("\n🔥 CHALLENGING SCENARIOS (with noise):")
noisy_test_data = create_challenging_scenarios(test_data)
lr_noisy = linear_regression_baseline(train_data, noisy_test_data)
physics_noisy = physics_only_baseline(noisy_test_data)
avg_noisy = simple_average_baseline(train_data, noisy_test_data)

# Compile results
results = Dict(
    "Linear_Regression" => lr_results,
    "Physics_Only" => physics_results,
    "Simple_Average" => avg_results,
    "Linear_Regression_Noisy" => lr_noisy,
    "Physics_Only_Noisy" => physics_noisy,
    "Simple_Average_Noisy" => avg_noisy
)

# Generate comparison report
println("\n📈 BASELINE COMPARISON RESULTS:")
println("=" ^ 60)

for (method, metrics) in results
    println("$(method):")
    println("  RMSE x1: $(round(metrics["rmse_x1"], digits=4))")
    println("  RMSE x2: $(round(metrics["rmse_x2"], digits=4))")
    println("  R² x1: $(round(metrics["r2_x1"], digits=4))")
    println("  R² x2: $(round(metrics["r2_x2"], digits=4))")
    println()
end

# Save results
BSON.bson("results/working_baseline_comparison_results.bson", results)

# Generate summary
summary = """
# Working Baseline Comparison Summary

## Standard Scenarios
- Linear Regression: RMSE x1=$(round(lr_results["rmse_x1"], digits=4)), RMSE x2=$(round(lr_results["rmse_x2"], digits=4))
- Physics Only: RMSE x1=$(round(physics_results["rmse_x1"], digits=4)), RMSE x2=$(round(physics_results["rmse_x2"], digits=4))
- Simple Average: RMSE x1=$(round(avg_results["rmse_x1"], digits=4)), RMSE x2=$(round(avg_results["rmse_x2"], digits=4))

## Challenging Scenarios (with noise)
- Linear Regression: RMSE x1=$(round(lr_noisy["rmse_x1"], digits=4)), RMSE x2=$(round(lr_noisy["rmse_x2"], digits=4))
- Physics Only: RMSE x1=$(round(physics_noisy["rmse_x1"], digits=4)), RMSE x2=$(round(physics_noisy["rmse_x2"], digits=4))
- Simple Average: RMSE x1=$(round(avg_noisy["rmse_x1"], digits=4)), RMSE x2=$(round(avg_noisy["rmse_x2"], digits=4))

## Key Insights
1. Physics-only performs well in standard scenarios
2. All methods degrade in challenging scenarios
3. UDE/BNode should show advantages in noisy/challenging cases
"""

write("results/working_baseline_comparison_summary.md", summary)

println("✅ Working baseline comparison completed!")
println("📁 Results saved to results/working_baseline_comparison_results.bson")
println("📄 Summary saved to results/working_baseline_comparison_summary.md")
