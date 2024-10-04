using CSV, DataFrames, Distributions, LinearAlgebra, BenchmarkTools, Printf

# Import OTC Data
data = CSV.read("dataps1q3_OTC_Data.csv", DataFrame)

# Import OTC Demographic Data
income = CSV.read("dataps1q3_OTC_Demographic.csv", DataFrame)

function market_share(delta, income_effect, brand_effect, price_vec, brand_vec, n_draws=1000)
    n_products = length(delta)

    # Pre-allocate arrays
    shares = zeros(n_products)
    utilities = zeros(n_products)
    
    for r in 1:n_draws
        @. utilities = delta + price_vec * income_effect[r] + brand_vec * brand_effect[r]
        denominator = 1 + sum(exp, utilities)
        @. shares += exp(utilities) / denominator
    end
    
    return shares ./ n_draws
end


function contraction_mapping_delta(starting_delta, market_share_observed, beta_income, beta_brand, price_vec, brand_vec, income)
    max_iter = 10000
    tol = 1e-3
    delta = starting_delta

    # Draw 100 values from a standard normal distribution for each value of income
    n_draws_per_income = 100
    nu = randn(length(income) * n_draws_per_income)

    # Repeat each income value 100 times to match the length of nu
    sim_income = repeat(income, inner=n_draws_per_income)

    # Vectorized operations
    income_effect = beta_income .* sim_income
    brand_effect = beta_brand .* nu

    for iter in 1:max_iter
        # Calculate the predicted market shares
        predicted_shares = market_share(delta, income_effect, brand_effect, price_vec, brand_vec)
        
        # Update delta using the contraction mapping formula
        delta_new = delta + log.(market_share_observed ./ predicted_shares)
        
        # Check for convergence
        if norm(delta_new - delta) < tol
            print("YOU DID IT!")
            return delta_new
        end
        println(@sprintf("%.3e", norm(delta_new - delta)))

        # Update delta for the next iteration
        delta = delta_new
    end
    
    error("Contraction mapping did not converge")

end 




# Call the contraction_mapping_delta function with test parameters
delta_converged = contraction_mapping_delta(starting_delta_test, market_share_observed_test, beta_income_test, beta_brand_test, price_vec_test, brand_vec_test, income_test)

# Print the test results
println("Converged Delta:")
println(delta_converged)








# Test the market_share function

# Define test parameters
delta_test = [1.0, 2.0, 3.0]
beta_income_test = 0.5
beta_brand_test = 0.3
price_vec_test = [1.0, 1.5, 2.0]
brand_vec_test = [0.0, 1.0, 0.0]

# Use a subset of the income data for testing
income_test = 1:10

# Benchmark the market_share function
benchmark = @benchmark market_share($delta_test, $beta_income_test, $beta_brand_test, $price_vec_test, $brand_vec_test, $income_test)

# Print the benchmark results
println("Benchmark results:")
display(benchmark)

# Call the market_share function with test parameters
shares_test = market_share(delta_test, beta_income_test, beta_brand_test, price_vec_test, brand_vec_test, income_test)

# Print the test results
println("Test Market Shares:")
println(shares_test)

# Benchmark the optimized function
benchmark_optimized = @benchmark market_share_optimized($delta_test, $beta_income_test, $beta_brand_test, $price_vec_test, $brand_vec_test, $income_test)

println("\nBenchmark results for optimized function:")
display(benchmark_optimized)

# Test the optimized function and print results
shares_test_optimized = market_share_optimized(delta_test, beta_income_test, beta_brand_test, price_vec_test, brand_vec_test, income_test)
println("\nFirst few market shares (optimized):")
println(shares_test_optimized[1:3])

# Compare results
println("\nAre the results approximately equal?")
println(isapprox(shares_test, shares_test_optimized, rtol=1e-4))





function log_likelihood(θ, data, income)
    δ = θ[1:end-1]  # All but last element are δ
    β = θ[end]      # Last element is β
    
    X = hcat(data.x1, data.x2)  # Assuming x1 and x2 are the product characteristics
    
    predicted_shares = market_share(δ, β, X, income.income)
    observed_shares = data.share
    
    return sum(observed_shares .* log.(predicted_shares))
end

# Calculate and print summary statistics for the OTC Data
println("Summary Statistics for OTC Data:")
println(describe(data))

# Calculate and print summary statistics for the Income Data
println("\nSummary Statistics for Income Data:")
println(describe(income))

# Print the first few rows of each dataset
println("\nFirst few rows of OTC Data:")
println(first(data, 5))

println("\nFirst few rows of Income Data:")
println(first(income, 5))

# Print the number of unique products and markets
println("\nNumber of unique products: ", length(unique(data.product)))
println("Number of unique markets: ", length(unique(data.market)))

# Calculate and print average price and share across all products and markets
println("\nAverage price across all products and markets: ", mean(data.price))
println("Average share across all products and markets: ", mean(data.share))

# Calculate and print average income across all markets
println("\nAverage income across all markets: ", mean(income.income))

println("Hello, World!")



