using CSV, DataFrames, Distributions, LinearAlgebra, BenchmarkTools, Printf

# Import OTC Data
data = CSV.read("dataps1q3_OTC_Data.csv", DataFrame)

# Import OTC Demographic Data
income = CSV.read("dataps1q3_OTC_Demographic.csv", DataFrame)

### PREPARE DATA

## Renumber stores
# The stores are not numbered consecutively in the data, to make things a little easier we renumber the stores from 1 to 73

# First, create a mapping of old store numbers to new store numbers
unique_stores = sort(unique(data.store))
store_mapping = Dict(old => new for (new, old) in enumerate(unique_stores))
data.store = [store_mapping[s] for s in data.store] # Update the store column in the original dataframe
income.store = [store_mapping[s] for s in income.store] # Update store numbers in income dataframe

## Preprocess market data into a dictionary of matrices
# We will have to repeatedly access the data for a given market in a given week.
# As extracting data from a dataframe can be slow, we first put all the data into a format which is easy to access
# We store the data for each market-week in a matrix (useful for multiplying later) and index it by a tuple for store and week

market_data = Dict{Tuple{Int64, Int64}, Matrix{Float64}}()

for group in groupby(data, [:store, :week])
    old_store = first(group.store)
    new_store = store_mapping[old_store]
    week = first(group.week)
    matrix = Matrix{Float64}(group[:, [:sales, :cost, :branded, :price, :promotion]])
    market_data[(new_store, week)] = matrix
end

## Preprocess income data into a dictionary of vectors 
# We will also have to repeatedly access the income distribution for each market-week combo
# Also, we need to create a vector that repeats the income values enough time to match our total number of random draws for our empirical integral
# To make the code efficient, we create these vectors first and store them in a dictionary for easy recall
income_data = Dict{Tuple{Int64, Int64}, Vector{Float64}}()

n_income_values = 20  # Number of income variables in the demographic data
n_repeats = div(n_draws, n_income_values)  # Number of times to repeat each income value

for row in eachrow(income)
    store = row.store
    week = row.week
    income_values = Vector{Float64}(row[3:end])
    repeated_income = repeat(income_values, inner=n_repeats)
    income_data[(store, week)] = repeated_income
end


# Draw the simulated nu values for the empirical integral in the inner loop
# We draw this first in order to make the code more efficient and not have to redraw random variables in each loop
# Set the number of draws for the nu variable, this should be a multiple of 20 to align with the observed income distributions
n_draws = 100
nu = randn(n_draws)




### MARKET SHARE PREDICTION FUNCTION
# Take the empirical integral of choice probabilities to get predicted market share
# We average the predicted probability of choosing each product over all our randomly drawn points
    
function predicted_market_share(delta, income_effect, brand_effect, price_vec, brand_vec)
    n_products = length(delta)

    # Create the initial vectors. `shares` and `utilities` are vectors the same length as the number of products
    shares = zeros(n_products)
    utilities = zeros(n_products)
    
    # For every one of our random draws calculate the utility of each product and then the probabilitiy of choosing each product
    # Then add all the probabilities together
    for i in 1:n_draws
        # The @. notation turns operations in this line into vector operations, and can give better performance
        @. utilities = delta + price_vec * income_effect[i] + brand_vec * brand_effect[i]
        denominator = 1 + sum(exp.(utilities))
        @. shares += exp(utilities) / denominator
    end
    
    # Divide the sum of probabilities by the number of random draws to get the empirical integral
    # This is the predicted market share of each product
    # Then return this vector of predicted market shares
    return shares ./ n_draws
end


function contraction_mapping_delta(starting_delta, market_share_observed, sigma_income, sigma_brand, price_vec, brand_vec, income)
    max_iter = 1000
    tol = 1e-3
    delta = starting_delta

    # Draw 100 values from a standard normal distribution for each value of income
    n_draws_per_income = 100
    nu = randn(length(income) * n_draws_per_income)

    # Repeat each income value 100 times to match the length of nu
    sim_income = repeat(income, inner=n_draws_per_income)

    # Vectorized operations
    income_effect = sigma_income .* sim_income
    brand_effect = sigma_brand .* nu

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



