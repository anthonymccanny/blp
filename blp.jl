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
    store = first(group.store)
    week = first(group.week)
    shares = group[:, :sales] ./ sum(group[:, :sales])
    matrix = Matrix{Float64}(group[:, [:sales, :cost, :branded, :price, :promotion]])
    shares = group[:, :sales] ./ sum(group[:, :sales]) # Calculate market shares
    matrix[:, 1] = shares # Replace sales with shares
    market_data[(store, week)] = matrix
end

# For future reference declare the name of each column in the matrix
SHARES_COL = 1
COST_COL = 2
BRANDED_COL = 3
PRICE_COL = 4
PROMOTION_COL = 5

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


function delta_contraction_mapping(market_matrix, income_vector, sigma_income, sigma_brand)
    max_iter = 1000
    tol = 1e-3


    # Declare the vectors of realized random coefficients from our randomly drawn data
    income_effect = sigma_income .* income_vector
    brand_effect = sigma_brand .* nu

    # Get the price and branded information about products for this market
    price_vec = market_matrix[:, PRICE_COL]
    brand_vec = market_matrix[:, BRANDED_COL]

    # Set a starting value for delta for each product
    delta = zeros(size(market_matrix, 1))

    for iter in 1:max_iter
        # Calculate the predicted market shares
        predicted_shares = predicted_market_share(delta, income_effect, brand_effect, price_vec, brand_vec)
        
        # Update delta using the contraction mapping formula
        delta_new = delta + log.(market_matrix[:, SHARES_COL] ./ predicted_shares)
        
        # Check for convergence
        if norm(delta_new - delta) < tol
            #print("YOU DID IT!")
            return delta_new
        end
        #println(@sprintf("%.3e", norm(delta_new - delta)))

        # Update delta for the next iteration
        delta = delta_new
    end
    
    error("Contraction mapping did not converge")

end 


function estimate_demand_parameters(sigma_income, sigma_brand)
    all_deltas = Float64[]
    all_prices = Float64[]
    all_promotions = Float64[]
    
    for store in 1:num_stores
        for week in 1:num_weeks
            data = market_data[(store, week)]
            income_vector = income_data[(store, week)]
            
            # Run contraction mapping
            delta_converged = delta_contraction_mapping(data, income_vector, sigma_income, sigma_brand)
            
            # Append results
            append!(all_deltas, delta_converged)
            append!(all_prices, data[:, PRICE_COL])
            append!(all_promotions, data[:, PROMOTION_COL])
        end
    end
    
    # Prepare data for regression
    X = hcat(ones(length(all_prices)), all_prices, all_promotions)
    y = all_deltas
    
    # Run OLS regression
    beta = inv(X' * X) * X' * y
    
    # Calculate residuals
    residuals = y - X * beta
    
    # Extract coefficients
    beta_intercept = beta[1]
    beta_price = beta[2]
    beta_promotion = beta[3]
    
    return beta_intercept, beta_price, beta_promotion, residuals
end

# Example usage:
beta_intercept, beta_price, beta_promotion, xi = estimate_demand_parameters(1, 1)
println("Beta Price: ", beta_price)
println("Beta Promotion: ", beta_promotion)
println("Number of residuals: ", length(xi))


function create_instrument_matrix(market_data)
    num_instruments = 31  # 1 for wholesale cost, 30 for other store prices
    total_observations = sum(size(data, 1) for (_, data) in market_data)
    Z = zeros(total_observations, num_instruments)
    
    row_index = 1
    for (market, data) in market_data
        store, week = market
        num_products = size(data, 1)
        
        # Add wholesale cost as the first instrument
        Z[row_index:row_index+num_products-1, 1] = data[:, COST_COL]
        
        # Add prices from other stores as instruments
        for i in 1:30
            other_store = mod1(store + i, 30)  # Wrap around to 1 if exceeds 30
            if haskey(market_data, (other_store, week))
                other_data = market_data[(other_store, week)]
                Z[row_index:row_index+num_products-1, i+1] = other_data[:, PRICE_COL]
            end
        end
        
        row_index += num_products
    end
    
    return Z

end

function calculate_loss(xi, Z)
    ZZ_inv = inv(Z' * Z)
    return xi' * Z * ZZ_inv * Z' * xi
end

function gmm_objective(params, market_data, income_data, Z)
    sigma_income, sigma_brand = params
    _, _, _, xi = estimate_demand_parameters(sigma_income, sigma_brand)
    return calculate_loss(xi, Z)
end

# Create instrument matrix
Z = create_instrument_matrix(market_data)

# Define the optimization problem
function optimize_gmm()
    initial_params = [1.0, 1.0]  # Initial guesses for sigma_income and sigma_brand
    result = optimize(params -> gmm_objective(params, market_data, income_data, Z),
                      initial_params,
                      BFGS(),
                      Optim.Options(show_trace = true, iterations = 1000))
    return Optim.minimizer(result)
end

# Run the optimization
optimal_params = optimize_gmm()
println("Optimal sigma_income: ", optimal_params[1])
println("Optimal sigma_brand: ", optimal_params[2])

# Calculate final estimates using optimal parameters
beta_intercept, beta_price, beta_promotion, xi = estimate_demand_parameters(optimal_params[1], optimal_params[2])
println("Final Beta Intercept: ", beta_intercept)
println("Final Beta Price: ", beta_price)
println("Final Beta Promotion: ", beta_promotion)








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



