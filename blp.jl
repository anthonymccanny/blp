using CSV, DataFrames, Distributions, LinearAlgebra

# Import OTC Data
data = CSV.read("Courses/IO 1/Problem Set 1/dataps1q3_OTC_Data.csv", DataFrame)

# Import OTC Demographic Data
income = CSV.read("Courses/IO 1/Problem Set 1/dataps1q3_OTC_Demographic.csv", DataFrame)


function market_share(delta, beta_income, beta_brand, price_vec, brand_vec, income, n_draws=1000)
    n_products = length(delta)
    shares = zeros(n_products)
    
    # Draw from standard normal for nu
    nu = rand(Normal(0, 1), n_draws)

    # Draw simulated income distribution from observed income distribution
    sim_income = income[rand(1:end, n_draws)]
    
    for r in 1:n_draws
        utilities = delta .+ price_vec * (beta_income * sim_income[r]) .+ brand_vec * (beta_brand * nu[r])
        denominator = 1 + sum(exp.(utilities))
        shares .+= exp.(utilities) ./ denominator
    end
    
    return shares ./ n_draws
end




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



