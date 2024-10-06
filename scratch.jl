  
function predicted_market_share(exp_delta, income_effect, brand_effect, price_vec, brand_vec)
    n_products = length(exp_delta)

    # Create the initial vectors. `shares` and `utilities` are vectors the same length as the number of products
    shares = zeros(n_products)
    utilities = zeros(n_products)
    
    # For every one of our random draws calculate the utility of each product and then the probability of choosing each product
    # Then add all the probabilities together
    for i in 1:n_draws
        # The @. notation turns operations in this line into vector operations, and can give better performance
        @. utilities = exp_delta * exp(price_vec * income_effect[i] + brand_vec * brand_effect[i])
        denominator = 1 + sum(utilities)
        @. shares += ifelse(isnan(utilities / denominator), 0, utilities / denominator)
    end
    
    # Divide the sum of probabilities by the number of random draws to get the empirical integral
    # This is the predicted market share of each product
    # Then return this vector of predicted market shares
    
    # Check if any of the shares are not a number (NaN)
    if any(isnan, shares)
        # Print the current shares and utilities

        println("Current exp_delta: ", exp_delta)
        println("Current shares: ", shares)
        println("Current utilities: ", utilities)
        
        error("Error: Predicted market shares contain NaN values")
    end
    
    return shares ./ n_draws
end


function delta_contraction_mapping(market_matrix, income_vector, sigma_income, sigma_brand)
    max_iter = 10000
    tol = 1e-3

    # Declare the vectors of realized random coefficients from our randomly drawn data
    income_effect = sigma_income .* income_vector
    brand_effect = sigma_brand .* nu

    # Get the price and branded information about products for this market
    price_vec = market_matrix[:, PRICE_COL]
    brand_vec = market_matrix[:, BRANDED_COL]

    # Set a starting value for exp(delta) for each product
    exp_delta = ones(size(market_matrix, 1))
    exp_delta_new = fill(exp(10.0), size(market_matrix, 1))
    predicted_shares = zeros(size(market_matrix, 1))

    for iter in 1:max_iter
        # Update exp_delta for the next iteration
        exp_delta = exp_delta_new

        # Calculate the predicted market shares
        predicted_shares = predicted_market_share(exp_delta, income_effect, brand_effect, price_vec, brand_vec)
        predicted_shares = [max(share, 0.00001) for share in predicted_shares]
        
        # Update exp_delta using the exponentiated contraction mapping formula
        exp_delta_new = exp_delta .* (market_matrix[:, SHARES_COL] ./ predicted_shares)
        
        # Check for convergence
        if norm(log.(exp_delta_new) - log.(exp_delta)) < tol
            #print("YOU DID IT!")
            return log.(exp_delta_new)
        end
        #println(@sprintf("%.3e", norm(log.(exp_delta_new) - log.(exp_delta))))

    end
    
    println("Final exp_delta_new: ", exp_delta_new)
    println("Final exp_delta: ", exp_delta)
    println("Final norm: ", norm(log.(exp_delta_new) - log.(exp_delta)))
    println("Final predicted_shares: ", predicted_shares)
    error("Contraction mapping did not converge")

end 
