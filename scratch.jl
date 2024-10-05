using BenchmarkTools, DataFrames, StaticArrays

# Create a sample DataFrame
df = data  # Assuming 'data' is your DataFrame

### Preprocess data into a dictionary of matrices
# We will have to repeatedly access the data for a given market in a given week.
# As extracting data from a dataframe can be slow, we first put all the data into a format which is easy to access
# We store the data for each market-week in a matrix (useful for multiplying later) and index it by a tuple for store and week
market_data = Dict{Tuple{Int64, Int64}, Matrix{Float64}}()

for group in groupby(data, [:store, :week])
    store = first(group.store)
    week = first(group.week)
    matrix = Matrix{Float64}(group[:, [:sales, :cost, :branded, :price, :promotion]])
    market_data[(store, week)] = matrix
end

# Extract data for market 2, week 1
market_2_week_1 = market_data[(2, 1)][:, [4, 5]]


# Function to get data using DataFrame subsetting
function get_data_df(df, store, week)
    subset = df[(df.store .== store) .& (df.week .== week), :]
    return [SVector{5, Float64}(row.sales, row.price, row.promotion, row.cost, row.branded) for row in eachrow(subset)]
end

# Function to get data using preprocessed dictionary
function get_data_dict(market_data, store, week)
    get(market_data, (store, week), SVector{5, Float64}[])
end

# Benchmark
store, week = 50, 25
@btime get_data_df($df, $store, $week);
@btime get_data_dict($market_data, $store, $week);