using ImageView
using CSV
using Flux
using Flux: @epochs
using LinearAlgebra
using DataFrames
using CuArrays
using CUDAnative
using Statistics
using ProgressBars
using MLDataUtils

train = CSV.read("data/training.csv") |> DataFrame;

describe(train)
## Lots of missing values



function missing_by_col(x)
    sum(ismissing.(x))
end


colwise(missing_by_col, train)

function replace_missing(df)
    for col in names(df)
        col_mean = mean(skipmissing(df[col]))
        df[ismissing.(df[col]), col] = col_mean
    end
    return df
end

trian = replace_missing(train)
colwise(missing_by_col, train)

## Fixed missing
X = train[:Image];

delete!(train, :Image);

function image_manipulator(X)
    l = []
    for i ∈ ProgressBar(1:size(X, 1))
        img = img_vals(X[i])
        # push!(l, img)
        append!(l, img)
    end
    reshape(l, 1, 96, 96, size(X, 1))
end


function img_vals(x)
    img = split(x)
    img = tryparse.(Float64, img)
    img = reshape(img, 96, 96)
    img = img' # To fix the orientation of the images
    img
end

x = image_manipulator(X)



test3 = reshape(x, 1, 96, 96, 7049)

imshow(x[1, :, :, 1])

## TODO
# Make train test split

