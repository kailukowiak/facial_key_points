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
using MLDataPattern


train = CSV.read("data/training.csv") |> DataFrame;
X = train[:Image];

deletecols!(train, :Image);

describe(train)
## Lots of missing values
"""
Calculates all missing values of a DF Col
"""
function missing_by_col(x)
    sum(ismissing.(x))
end


[missing_by_col(col) for col = eachcol(train)]

"""
Replaces all missing values for a dataframe.
N.B. This only works on numeric columns.
"""
function replace_missing(df)
    for col in names(df)
        col_mean = mean(skipmissing(df[col]))
        df[ismissing.(df[col]), col] = col_mean
    end
    return df
end


trian = replace_missing(train);
[missing_by_col(col) for col = eachcol(train)] |> sum

## Image Manipulator

"""
Creates list of image values.
"""
function image_manipulator(X)
    l = []
    for i ∈ ProgressBar(1:size(X, 1))
        img = img_vals(X[i])
        img = Float64.(img) # Might have to make it 32
        append!(l, img)
    end
    l = reshape(l, 1, 96, 96, size(X, 1))
    Float64.(l)
end

"""
Splits cell of a dataframe into an image when seperated by spaces.
"""
function img_vals(x)
    img = split(x)
    img = tryparse.(Float64, img)
    img = reshape(img, 96, 96)
    img = img' # To fix the orientation of the images
    img = Float32.(img)
    img
end

x = image_manipulator(X);

imshow(x[1, :, :, 2])

# Make train test split

y_train, y_val = splitobs(train, 0.7);
x_train, x_val = splitobs(x, 0.7);

## Model

