#
#  Split - Apply - Combine operations
#

##############################################################################
##
## GroupedDataFrame...
##
##############################################################################

"""
The result of a `groupby` operation on an AbstractDataFrame; a
view into the AbstractDataFrame grouped by rows.

Not meant to be constructed directly, see `groupby`.
"""
mutable struct GroupedDataFrame{T<:AbstractDataFrame}
    parent::T
    cols::Vector{Int}    # columns used for sorting
    idx::Vector{Int}     # indexing vector when sorted by the given columns
    starts::Vector{Int}  # starts of groups
    ends::Vector{Int}    # ends of groups
end

#
# Split
#

"""
A view of an AbstractDataFrame split into row groups

```julia
groupby(d::AbstractDataFrame, cols; sort = false, skipmissing = false)
groupby(cols; sort = false, skipmissing = false)
```

### Arguments

* `d` : an AbstractDataFrame to split (optional, see [Returns](#returns))
* `cols` : data table columns to group by
* `sort`: whether to sort rows according to the values of the grouping columns `cols`
* `skipmissing`: whether to skip rows with `missing` values in one of the grouping columns `cols`

### Returns

* `::GroupedDataFrame` : a grouped view into `d`
* `::Function`: a function `x -> groupby(x, cols)` (if `d` is not specified)

### Details

An iterator over a `GroupedDataFrame` returns a `SubDataFrame` view
for each grouping into `d`. A `GroupedDataFrame` also supports
indexing by groups and `map`.

See the following for additional split-apply-combine operations:

* `by` : split-apply-combine using functions
* `aggregate` : split-apply-combine; applies functions in the form of a cross product
* `combine` : combine (obviously)
* `colwise` : apply a function to each column in an AbstractDataFrame or GroupedDataFrame

### Examples

```julia
df = DataFrame(a = repeat([1, 2, 3, 4], outer=[2]),
               b = repeat([2, 1], outer=[4]),
               c = randn(8))
gd = groupby(df, :a)
gd[1]
last(gd)
vcat([g[:b] for g in gd]...)
for g in gd
    println(g)
end
combine(d -> mean(skipmissing(d[:c])), gd)
```

"""
function groupby(df::AbstractDataFrame, cols::Vector;
                 sort::Bool = false, skipmissing::Bool = false)
    sdf = df[cols]
    df_groups = group_rows(sdf, skipmissing)
    # sort the groups
    if sort
        group_perm = sortperm(view(sdf, df_groups.rperm[df_groups.starts]))
        permute!(df_groups.starts, group_perm)
        Base.permute!!(df_groups.stops, group_perm)
    end
    GroupedDataFrame(df, DataFrames.index(df)[cols], df_groups.rperm,
                     df_groups.starts, df_groups.stops)
end
groupby(d::AbstractDataFrame, cols;
        sort::Bool = false, skipmissing::Bool = false) =
    groupby(d, [cols], sort = sort, skipmissing = skipmissing)

function Base.iterate(gd::GroupedDataFrame, i=1)
    if i > length(gd.starts)
        nothing
    else
        (view(gd.parent, gd.idx[gd.starts[i]:gd.ends[i]]), i+1)
    end
end

Base.length(gd::GroupedDataFrame) = length(gd.starts)
Compat.lastindex(gd::GroupedDataFrame) = length(gd.starts)
Base.first(gd::GroupedDataFrame) = gd[1]
Base.last(gd::GroupedDataFrame) = gd[end]

Base.getindex(gd::GroupedDataFrame, idx::Int) =
    view(gd.parent, gd.idx[gd.starts[idx]:gd.ends[idx]])
Base.getindex(gd::GroupedDataFrame, I::AbstractArray{Bool}) =
    GroupedDataFrame(gd.parent, gd.cols, gd.idx, gd.starts[I], gd.ends[I])

Base.names(gd::GroupedDataFrame) = names(gd.parent)
_names(gd::GroupedDataFrame) = _names(gd.parent)

wrap(df::AbstractDataFrame) = df
wrap(nt::NamedTuple) = nt
wrap(A::Matrix) = convert(DataFrame, A)
wrap(s::Union{AbstractVector, Tuple}) = DataFrame(x1 = s)
wrap(s::Any) = (x1 = s,)

function _vcat(vals::AbstractVector{<:NamedTuple})
    df = DataFrame()
    isempty(vals) && return df

    for col in fieldnames(typeof(first(vals)))
        df[col] = [x[col] for x in vals]
    end
    df
end

"""
Combine a GroupApplied object (rudimentary)

```julia
combine(ga::GroupApplied)
```

### Arguments

* `ga` : a GroupApplied

### Returns

* `::DataFrame`

### Examples

```julia
df = DataFrame(a = repeat([1, 2, 3, 4], outer=[2]),
               b = repeat([2, 1], outer=[4]),
               c = randn(8))
gd = groupby(df, :a)
combine(d -> sum(skipmissing(d[:c])), gd)
```

"""
function Base.map(f::Function, gd::GroupedDataFrame)
    if length(gd) > 0
        idx, valscat = _combine(wrap(f(gd[1])), f, gd)
        return hcat!(gd.parent[idx, gd.cols], valscat)
    else
        return similar(gd.parent, 0)[gd.cols]
    end
end

@deprecate combine(f::Function, gd::GroupedDataFrame) map(f, gd)
@deprecate combine(gd::GroupedDataFrame) map(identity, gd)

function _combine(first::NamedTuple, f::Function, gd::GroupedDataFrame)
    m = length(first)
    n = length(gd)
    idx = Vector{Int}(undef, n)
    initialcols = ntuple(i -> Vector{typeof(first[i])}(undef, n), m)
    cols = _combine!(initialcols, 1, first, f, gd, idx)
    valscat = DataFrame(collect(cols), collect(propertynames(first)))
    idx, valscat
end

function _combine!(oldcols, start::Integer, first::NamedTuple, f::Function, gd::GroupedDataFrame, idx::Vector{Int})
    n = length(first)
    len = length(gd)
    cols = ntuple(n) do i
        T = eltype(oldcols[i])
        if first[i] isa T
            return oldcols[i]
        else
            return copyto!(Vector{promote_type(T, typeof(first[i]))}(undef, len), 1,
                           oldcols[i], 1, start-1)
        end
    end
    # Handle first group
    idx[start] = gd.idx[gd.starts[start]]
    for j in 1:n
        col = cols[j]
        if first[j] isa eltype(col)
            col[start] = first[j]
        else
            return _combine!(cols, start, first, f, gd, idx)
        end
    end
    # Handle remaining groups
    @inbounds for i in start+1:len
        val = wrap(f(gd[i]))
        idx[i] = gd.idx[gd.starts[i]]
        for j in 1:n
            col = cols[j]
            if val[j] isa eltype(col)
                col[i] = val[j]
            else
                return _combine!(cols, i, val, f, gd, idx)
            end
        end
    end
    cols
end

function _combine(first::AbstractDataFrame, f::Function, gd::GroupedDataFrame)
    len = length(gd)
    idx = Vector{Int}(undef, len)
    # Handle first group
    n = size(first, 1)
    idx[1:n] .= gd.idx[1]
    valscat = deepcopy(first)
    # Handle remaining groups
    j = size(first, 1)
    @inbounds for i in 2:len
        val = wrap(f(gd[i]))
        n = size(val, 1)
        idx[j .+ (1:n)] .= gd.idx[gd.starts[i]]
        append!(valscat, val)
        j += n
    end
    idx, valscat
end


"""
Apply a function to each column in an AbstractDataFrame or
GroupedDataFrame

```julia
colwise(f::Function, d)
colwise(d)
```

### Arguments

* `f` : a function or vector of functions
* `d` : an AbstractDataFrame of GroupedDataFrame

If `d` is not provided, a curried version of groupby is given.

### Returns

* various, depending on the call

### Examples

```julia
df = DataFrame(a = repeat([1, 2, 3, 4], outer=[2]),
               b = repeat([2, 1], outer=[4]),
               c = randn(8))
colwise(sum, df)
colwise([sum, length], df)
colwise((minimum, maximum), df)
colwise(sum, groupby(df, :a))
```

"""
colwise(f, d::AbstractDataFrame) = [f(d[i]) for i in 1:ncol(d)]

# apply several functions to each column in a DataFrame
colwise(fns::Union{AbstractVector, Tuple}, d::AbstractDataFrame) = [f(d[i]) for f in fns, i in 1:ncol(d)]
colwise(f, gd::GroupedDataFrame) = [colwise(f, g) for g in gd]

"""
Split-apply-combine in one step; apply `f` to each grouping in `d`
based on columns `col`

```julia
by(d::AbstractDataFrame, cols, f::Function; sort::Bool = false)
by(f::Function, d::AbstractDataFrame, cols; sort::Bool = false)
```

### Arguments

* `d` : an AbstractDataFrame
* `cols` : a column indicator (Symbol, Int, Vector{Symbol}, etc.)
* `f` : a function to be applied to groups; expects each argument to
  be an AbstractDataFrame
* `sort`: sort row groups (no sorting by default)

`f` can return a value, a vector, or a DataFrame. For a value or
vector, these are merged into a column along with the `cols` keys. For
a DataFrame, `cols` are combined along columns with the resulting
DataFrame. Returning a DataFrame is the clearest because it allows
column labeling.

A method is defined with `f` as the first argument, so do-block
notation can be used.

`by(d, cols, f)` is equivalent to `combine(map(f, groupby(d, cols)))`.

### Returns

* `::DataFrame`

### Examples

```julia
df = DataFrame(a = repeat([1, 2, 3, 4], outer=[2]),
               b = repeat([2, 1], outer=[4]),
               c = randn(8))
by(df, :a, d -> sum(d[:c]))
by(df, :a, d -> 2 * skipmissing(d[:c]))
by(df, :a, d -> DataFrame(c_sum = sum(d[:c]), c_mean = mean(skipmissing(d[:c]))))
by(df, :a, d -> DataFrame(c = d[:c], c_mean = mean(skipmissing(d[:c]))))
by(df, [:a, :b]) do d
    DataFrame(m = mean(skipmissing(d[:c])), v = var(skipmissing(d[:c])))
end
```

"""
by(d::AbstractDataFrame, cols, f::Function; sort::Bool = false) =
    map(f, groupby(d, cols, sort = sort))
by(f::Function, d::AbstractDataFrame, cols; sort::Bool = false) =
    by(d, cols, f, sort = sort)

#
# Aggregate convenience functions
#

# Applies a set of functions over a DataFrame, in the from of a cross-product
"""
Split-apply-combine that applies a set of functions over columns of an
AbstractDataFrame or GroupedDataFrame

```julia
aggregate(d::AbstractDataFrame, cols, fs)
aggregate(gd::GroupedDataFrame, fs)
```

### Arguments

* `d` : an AbstractDataFrame
* `gd` : a GroupedDataFrame
* `cols` : a column indicator (Symbol, Int, Vector{Symbol}, etc.)
* `fs` : a function or vector of functions to be applied to vectors
  within groups; expects each argument to be a column vector

Each `fs` should return a value or vector. All returns must be the
same length.

### Returns

* `::DataFrame`

### Examples

```julia
df = DataFrame(a = repeat([1, 2, 3, 4], outer=[2]),
               b = repeat([2, 1], outer=[4]),
               c = randn(8))
aggregate(df, :a, sum)
aggregate(df, :a, [sum, x->mean(skipmissing(x))])
aggregate(groupby(df, :a), [sum, x->mean(skipmissing(x))])
```

"""
aggregate(d::AbstractDataFrame, fs::Function; sort::Bool=false) = aggregate(d, [fs], sort=sort)
function aggregate(d::AbstractDataFrame, fs::Vector{T}; sort::Bool=false) where T<:Function
    headers = _makeheaders(fs, _names(d))
    _aggregate(d, fs, headers, sort)
end

# Applies aggregate to non-key cols of each SubDataFrame of a GroupedDataFrame
aggregate(gd::GroupedDataFrame, f::Function; sort::Bool=false) = aggregate(gd, [f], sort=sort)
function aggregate(gd::GroupedDataFrame, fs::Vector{T}; sort::Bool=false) where T<:Function
    headers = _makeheaders(fs, setdiff(_names(gd), _names(gd.parent[gd.cols])))
    res = combine(x -> _aggregate(without(x, gd.cols), fs, headers), gd)
    sort && sort!(res, headers)
    res
end

# Groups DataFrame by cols before applying aggregate
function aggregate(d::AbstractDataFrame,
                   cols::Union{S, AbstractVector{S}},
                   fs::Union{T, Vector{T}};
                   sort::Bool=false) where {S<:ColumnIndex, T <:Function}
    aggregate(groupby(d, cols, sort=sort), fs)
end

_makeheaders(fs::Vector{<:Function}, cn::Vector{Symbol}) =
    [Symbol(colname, '_', nameof(f)) for f in fs for colname in cn]

function _aggregate(d::AbstractDataFrame, fs::Vector{T}, headers::Vector{Symbol}, sort::Bool=false) where T<:Function
    res = DataFrame(Any[vcat(f(d[i])) for f in fs for i in 1:size(d, 2)], headers)
    sort && sort!(res, headers)
    res
end
