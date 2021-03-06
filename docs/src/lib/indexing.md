
```@meta
CurrentModule = DataFrames
```

# Indexing

```@index
Pages = ["indexing.md"]
```

## General rules

The following rules explain target functionality of how `getindex`, `setindex!`, and `view` are intended to work with `DataFrame`, `SubDataFrame` and `DataFrameRow` objects.

The rules for a valid type of index into a column are the following:
* a value, later denoted as `col`:
    * a `Symbol`;
    * an `Integer` that is not `Bool`;
* a vector, later denoted as `cols`:
    * a vector of `Symbol` (does not have to be a subtype of `AbstractVector{Symbol}`);
    * a vector of `Integer` other than `Bool` (does not have to be a subtype of `AbstractVector{<:Integer}`);
    * a vector of `Bool` that has to be a subtype of `AbstractVector{Bool}`;
    * a colon;
    * a regular expression, which gets expanded to a vector of matching column names.

The rules for a valid type of index into a row are the following:
* a value, later denoted as `row`:
    * an `Integer` that is not `Bool`;
* a vector, later denoted as `rows`:
    * a vector of `Integer` other than `Bool` (does not have to be a subtype of `AbstractVector{<:Integer}`);
    * a vector of `Bool` that has to be a subtype of `AbstractVector{Bool}`;
    * a colon.

In the descriptions below `df` represents a `DataFrame`, `sdf` is a `SubDataFrame` and `dfr` is a `DataFrameRow`.

## `getindex`

The following list specifies return types of `getindex` operations depending on argument types.

In all operations copying vectors is avoided where possible.
If it is performed a description explicitly mentions that the data is *copied*.

For performance reasons, accessing, via `getindex` or `view`, a single `row` and multiple `cols` of a `DataFrame`, a `SubDataFrame` or a `DataFrameRow` always returns a `DataFrameRow` (which is a view-like type).

`DataFrame`:
* `df[col]` -> the vector contained in column `col`;
* `df[cols]` -> a freshly allocated `DataFrame` containing the copies of vectors contained in columns `cols`;
* `df[row, col]` -> the value contained in row `row` of column `col`, the same as `df[col][row]`;
* `df[CartesianIndex(row, col)]` -> the same as `df[row,col]`;
* `df[row, cols]` -> a `DataFrameRow` with parent `df` if `cols` is a colon and `df[cols]` otherwise;
* `df[rows, col]` -> a copy of the vector `df[col]` with only the entries corresponding to `rows` selected, the same as `df[col][rows]`;
* `df[rows, cols]` -> a `DataFrame` containing copies of columns `cols` with only the entries corresponding to `rows` selected.
* `@view df[col]` -> the vector contained in column `col` (this is equivalent to `df[col]`);
* `@view df[cols]` -> a `SubDataFrame` with parent `df` if `cols` is a colon and `df[cols]` otherwise;
* `@view df[row, col]` -> a `0`-dimensional view into `df[col]`, the same as `view(df[col], row)`;
* `@view df[row, cols]` -> a `DataFrameRow` with parent `df` if `cols` is a colon and `df[cols]` otherwise;
* `@view df[rows, col]` -> a view into `df[col]` with `rows` selected, the same as `view(df[col], rows)`;
* `@view df[rows, cols]` -> a `SubDataFrame` with `rows` selected with parent `df` if `cols` is a colon and `df[cols]` otherwise.

`SubDataFrame`:
* `sdf[col]` -> a view of the vector contained in column `col` of `parent(sdf)` with `DataFrames.rows(sdf)` as a selector;
* `sdf[cols]` -> a `SubDataFrame`, with parent `parent(sdf)` if `cols` is a colon and `parent(sdf)[cols]` otherwise;
* `sdf[row, col]` -> a value contained in row `row` of column `col`;
* `sdf[row, cols]` -> a `DataFrameRow` with parent `parent(sdf)` if `cols` is a colon and `parent(sdf)[cols]` otherwise;
* `sdf[rows, col]` -> a copy of a vector `sdf[col]` with only rows `rows` selected;
* `sdf[rows, cols]` -> a `DataFrame` containing columns `cols` and `df[rows, col]` as a vector in each `col` in `cols`.
* `@view sdf[col]` -> a view of vector contained in column `col` of `parent(sdf)` with `DataFrames.rows(sdf)` as selector;
* `@view sdf[cols]` -> a `SubDataFrame` with parent `parent(sdf)` if `cols` is a colon and `parent(sdf)[cols]` otherwise;
* `@view sdf[row, col]` -> translates to `view(sdf[col], row)` (a `0`-dimensional view into `df[col]`);
* `@view sdf[row, cols]` -> a `DataFrameRow` with parent `parent(sdf)` if `cols` is a colon and `parent(sdf)[cols]` otherwise;
* `@view sdf[rows, col]` -> translates to `view(sdf[col], rows)` (a standard view into `sdf[col]` vector);
* `@view sdf[rows, cols]` -> a `SubDataFrame` with parent `parent(sdf)` if `cols` is a colon and `sdf[cols]` otherwise.

`DataFrameRow`:
* `dfr[col]` -> the value contained in column `col` of `dfr`;
* `dfr[cols]` -> a `DataFrameRow` with parent `parent(dfr)` if `cols` is a colon and `parent(dfr)[cols]` otherwise;
* `@view dfr[col]` -> a `0`-dimensional view into `parent(dfr)[DataFrames.row(dfr), col]`;
* `@view dfr[cols]` -> a `DataFrameRow` with parent `parent(dfr)` if `cols` is a colon and `parent(dfr)[cols]` otherwise;

## `setindex!`

Under construction

## Broadcasting

The following broadcasting rules apply to `AbstractDataFrame` objects:
* `AbstractDataFrame` behaves in broadcasting like a two-dimensional collection compatible with matrices.
* If an `AbstractDataFrame` takes part in broadcasting then a `DataFrame` is always produced as a result.
  In this case the requested broadcasting operation produce an object with exactly two dimensions.
  An exception is when an `AbstractDataFrame` is used only as a source of broadcast assignment into an object
  of dimensionality higher than two.
* If multiple `AbstractDataFrame` objects take part in broadcasting then they have to have identical column names.

It is possible to assign a value to `AbstractDataFrame` and `DataFrameRow` objects using the `.=` operator.
In such an operation `AbstractDataFrame` is considered as two-dimensional and `DataFrameRow` as single-dimensional.

!!! note

    The rule above means that, similar to single-dimensional objects in Base (e.g. vectors),
    `DataFrameRow` is considered to be column-oriented.

If column indexing using `Symbol` names is performed the order of columns in the operation is specified
by the order of names.

`df[col] .= value` is allowed when `col` is a `Symbol` even if `col` is not present in the `DataFrame`
under the condition that `df` is not empty: a new column will be created.
On the contrary, `df.col .= value` is not allowed if `col` is not present in `df`.
