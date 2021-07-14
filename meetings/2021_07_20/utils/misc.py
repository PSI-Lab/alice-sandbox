import pandas as pd
from tqdm.autonotebook import tqdm


def add_column(df, output_col, input_cols, func, pbar=False):
    """Efficiently add a column to a DataFrame.
    This function is similar to Dataframe.apply() followed by adding
    the new series as a column in the dataframe.  Unfortunately, the
    DataFrame.apply() is slow, especially for large dataframes
    (possibly because a Series object is allocated and initialized
    with the values of the current row for each invocation of the function).
    The values of the new column are computed by calling ``func`` once
    for each row in the dataframe.  The arguments to ``func`` are the
    values of the columns specified by ``input_cols`` (in
    corresponding order).  The return value of ``func`` is set as the
    value of the new column for that row.  Note that you can use a
    lambda function with captured variables if necessary.
    Parameters
    ----------
    df : :py:class:`pandas.DataFrame`
        The dataframe to extend with a new column.  This dataframe is
        not modified.  Instead a new dataframe is returned.
    output_col : :py:class:`str`
        The name of the column to be added.
    input_cols : :py:class:`list` of :py:class:`str`
        The list of columns from ``df`` required to compute the new column.
    func : :py:class:`function`
        The function to be called once per row to compute the new column values.
    pbar : :py:class:`bool`
        Whether to display a progress bar
    Returns
    -------
    :py:class:`pandas.DataFrame`
        A new dataframe based of ``df`` with the new column added.
    """

    # make a tuple of values of the requested input columns
    input_values = tuple(df[x].values for x in input_cols)

    # transpose to make a list of value tuples, one per row
    args = zip(*input_values)

    # Attach a progress bar if required
    if pbar:
        args = tqdm(args, total=len(df))
        args.set_description("Processing %s" % output_col)

    # evaluate the function to generate the values of the new column
    output_values = [func(*x) for x in args]

    # make a new dataframe with the new column added
    columns = {x: df[x].values for x in df.columns}
    columns[output_col] = output_values
    return pd.DataFrame(columns)


def add_columns(df, output_cols, input_cols, func, keep_index=False, pbar=False):
    """Efficiently add multiple columns to a DataFrame.
    This function is similar to ``add_column`` but it adds multiple
    columns instead of just one.  The values of the new columns are
    computed by calling ``func`` once for each row in the dataframe.
    The arguments to ``func`` are the values of the columns sepcified
    by ``input_cols`` (in corresponding order).  ``func`` should
    return a value for each of the output columns (in corresponding
    order).  Note that you can use a lambda function with captured
    variables if necessary.
    Parameters
    ----------
    Parameters
    ----------
    df : :py:class:`pandas.DataFrame`
        The dataframe to extend with a new column.  This dataframe is
        not modified.  Instead a new dataframe is returned.
    output_cols : :py:class:`list` of :py:class:`str`
        The names of the columns to be added.
    input_cols : :py:class:`list` of :py:class:`str`
        The list of columns from ``df`` required to compute the new columns.
    func : :py:class:`function`
        The function to be called once per row to compute the new column values.
    pbar : :py:class:`bool`
        Whether to display a progress bar
    Returns
    -------
    :py:class:`pandas.DataFrame`
        A new dataframe based of ``df`` with the new columns added.
    """
    # handle the empty case
    if df.empty:
        new_df = pd.DataFrame(columns=df.columns)
        for output_col in output_cols:
            new_df[output_col] = []
        return new_df

    # make a tuple of values of the requested input columns
    input_values = tuple(map(lambda x: df[x].values, input_cols))

    # transpose to make a list of value tuples, one per row
    args = zip(*input_values)

    # Attach a progress bar if required
    if pbar:
        args = tqdm(args)
        args.set_description("Processing %s" % ', '.join(output_cols))

    # evaluate the function to generate the values of the new column
    output_values = zip(*map(lambda x: func(*x), args))

    # make a new dataframe with the new column added
    columns = {x: df[x].values for x in df.columns}
    for output_col, output_value in zip(output_cols, output_values):
        columns[output_col] = output_value

    index = df.index if keep_index else None
    return pd.DataFrame(columns, index=index)

