def merge_df_on_float(df_a, df_b, on, accuracy=1e3, how='inner'):
    df_a = df_a.assign(merger=(df_a[on] * accuracy).astype(int))
    df_b = df_b.assign(merger=(df_b[on] * accuracy).astype(int))
    return df_a.merge(df_b.drop(on, axis=1), how=how, on='merger').drop('merger', axis=1)