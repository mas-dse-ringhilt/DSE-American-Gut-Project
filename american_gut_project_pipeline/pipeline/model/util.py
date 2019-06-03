import pandas as pd

def balance(x, y, target):
    sample_df = pd.concat([x, y], axis=1)

    df_0 = sample_df[sample_df[target] == 0]
    df_1 = sample_df[sample_df[target] == 1]

    df_class_1_over = df_1.sample(len(df_0), replace=True)
    return pd.concat([df_0, df_class_1_over], axis=0)
