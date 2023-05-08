import pandas as pd

if __name__ == '__main__':

    init_dir = '/projectnb2/sc3dm/eglukhov/compress/output/init_5_18'
    version = 'val_init_5'
    output_csv = f'{init_dir}_{version}.csv'

    # Initialize an empty DataFrame to store the merged data
    merged_df = pd.DataFrame()

    # Loop through file names and the corresponding i values
    for i in range(1, 6):
        # Read the CSV file

        df = pd.read_csv(f'{init_dir}_{i}/{version}/metrics.csv')

        # Add a new column with the value of i
        df['model'] = i

        # Merge the current DataFrame with the merged_df
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    merged_df.to_csv(output_csv, index=False)