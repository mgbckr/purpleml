import pandas as pd


def to_excel(df, filename, **to_excel_kwargs):
    """
    Write pandas DataFrame to Excel file (via `df.to_excel`) and sets optimal column widths.
    Requires `xlsxwriter` to be installed.
    
    Source: https://stackoverflow.com/a/40535454

    """
    
    # write to excel
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    df.to_excel(writer, **to_excel_kwargs)

    # set columns widths
    worksheet = writer.sheets["Sheet1"] # pull worksheet object
    for idx, col in enumerate(df):  # loop through all columns
        series = df[col]
        max_len = max((
            series.astype(str).map(len).max(),  # len of largest item
            len(str(series.name))  # len of column name/header
            )) + 1  # adding a little extra space
        worksheet.set_column(idx, idx, max_len)  # set column width

    writer.save()
