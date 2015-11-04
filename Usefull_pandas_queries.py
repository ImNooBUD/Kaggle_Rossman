__author__ = 'dworkin'

ext_sales_df[ext_sales_df.columns[~ext_sales_df.columns.isin(['PreviousMonthSales', 'PreviousWeekSales'])]].isnull().any(axis=1)

extended_sales_df['DateConmpetitionOpened'] = \
        extended_sales_df.CompetitionOpenSinceYear.astype(str).apply(lambda x: x if x=='nan' else x[:-2]) \
        + '-' + extended_sales_df.CompetitionOpenSinceMonth.astype(str).apply(lambda x: x if x=='nan' else x[:-2]) + '-01'


extended_sales_df['IsPromoInThisMonth'] = [x in y for x, y in zip(extended_sales_df['DateMonthAbbr'].astype(str),
                                                        extended_sales_df['PromoInterval'].astype(str))]

group_sales_monthly = extended_sales_df.groupby(['Store', 'DateYear', 'DateMonth'])['Sales'].sum()

    extended_sales_df['PreviousMonthSales'] = [
        group_sales_monthly[x, (y-relativedelta(months=1)).year, (y-relativedelta(months=1)).month]
        if (x, (y-relativedelta(months=1)).year, (y-relativedelta(months=1)).month) in group_sales_monthly.index else np.nan
        for x, y in zip(extended_sales_df.Store, extended_sales_df.Date)
        ]

