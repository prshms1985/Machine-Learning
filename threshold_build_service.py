# %% [markdown]
# # Threshold Build Class and Fucntion Definition Module

# %%
__version__ = '0.1.4'

# %% Importing required modules

import pandas as pd
import numpy as np
from numpy import trapz

import datetime
import pickle
##from pyhive import hive
import os

import dash_bootstrap_components as dbc
import dash
import dash_core_components as dcc
import dash_html_components as html
##import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import dash_table

# %% Creating a proccesing-time tracking function
def track_time(mode='start',start=None,toggle=False):
    # Toggle: False or True to turn print messages off and on
    if mode=='start':
        start = datetime.datetime.now()
        if toggle: print('Start: {}'.format(start))
        return start

    if mode=='end':
        end = datetime.datetime.now()
        dif = (end- start).total_seconds()
        if toggle: print('End: {}\nTotal Time Elapsed: {} seconds'.format(end,dif))
        return dif

# %% [markdown]
#  NECESSARY STANDLONE FUNCTIONS

# %% Extract Data From Hive

def hive_extract(
        metricname,
        metricquery,
        chnl,
        typology,
        connection,
        file_name=None,
        verbose=False):
    """This function pulls data from raw Hive tables and stores in a pandas df.

    metricname: How to store the metric in pandas df
    metricquery: Name of metric in deltap_prd_qmtbls.{chnl}_events_datamart
    chnl: the prefix to correctly call deltap_prd_qmtbls.{chnl}_events_datamart
    typology: the specific typology to extract, filtering on primary_high_level_reason
    connection: Hive connection script
    file_name: If not empty, will save to a csv in working folder with this name
    verbose: set flag to see additional print outputs

    Source tables in Hive:
        1. deltap_prd_qmtbls.{chnl}_events_datamart
        2. deltap_prd_qmtbls.customer_datamart

    Required fields:
        - deltap_prd_qmtbls.{chnl}_events_datamart.primary_high_level_reason
        - deltap_prd_qmtbls.customer_datamart.churn_vol_m1_flag
        - month field between 201810 and 201903 on both
        - msisdn field on both

    Query of form:
        select      metricquery                     as metricname
                    ,a.primary_high_level_reason    as primary_high_level_reason
                    ,count(*)                       as events
                    ,sum(churn_vol_m1_flag)         as churn_m1
                    ,sum(coalesce(churn_vol_m1_flag,0)) / count(*) as churn_rate_m1
        from        deltap_prd_qmtbls.{chnl}_events_datamart        as a

                    left join deltap_prd_qmtbls.customer_datamart   as b
                    on  a.msisdn = b. msisdn
                        and a.month = b.month
        where       a.month between 201810 and 201903
                    and lower(a.primary_high_level_reason) =
        group by    metricquery
                    ,a.primary_high_level_reason
    """

    # Step1: Generate Query
    hive_query = '''
        select
            {v2} as {v1}'''.format(v1=metricname, v2=metricquery)

    if typology.lower() != 'all':
        hive_query += '''
            , a.primary_high_level_reason as primary_high_level_reason'''

    hive_query += '''
            , count(*) as events
            , sum(churn_vol_m1_flag) as churn_m1
            , sum(coalesce(churn_vol_m1_flag,0)) / count(*) as churn_rate_m1

        from    deltap_prd_qmtbls.{chnl}_events_datamart as a
                left join deltap_prd_qmtbls.customer_datamart as b
                on  a.msisdn = b. msisdn
                    and a.month = b.month
        where   a.month between 201810 and 201903'''.format(chnl=chnl)

    if typology.lower() != 'all':
        hive_query += '''
        and     lower(a.primary_high_level_reason) = '{t}' '''.format(t=typology.lower())

    hive_query += '''
        group by
            {v2}'''.format(v2=metricquery)

    if typology.lower() != 'all':
        hive_query +='''
            , a.primary_high_level_reason'''

    # Step2: Submit Query to Hive

    if verbose: print(hive_query)

    base_df = pd.read_sql(hive_query, connection)

    # Step3: Add column for metric name
    base_df['metricname'] = metricname

    if typology.lower() == 'all':
        base_df['primary_high_level_reason'] = 'all'

    # Step4: Save File if instructed to do so

    if file_name:
        base_df.to_csv(file_name)

    # Return the created dataframe
    return base_df

# %% Create A Percentile Df From An Aggregated Df

def create_percentiles(df,metricname,higher_is_better=True):
    '''This function creates a percentile dataframe

    Needed for other methods within the threshold build class.
    '''
    # First, we create percentile upper boundaries
    temp_df = df[df[metricname].notnull()].copy(deep=True)

    temp_df.sort_values(by=metricname, inplace=True, ascending=higher_is_better)

    temp_df['cumsum'] = temp_df['events'].cumsum()

    factor = sum(temp_df['events']) / 100

    temp_df['percentiles_ub'] = (temp_df['cumsum'] / factor).apply(int)
    temp_df['percentiles_ub'] =  temp_df['percentiles_ub'] / 100

    # Second, we aggregate using the percentiles and create a new dataframe

    percentile_df = temp_df.groupby('percentiles_ub').agg({metricname:'max', 'events':'sum', 'churn_m1':'sum'}).reset_index()

    percentile_df.loc[percentile_df['percentiles_ub'] == 1, 'percentiles_ub'] = 0.99 #The 100th percentile should be 99

    percentile_df.sort_values(by='percentiles_ub',inplace=True)

    percentile_df['percentiles_lb'] = percentile_df['percentiles_ub'].shift(1).fillna(-0.01)

    min_value = min(temp_df[metricname])
    percentile_df[metricname+'_lb'] = percentile_df[metricname].shift(1).fillna(min_value - 1)

    percentile_df['churn_rate_m1'] = percentile_df['churn_m1'] / percentile_df['events']

    return percentile_df

# %% Finding Cutoff Values In A Df associated with a particular percentile

def find_cutoff_value(df,metricname,cutoff_pct):
    '''DOCSTRING
    '''
    percentile_df = create_percentiles(df,metricname)

    lb_condition = (cutoff_pct >= percentile_df['percentiles_lb'] + 0.0001)
    ub_condition = (cutoff_pct <= percentile_df['percentiles_ub'] + 0.0001)

    cutoff_value = percentile_df.loc[lb_condition & ub_condition,metricname]

    return float(cutoff_value)

# %% Finding a percentile value in a DF associated with a particular cutoff value

def find_percentile(df,metricname,value):
    '''DOCSTRING
    '''

    percentile_df = create_percentiles(df,metricname)

    lb_condition = (value >= percentile_df[metricname + '_lb'] + 0.0001)
    ub_condition = (value <= percentile_df[metricname] + 0.0001)

    percentile = percentile_df.loc[lb_condition & ub_condition, 'percentiles_ub']

    return float(percentile)

# %% Finding churn rates in a DF associated with a particular cutoff value

def find_churn_rate(df,metricname,upper_bound=None,lower_bound=None):
    '''DOCSTRING
    '''

    condition = True

    if lower_bound != None:
        lb_condition = (df[metricname] >= lower_bound)
        condition = (condition & lb_condition)

    if upper_bound != None:
        ub_condition = (df[metricname] <= upper_bound)
        condition = (condition & ub_condition)

    aggregated_df = df.loc[condition, :].sum()
    churn_rate = aggregated_df['churn_m1'] / aggregated_df['events']

    return float(churn_rate)

# %% Scoring Formula to Rescale Values


def qes_scoring(value,one,two,three,four,five,direction='proportional'):
    '''DOCSTRING

    # TODO: Implement inverse scoring
    '''

    if direction == 'proportional':
        if value <= one:
            score = 1
        elif one < value <= two:
            score = 1 + (value - one) / (two - one)
        elif two < value <= three:
            score = 2 + (value - two) / (three - two)
        elif three < value <= four:
            score = 3 + (value - three) / (four - three)
        elif four < value <= five:
            score = 4 + (value - four) / (five - four)
        elif value >= five:
            score = 5
        else:
            score = None

    return score


def qes_score_bins(x):
    '''DEPRECATED
    '''
    if x == 1:
        result = 'a. x = 1'
    elif 1 < x < 2:
        result = 'b. 1 < x < 2'
    elif 2 <= x < 3:
        result = 'c. 2 <= x < 3'
    elif 3 <= x < 4:
        result = 'd. 3 <= x < 4'
    elif 4 <= x < 5:
        result = 'e. 4 <= x < 5'
    elif x == 5:
        result = 'f. x = 5'
    else:
        result = None

    return result

def qes_score_4_bins(x):
    '''DEPRECATED
    '''
    if 1 <= x < 2:
        result = 'a. 1 <= x < 2'
    elif 2 <= x < 3:
        result = 'b. 2 <= x < 3'
    elif 3 <= x < 4:
        result = 'c. 3 <= x < 4'
    elif 4 <= x <= 5:
        result = 'd. 4 <= x <= 5'
    else:
        result = None

    return result

# %% Shortlist Cutoffs
def shortlist_cutoffs(df, metricname, cutoff, benchmark, lo_bound, hi_bound, suffix, df_size=100, prints=True):
    ''' 1. Subset
    2. sort
    3. calculate cumulative sums
    4. use cumulative sums to calculate churn rates before each point
    5. repeat step 2 to 4 with a reverse sorting order to calculate churn rates after each point
    '''
    # attempt to create a pandas dataframe with the cumulative churn + events for every value
    # try:
    # Create a base condition that will not filter any observations
    condition = True
    # Add a lower bound if one is specified
    if lo_bound != None:
        lb_condition = (df[metricname] > lo_bound)
        condition = condition & lb_condition
    # Add an higher bound if one is specified
    if hi_bound != None:
        ub_condition = (df[metricname] < hi_bound)
        condition = condition & ub_condition

    # Filter input df for all values outside of bounds
    temp_df = df.loc[condition, :].copy(deep=True)
    if prints and len(temp_df) <= 0: print('No possible cutoffs to choose from for cutoff {}'.format(cutoff))

    # Sort the remaining values in ascending order to calculate churn rates before the cutoff values
    temp_df.sort_values(by=metricname, ascending=True, inplace=True)
    # Calculate cumulative sum of events before the particular value in any given row
    temp_df['events_before'+suffix] = temp_df['events'].cumsum()
    # Remove overlap from current row
    temp_df['events_before'+suffix] = temp_df['events_before'+suffix] - temp_df['events']
    # Calculate cumulative sum of churners before the particular value in any given row
    temp_df['churners_before'+suffix] = temp_df['churn_m1'].cumsum()
    # Remove overlap from current row
    temp_df['churners_before'+suffix] = temp_df['churners_before'+suffix] - temp_df['churn_m1']
    # Calculate the weighted churn rate for the values before the cutoff
    temp_df['churn_before'+suffix] = temp_df['churners_before'+suffix] / temp_df['events_before'+suffix]

    # Sort the remaining values in ascending order to calculate churn rates after the cutoff values
    temp_df.sort_values(by=metricname, ascending=False, inplace=True)
    # Calculate cumulative sum of events after the particular value in any given row
    temp_df['events_after'+suffix] = temp_df['events'].cumsum()
    # Remove overlap from current row
    temp_df['events_after'+suffix] = temp_df['events_after'+suffix] - temp_df['events']
    # Calculate cumulative sum of churners after the particular value in any given row
    temp_df['churners_after'+suffix] = temp_df['churn_m1'].cumsum()
    # Remove overlap from current row
    temp_df['churners_after'+suffix] = temp_df['churners_after'+suffix] - temp_df['churn_m1']
    # Calculate the weighted churn rate for the values after the cutoff
    temp_df['churn_after'+suffix] = temp_df['churners_after'+suffix] / temp_df['events_after'+suffix]

    # Calculate churn delta for sorting the right cutoffs
    temp_df['churn_rt_delta'+suffix] = temp_df['churn_before'+suffix] - temp_df['churn_after'+suffix]
    # Rounding the churn delta to negate the less significant differenceson ranking
    temp_df['churn_rt_delta'+suffix] = temp_df['churn_rt_delta'+suffix].apply(lambda x: round(x,5))

    # Calculate the the difference between both remaining bin distributions
    temp_df['bin_pop_delta'+suffix] = abs(temp_df['events_before'+suffix] - temp_df['events_after'+suffix])

    # Sort the values accordingly to find the best candidates
    temp_df.sort_values(by=['churn_rt_delta'+suffix,'bin_pop_delta'+suffix], ascending=[False,True], inplace=True)

    # Filter the resulting dataframe according to specified number of records
    result = temp_df.iloc[:df_size,:].copy(deep=True)

    result.rename(columns={metricname:'cutoff'+suffix},inplace=True)

    # in the case where the above dataframe creation has failed, initialize an empty dataframe
    # except:
    #     result = pd.DataFrame()

    return result
# %% Results

def shortlist_other_cutoffs(df, var, cutoff_1, cutoff_5, cutoff, lo_benchmark, hi_benchmark, lo_bound, hi_bound, suffix, df_size=100, prints=True):
    '''
    1. Subset
    2. sort
    3. calculate cumulative sums
    4. use cumulative sums to calculate churn rates before each point
    5. repeat step 2 to 4 with a reverse sorting order to calculate churn rates after each point
    '''

    if cutoff == 'split_1' or cutoff == 'split_2' or cutoff == 'split_3':
        lb_condition = (df[var] >= lo_bound)
        ub_condition = (df[var] < hi_bound)

    if cutoff == 'split_4':
        lb_condition = (df[var] >= lo_bound)
        ub_condition = (df[var] <= hi_bound)

    try:

        temp_df = df.loc[lb_condition & ub_condition , :].copy(deep=True)
        if prints and len(temp_df) <= 0: print('No possible cutoffs to choose from for cutoff {}'.format(cutoff))

        temp_df.sort_values(by=var, ascending=True, inplace=True)

        temp_df['events_before'+suffix] = temp_df['events'].cumsum()
        temp_df['events_before'+suffix] = temp_df['events_before'+suffix] - temp_df['events']
        temp_df['churners_before'+suffix] = temp_df['churn_m1'].cumsum()
        temp_df['churners_before'+suffix] = temp_df['churners_before'+suffix] - temp_df['churn_m1']
        temp_df['churn_before'+suffix] = temp_df['churners_before'+suffix] / temp_df['events_before'+suffix]

        temp_df.sort_values(by=var, ascending=False, inplace=True)

        temp_df['events_after'+suffix] = temp_df['events'].cumsum()
        temp_df['events_after'+suffix] = temp_df['events_after'+suffix] - temp_df['events']
        temp_df['churners_after'+suffix] = temp_df['churn_m1'].cumsum()
        temp_df['churners_after'+suffix] = temp_df['churners_after'+suffix] - temp_df['churn_m1']
        temp_df['churn_after'+suffix] = temp_df['churners_after'+suffix] / temp_df['events_after'+suffix]

        # check for monotonicity
        if cutoff == 'split_1':
            condition = (temp_df[var] >= cutoff_1) & (temp_df['churn_before'+suffix] > temp_df['churn_after'+suffix])

        if cutoff == 'split_2' or cutoff == 'split_3':
            condition = (temp_df['churn_before'+suffix] < lo_benchmark) & (temp_df['churn_before'+suffix] > temp_df['churn_after'+suffix]) & (temp_df['churn_after'+suffix] > hi_benchmark)

        if cutoff == 'split_4':
            condition = (temp_df[var] <= cutoff_5) & (temp_df['churn_before'+suffix] < lo_benchmark) & (temp_df['churn_before'+suffix] > temp_df['churn_after'+suffix])

        if prints: print('Number of potential monotonic cutoffs for {}: {}'.format(cutoff,sum(condition)))

        shortlist_cutoffs = temp_df.loc[condition, :].copy(deep=True)

        # select best split
        shortlist_cutoffs.sort_values(by='churn_before'+suffix, ascending=False, inplace=True)

        winner = shortlist_cutoffs[[var, 'churn_before'+suffix, 'churn_after'+suffix]].iloc[0]

    except:
        winner = pd.DataFrame(columns = [var, 'churn_before'+suffix, 'churn_after'+suffix])

    return winner

# %%
def load_object(file_name):
        '''DOCSTRING
        '''
        global __version__
        print('This is script version: {}'.format(__version__))
        file_tb = open(file_name,'rb')

        x = pickle.load(file_tb)

        file_tb.close()

        try:
            print('This object was saved with version: ',x.__version__)
        except:
            print('Not a versioned threshold build object')

        return x

# %% [markdown]
#  THRESHOLD BUILD CLASS DEFINITION

# %%
class Threshold_Build:
    ''' Description of Class
    '''

    def __init__(self,
                    channel=None,
                    metricname=None,
                    input_df=None,
                    result_table=pd.DataFrame(),
                    score_dict={},
                    bivariate=None,
                    churn_curve=None,
                    rescaled_curve=None,
                    trivariate=None,
                    cutoff_universe_df = pd.DataFrame(),
                    progress_tracker = pd.DataFrame()
                    ):

        self.channel = channel
        self.metricname = metricname

        self.input_df = input_df
        self.result_table = result_table
        self.score_dict = score_dict

        self.bivariate = bivariate
        self.churn_curve = churn_curve
        self.rescaled_curve = rescaled_curve
        self.trivariate = trivariate

        self.cutoff_universe_df = cutoff_universe_df

        steps_to_take = [
            'Load an input dataframe',
            'Specify the name of the metric',
            'Specify the channel',
            'Plot the underlying relationship',
            'Specify the thresholds to test',
            'Create 1s and 5s',
            'Create 2s and 4s',
            'Check validity of cutoffs',
            'Rank order the cutoffs',
            'Shortlist the valid cutoffs'
        ]

        status = ['Not attempted yet'] * len(steps_to_take)

        output = ['None'] * len(steps_to_take)

        self.progress_tracker = pd.DataFrame(data=zip(steps_to_take,status,output), columns=['step','status','output'])

        if not input_df.empty:
            self.progress_tracker.at[0,'status'] = 'Done at instantiation'
            self.progress_tracker.at[0,'output'] = 'self.input_df'

        if self.metricname:
            self.progress_tracker.at[1,'status'] = 'Done at instantiation'
            self.progress_tracker.at[1,'output'] = 'self.metricname'

        if self.channel:
            self.progress_tracker.at[2,'status'] = 'Done at instantiation'
            self.progress_tracker.at[2,'output'] = 'self.channel'


        global __version__
        self.__version__ = __version__

    def assign_df(self,df,prints=True):
        ''' Assign a pre-existing df to the object

        This resets everything
        '''
        assert isinstance(df,pd.DataFrame)
        self.input_df = df
        #self.percentile_df = create_percentiles(df, self.metricname)

        if prints: print('Resetting result_table and score_dict')
        self.result_table = pd.DataFrame()
        self.score_dict = {}

    def assign_value(self,value,threshold,col):

        self.result_table.loc[self.result_table['threshold_3'] == threshold, col] = value

    def plot_churn(self, bins=None, df=None, metricname=None, plot_type='scatter',title='Churn Plot',prints=False,file_name=None,store_object=False, show_results=True):
        '''Creates graph objects and plots them

        Can return graph objects if specified
        Default is to print underlying bivariates using bins specified

        TODO:
            Add the ability to group all final observation in last bin
            Change labels
            Clean up the layout.
        '''

        if prints:
            print('Creating Graph Objects for Bivariates')

        if metricname == None:
            metricname = self.metricname

        if not isinstance(df,pd.DataFrame):
            df = self.input_df

        if isinstance(bins,str):
            temp_df = df.groupby(bins).agg({'events':'sum','churn_m1':'sum'}).reset_index()
            _x = temp_df[bins]

        else:
            df['bins'] = pd.cut(df[metricname],bins=bins,include_lowest=True).apply(lambda x: x.right)
            temp_df = df.groupby('bins').agg({metricname:'mean','events':'sum','churn_m1':'sum'}).reset_index()
            df.drop(columns='bins', inplace=True)
            _x = temp_df['bins']

        temp_df['churn_rate_m1'] = temp_df['churn_m1'] / temp_df['events']


        if plot_type == 'scatter':
            churn = go.Scatter(
                x = _x
                , y = temp_df['churn_rate_m1']
                , mode = 'markers'
            )
        elif plot_type == 'lines':
            churn = go.Scatter(
                x = _x
                , y = temp_df['churn_rate_m1']
                , mode = 'lines+markers'
            )
        elif plot_type == 'bars':
             churn = go.Scatter(
                x = temp_df[metricname].astype(str)
                , y = temp_df['churn_rate_m1']
                , mode = 'lines+markers'
            )

        population = go.Histogram(
            x = _x
            , y = temp_df['events']
            #, histnorm='probability'
            , opacity=0.5
            , nbinsx=100
            , histfunc='sum'
            , yaxis = 'y2'

        )

        bivariate = [churn, population]

        if prints:
            print('Plotting Graph Objects for Bivariates')

        layout = go.Layout(
            title=title
            , yaxis1 = dict(
                title='Churn Rates'
            )
            , yaxis2 = dict(
                title='Event Frequency Distribution'
                , overlaying='y'
                , side='right'
            )
        )

        if show_results:
            init_notebook_mode(connected=True)
            fig = go.Figure(data=bivariate, layout=layout)
            if file_name:
                iplot(fig, filename=file_name)
            else:
                iplot(fig) #, filename='something')

        if store_object:
            bivariate_graph = dcc.Graph(
                figure=go.Figure(data=bivariate,
                                layout=layout
                ),
                config={"displayModeBar": False}
            )

            self.bivariate = bivariate_graph

    def create_3(self,list_of_thresholds,prints=True):
        '''DOCTSTRING
        '''

        #list_of_thresholds = [i for i in range(start,stop,skip)] Deprecated

        events = []
        churn = []

        for threshold in list_of_thresholds:
            # Find the number of events exactly at the value specified for the threshold
            count = self.input_df.loc[self.input_df[self.metricname] == threshold, 'events']
            if len(count) <= 0:
                count = 0
            else:
                count = int(count)
            events.append(count)

            churners = self.input_df.loc[self.input_df[self.metricname] == threshold, 'churn_m1']
            if len(churners) <= 0:
                churners = 0
            else:
                churners = int(churners)
            churn.append(churners)

        results = zip(list_of_thresholds,events,churn)

        if prints: print('Resetting result_table')
        self.result_table = pd.DataFrame(results, columns=['threshold_3','events_3','churn_3'])
        self.result_table['threshold_validity'] = True
        self.result_table['threshold_validity_2_4'] = True
        self.result_table['threshold_validity_2'] = True
        self.result_table['threshold_validity_4'] = True

    def create_1_5(self, cutoff_pct_1=0.1, cutoff_pct_5=0.9, split=False, prints=True):
        '''DOCSTRING

        NOTE: Dropped the for loop over thresholds for time-efficiency. We can re-implement it if we decide to split the observations later
        '''

        #for threshold in self.result_table['threshold_3']:

        _churn_1 = 0
        for c in range(1,11):

            cutoff_pct_1 = c / 100
            print(f'Cutoff percent: {cutoff_pct_1}')

            cutoff_1 = find_cutoff_value(self.input_df,self.metricname,cutoff_pct_1)
            churn_1 = find_churn_rate(self.input_df,self.metricname,upper_bound=cutoff_1)

            if churn_1 > _churn_1:
                # Set a new benchmark to beat
                _churn_1 = churn_1

                # Update the cutoff value (cutoff_1), the percentile of that cutoff (percentile_1), and the churn rate (churn_rt_1)
                self.result_table['cutoff_1'] = cutoff_1
                self.result_table['percentile_1'] = cutoff_pct_1
                self.result_table['churn_rt_1'] = churn_1

                # Update the number of events that have the exact score of 1
                events_1 = self.input_df.loc[self.input_df[self.metricname] <= cutoff_1, 'events'].sum()
                events_1 = int(events_1)
                self.result_table['events_1'] = events_1

                # Update the number for churners that have a score of 1
                churners_1 = self.input_df.loc[self.input_df[self.metricname] <= cutoff_1, 'churn_m1'].sum()
                churners_1 = int(churners_1)
                self.result_table['churn_1'] = churners_1

        _churn_5 = 1
        for c in range(90,100):

            cutoff_pct_5 = c / 100

            cutoff_5 = find_cutoff_value(self.input_df,self.metricname,cutoff_pct_5)
            churn_5 = find_churn_rate(self.input_df,self.metricname,lower_bound=cutoff_5)

            if churn_5 < _churn_5 and churn_5 != 0:
                # Set a new benchmark to beat
                _churn_5 = churn_5

                # Update the cutoff value (cutoff_5), the percentile of that cutoff (percentile_5), and the churn rate (churn_rt_5)
                self.result_table['cutoff_5'] = cutoff_5
                self.result_table['percentile_5'] = cutoff_pct_5
                self.result_table['churn_rt_5'] = churn_5

                # Update the number of events that have the exact score of 1
                events_5 = self.input_df.loc[self.input_df[self.metricname] >= cutoff_5, 'events'].sum()
                events_5 = int(events_5)
                self.result_table['events_5'] = events_5

                # Update the number for churners that have a score of 1
                churners_5 = self.input_df.loc[self.input_df[self.metricname] >= cutoff_5, 'churn_m1'].sum()
                churners_5 = int(churners_5)
                self.result_table['churn_5'] = churners_5

    def create_1_5_wen(self, cutoff_pct_1, cutoff_pct_5, split=False, prints=True):
        '''DOCSTRING

        NOTE: Dropped the for loop over thresholds for time-efficiency. We can re-implement it if we decide to split the observations later
        '''

#         for threshold in self.result_table['threshold_3']:
        print('--- threshold 1 ---')
        _churn_1 = 0
        for c in np.arange(0,int(cutoff_pct_1*100)+1,1):

            print(f'Cutoff percent: {c/100}')

            cutoff_1 = find_cutoff_value(self.input_df,self.metricname,c/100)
            churn_1 = find_churn_rate(self.input_df,self.metricname,upper_bound=cutoff_1)

            if churn_1 > _churn_1:
                # Set a new benchmark to beat
                _churn_1 = churn_1

                # Update the cutoff value (cutoff_1), the percentile of that cutoff (percentile_1), and the churn rate (churn_rt_1)
                self.result_table['cutoff_1'] = cutoff_1
                self.result_table['percentile_1'] = cutoff_pct_1
                self.result_table['churn_rt_1'] = churn_1

                # Update the number of events that have the exact score of 1
                events_1 = self.input_df.loc[self.input_df[self.metricname] <= cutoff_1, 'events'].sum()
                events_1 = int(events_1)
                self.result_table['events_1'] = events_1

                # Update the number for churners that have a score of 1
                churners_1 = self.input_df.loc[self.input_df[self.metricname] <= cutoff_1, 'churn_m1'].sum()
                churners_1 = int(churners_1)
                self.result_table['churn_1'] = churners_1

        print('--- threshold 5 ---')
        _churn_5 = 1
        for c in np.arange(int(cutoff_pct_5*100),100, 1):

            print(f'Cutoff percent: {c/100}')

            cutoff_5 = find_cutoff_value(self.input_df,self.metricname,c/100)
            churn_5 = find_churn_rate(self.input_df,self.metricname,lower_bound=cutoff_5)

            if churn_5 < _churn_5 and churn_5 != 0:
                # Set a new benchmark to beat
                _churn_5 = churn_5

                # Update the cutoff value (cutoff_5), the percentile of that cutoff (percentile_5), and the churn rate (churn_rt_5)
                self.result_table['cutoff_5'] = cutoff_5
                self.result_table['percentile_5'] = cutoff_pct_5
                self.result_table['churn_rt_5'] = churn_5

                # Update the number of events that have the exact score of 1
                events_5 = self.input_df.loc[self.input_df[self.metricname] >= cutoff_5, 'events'].sum()
                events_5 = int(events_5)
                self.result_table['events_5'] = events_5

                # Update the number for churners that have a score of 1
                churners_5 = self.input_df.loc[self.input_df[self.metricname] >= cutoff_5, 'churn_m1'].sum()
                churners_5 = int(churners_5)
                self.result_table['churn_5'] = churners_5

    def create_2_4(self,prints=True):
        '''DOCTSTRING
        '''

        # initialize a dataframe to store all possible combinations of 2 and 4
        cutoff_universe_df = pd.DataFrame()

        # loop through each threshold to shortlist possible 2s and 4s
        for threshold in self.result_table['threshold_3']:
            if prints: print('Threshold: {}'.format(threshold))

            row = self.result_table.loc[self.result_table['threshold_3'] == threshold]

            # Shortlisting cutoffs for 2:
            cutoff_1 = int(row['cutoff_1'])
            churners_1 = int(row['churn_1'])
            events_1 = int(row['events_1'])
            churn_rt_1 = float(row['churn_rt_1'])
            potential_2s = shortlist_cutoffs(
                df = self.input_df,
                metricname = self.metricname,
                cutoff = 2,
                benchmark = churn_rt_1,
                lo_bound = cutoff_1,
                hi_bound = threshold,
                suffix = '_2',
                df_size = 100,
                prints = True
            )
            potential_2s['key'] = 1

            # Shortlisting cutoffs for 4:

            cutoff_5 = int(row['cutoff_5'])
            churners_5 = int(row['churn_5'])
            events_5 = int(row['events_5'])
            churn_rt_5 = float(row['churn_rt_5'])
            potential_4s = shortlist_cutoffs(
                df = self.input_df,
                metricname = self.metricname,
                cutoff = 4,
                benchmark = churn_rt_5,
                lo_bound = threshold,
                hi_bound = cutoff_5,
                suffix = '_4',
                df_size = 100,
                prints = True
            )

            potential_4s['key'] = 1

            # Shortlist all potential cutoffs for 2 and 4 that ensure monotonicity:
            cartesian = potential_2s.merge(potential_4s, on='key', suffixes=('_2','_4'), how='inner')
            try:
                # create a monotonicity flag
                cartesian['monotonicity'] = np.where(cartesian['churn_after_2'] > cartesian['churn_before_4'], True, False)
                if prints: print('Size of filtered cartesian df: {}'.format(len(cutoff_shortlist)))
            except:
                cutoff_shortlist = pd.DataFrame()
                if prints: print('Size of filtered cartesian df: {}'.format(len(cutoff_shortlist)))

            # Check validity of our cutoffs: - approaches will differ based on this
            valid_2 = len(potential_2s) > 0
            valid_4 = len(potential_4s) > 0
            # valid_2_4 can only be true if cartesian exists
            if len(cartesian) > 0:
                valid_2_4 = len(cartesian.loc[cartesian['monotonicity'] == True,:]) > 0
            else:
                valid_2_4 = False

            # Case 1: We have monotonicity for 2s and 4s
            if valid_2_4:
                # create a field with % of events in the biggest bin
                cartesian['max_bin_perc'] = cartesian.apply(lambda x:
                                    max([
                                        x['events_before_4'],
                                        x['events_after_4'],
                                        x['events_before_2'],
                                        x['events_after_2']])
                                    / sum([
                                        x['events_before_4'],
                                        x['events_after_4'],
                                        x['events_before_2'],
                                        x['events_after_2']])
                                    , axis=1)

                                # create a field with % of events in all bins except the biggest one
                cartesian['remaining_bin_perc'] = 1 - cartesian['max_bin_perc']

                # create a field calculating multiplier effect between first and last bins
                # cartesian['max_multiplier'] = cartesian.apply(lambda x: x['churn_before_2'] / x['churn_before_4'], axis=1)
                # I commented this out to stop it from breaking the script
                cartesian['max_multiplier'] = 'Temporarily unavailable'

                # add the current threshold to the cartesian df
                cartesian['threshold'] = threshold
                cartesian['events_3'] = int(row['events_3'])
                cartesian['churn_3'] = int(row['churn_3'])

                # add the data about cutoff 1 to the current df
                cartesian['cutoff_1'] = cutoff_1
                cartesian['events_1'] = events_1
                cartesian['churn_1'] = churners_1

                # add the data about cutoff 5 to the current df
                cartesian['cutoff_5'] = cutoff_5
                cartesian['events_5'] = events_5
                cartesian['churn_5'] = churners_5

                # only select those rows where monotonicity criteria is true
                cutoff_shortlist = cartesian.loc[cartesian['monotonicity'] == True,:].copy(deep=True)

                # save the cartesian to store all possible combinations for all thresholds
                # keeping only necessary fields for debugging if required
                if cutoff_universe_df.empty:
                    print('Big table is empty... Initializing')
                    cutoff_universe_df = cartesian[[
                        # Cutoff Values
                        'threshold',
                        'cutoff_1',
                        'cutoff_2',
                        'cutoff_4',
                        'cutoff_5',
                        # Events (Absolute number)
                        'events_1',
                        'events_before_2',
                        'events_2',
                        'events_after_2',
                        'events_3',
                        'events_before_4',
                        'events_4',
                        'events_after_4',
                        'events_5',
                        # Churners (absolute number)
                        'churn_1',
                        'churners_before_2',
                        'churn_m1_2',
                        'churners_after_2',
                        'churn_3',
                        'churners_before_4',
                        'churn_m1_4',
                        'churners_after_4',
                        'churn_5',
                        # Other metrics
                        'max_bin_perc',
                        'remaining_bin_perc',
                        'max_multiplier']].copy(deep=True)
                    print('Size of big table: {}'.format(len(cutoff_universe_df)))
                else:
                        # append if df already exists
                        print("Big table is not empty... Appending")
                        cutoff_universe_df = cutoff_universe_df.append(cartesian[[
                            # Cutoff Values
                            'threshold',
                            'cutoff_1',
                            'cutoff_2',
                            'cutoff_4',
                            'cutoff_5',
                            # Events (Absolute number)
                            'events_1',
                            'events_before_2',
                            'events_2',
                            'events_after_2',
                            'events_3',
                            'events_before_4',
                            'events_4',
                            'events_after_4',
                            'events_5',
                            # Churners (absolute number)
                            'churn_1',
                            'churners_before_2',
                            'churn_m1_2',
                            'churners_after_2',
                            'churn_3',
                            'churners_before_4',
                            'churn_m1_4',
                            'churners_after_4',
                            'churn_5',
                            # Other metrics
                            'max_bin_perc',
                            'remaining_bin_perc',
                            'max_multiplier']],
                            ignore_index=True)
                        print("New size of big table: {}".format(len(cutoff_universe_df)))

                # best 2 and 4 combination is decided by the multiplier for each threshold
                cutoff_shortlist.sort_values(by='max_multiplier', ascending=False, inplace=True)

                winner = cutoff_shortlist.iloc[0,:]

                self.assign_value(winner['cutoff_2'],threshold,'cutoff_2')
                self.assign_value(winner['churn_before_2'],threshold,'churn_before_2')
                self.assign_value(winner['events_before_2'],threshold,'events_before_2')
                self.assign_value(winner['churn_after_2'],threshold,'churn_after_2')
                self.assign_value(winner['events_after_2'],threshold,'events_after_2')

                self.assign_value(winner['cutoff_4'],threshold,'cutoff_4')
                self.assign_value(winner['churn_before_4'],threshold,'churn_before_4')
                self.assign_value(winner['events_before_4'],threshold,'events_before_4')
                self.assign_value(winner['churn_after_4'],threshold,'churn_after_4')
                self.assign_value(winner['events_after_4'],threshold,'events_after_4')

            else:
                # error handling in case of no valid combinations
                self.assign_value(False,threshold,'threshold_validity_2_4')
                self.assign_value(False,threshold,'threshold_validity') # Negate overall threshold validity
                if prints: print('No cutoffs found that were monotonic between 2 and 4 for threshold: {}'.format(threshold))

                # Placeholder cutoffs are calculated here
                temp_cutoff_2 = int((threshold - cutoff_1) / 2) + cutoff_1
                temp_cutoff_4 = int((cutoff_5 - threshold) / 2) + threshold

            # Case 2: We have a cutoff for 2 but not 4:
            if valid_2 and not valid_4:
                potential_2s['churn_delta'] = potential_2s['churn_before_2'] - potential_2s['churn_after_2']
                potential_2s.sort_values(by='churn_delta', ascending=False,inplace=True)
                winner = potential_2s.iloc[0,:]

                self.assign_value(winner['cutoff_2'],threshold,'cutoff_2')
                self.assign_value(winner['churn_before_2'],threshold,'churn_before_2')
                self.assign_value(winner['events_before_2'],threshold,'events_before_2')
                self.assign_value(winner['churn_after_2'],threshold,'churn_after_2')
                self.assign_value(winner['events_after_2'],threshold,'events_after_2')

                self.assign_value(temp_cutoff_4,threshold,'cutoff_4')

                self.assign_value(False,threshold,'threshold_validity') # Negate overall threshold validity
                self.assign_value(False,threshold,'threshold_validity_4')

                if prints: print('Monotonic cutoffs for 2 but not 4 were found for threshold: {}'.format(threshold))


            # Case 3: We have a cutoff for 4 but not 2:
            elif not valid_2 and valid_4:
                potential_4s['churn_delta'] = potential_4s['churn_before_4'] - potential_4s['churn_after_4']
                potential_4s.sort_values(by='churn_delta', ascending=False,inplace=True)
                winner = potential_4s.iloc[0,:]

                self.assign_value(winner['cutoff_4'],threshold,'cutoff_4')
                self.assign_value(winner['churn_before_4'],threshold,'churn_before_4')
                self.assign_value(winner['events_before_4'],threshold,'events_before_4')
                self.assign_value(winner['churn_after_4'],threshold,'churn_after_4')
                self.assign_value(winner['events_after_4'],threshold,'events_after_4')

                self.assign_value(temp_cutoff_2,threshold,'cutoff_2')

                self.assign_value(False,threshold,'threshold_validity') # Negate overall threshold validity
                self.assign_value(False,threshold,'threshold_validity_2')

                if prints: print('Monotonic cutoffs for 4 but not 2 were found for threshold: {}'.format(threshold))

            # Case 4: We have valid 2s and 4s, but no monotonicity between them
            elif valid_2 and valid_4 and not valid_2_4:
                potential_2s['churn_delta'] = potential_2s['churn_before_2'] - potential_2s['churn_after_2']
                potential_2s.sort_values(by='churn_delta', ascending=False,inplace=True)
                winner_2 = potential_2s.iloc[0,:]

                self.assign_value(winner_2['cutoff_2'],threshold,'cutoff_2')
                self.assign_value(winner_2['churn_before_2'],threshold,'churn_before_2')
                self.assign_value(winner_2['events_before_2'],threshold,'events_before_2')
                self.assign_value(winner_2['churn_after_2'],threshold,'churn_after_2')
                self.assign_value(winner_2['events_after_2'],threshold,'events_after_2')

                potential_4s['churn_delta'] = potential_4s['churn_before_4'] - potential_4s['churn_after_4']
                potential_4s.sort_values(by='churn_delta', ascending=False,inplace=True)
                winner_4 = potential_4s.iloc[0,:]

                self.assign_value(winner_4['cutoff_4'],threshold,'cutoff_4')
                self.assign_value(winner_4['churn_before_4'],threshold,'churn_before_4')
                self.assign_value(winner_4['events_before_4'],threshold,'events_before_4')
                self.assign_value(winner_4['churn_after_4'],threshold,'churn_after_4')
                self.assign_value(winner_4['events_after_4'],threshold,'events_after_4')

                self.assign_value(False,threshold,'threshold_validity') # Negate overall threshold validity
                self.assign_value(False,threshold,'threshold_validity_2_4')

                if prints: print('We have valid 2s and 4s, but no monotonicty between them for threshold: {}'.format(threshold))

            # Case 5 we have no cutoffs at all
            elif not valid_2 and not valid_4 and not valid_2_4:
                self.assign_value(temp_cutoff_2,threshold,'cutoff_2')
                self.assign_value(temp_cutoff_4,threshold,'cutoff_4')

                self.assign_value(False,threshold,'threshold_validity') # Negate overall threshold validity
                self.assign_value(False,threshold,'threshold_validity_2')
                self.assign_value(False,threshold,'threshold_validity') # Negate overall threshold validity
                self.assign_value(False,threshold,'threshold_validity_4')

                if prints: print('No monotonic threshold were found at all for threshold: {}'.format(threshold))

            else:
                if valid_2_4: print('This condition picked up completely valid cutoffs')
                else: print('Mistake in Logic')

        # Make the cutoff universe dataframe persist within the class
        self.cutoff_universe_df = cutoff_universe_df

        self.cutoff_universe_df.rename(columns={
            'events_before_2':'events_btwn_1_2',
            'events_after_2':'events_btwn_2_3',
            'events_before_4':'events_btwn_3_4',
            'events_after_4':'events_btwn_4_5',
            'churn_1':'churners_1',
            'churners_before_2':'churners_btwn_1_2',
            'churn_m1_2':'churners_2',
            'churners_after_2':'churners_btwn_2_3',
            'churn_3':'churners_3',
            'churners_before_4':'churners_btwn_3_4',
            'churn_m1_4':'churners_4',
            'churners_after_4':'churners_btwn_4_5',
            'churn_5':'churners_5'
        }, inplace=True)

    def create_other_cutoffs(self,prints=True): # Method created by Chan for network
        '''DOCTSTRING

        TODO: function to create cutoffs for 8 bins
        '''

        for threshold in self.result_table['threshold_3']:
            if prints: print('Threshold: {}'.format(threshold))

            row = self.result_table.loc[self.result_table['threshold_3'] == threshold]

            min_var = min(self.input_df[self.metricname])
            max_var = max(self.input_df[self.metricname])

            # Shortlisting cutoffs for 8 bins:
            cutoff_1 = int(row['cutoff_1'])
            cutoff_2 = int(row['cutoff_2'])
            cutoff_3 = int(row['threshold_3'])
            cutoff_4 = int(row['cutoff_4'])
            cutoff_5 = int(row['cutoff_5'])

            churn_1 = float(row['churn_rt_1'])
            churn_before_2 = float(row['churn_before_2'])
            churn_after_2 = float(row['churn_after_2'])
            churn_before_4 = float(row['churn_before_4'])
            churn_after_4 = float(row['churn_after_4'])
            churn_5 = float(row['churn_rt_5'])

            potential_split_1 = shortlist_other_cutoffs(df = self.input_df, var = self.metricname, cutoff_1 = cutoff_1,
                                                        cutoff_5 = cutoff_5, cutoff = 'split_1',
                                                        hi_benchmark = churn_after_2, lo_benchmark = churn_1, lo_bound = min_var,
                                                        hi_bound = cutoff_2, suffix = '_split_1')
            print(potential_split_1)

            if potential_split_1.empty:
                potential_split_2 = shortlist_other_cutoffs(df = self.input_df, var = self.metricname, cutoff_1 = cutoff_1,
                                                            cutoff_5 = cutoff_5, cutoff = 'split_2',
                                                            hi_benchmark = churn_before_4, lo_benchmark = churn_before_2,
                                                            lo_bound = cutoff_2, hi_bound = cutoff_3, suffix = '_split_2')

            else:
                churn_after_split_1 = float(potential_split_1['churn_after_split_1'])
                potential_split_2 = shortlist_other_cutoffs(df = self.input_df, var = self.metricname, cutoff_1 = cutoff_1, cutoff_5 = cutoff_5, cutoff = 'split_2',
                                                            hi_benchmark = churn_before_4, lo_benchmark = churn_after_split_1, lo_bound = cutoff_2,
                                                            hi_bound = cutoff_3, suffix = '_split_2')

    #             print(potential_split_2)

            if potential_split_2.empty:
                potential_split_3 = shortlist_other_cutoffs(df = self.input_df, var = self.metricname, cutoff_1 = cutoff_1, cutoff_5 = cutoff_5, cutoff = 'split_3',
                                                            hi_benchmark = churn_after_4, lo_benchmark = churn_after_2, lo_bound = cutoff_3,
                                                            hi_bound = cutoff_4, suffix = '_split_3')

            else:
                churn_after_split_2 = float(potential_split_2['churn_after_split_2'])
                potential_split_3 = shortlist_other_cutoffs(df = self.input_df, var = self.metricname, cutoff_1 = cutoff_1, cutoff_5 = cutoff_5, cutoff = 'split_3',
                                                            hi_benchmark = churn_after_4, lo_benchmark = churn_after_split_2, lo_bound = cutoff_3,
                                                            hi_bound = cutoff_4, suffix = '_split_3')

    #             print(potential_split_3)

            if potential_split_3.empty:
                potential_split_4 = shortlist_other_cutoffs(df = self.input_df, var = self.metricname, cutoff_1 = cutoff_1, cutoff_5 = cutoff_5, cutoff = 'split_4',
                                                            hi_benchmark = churn_5,lo_benchmark = churn_before_4, lo_bound = cutoff_4,
                                                            hi_bound = max_var, suffix = '_split_4')

            else:
                churn_after_split_3 = float(potential_split_3['churn_after_split_3'])
                potential_split_4 = shortlist_other_cutoffs(df = self.input_df, var = self.metricname, cutoff_1 = cutoff_1, cutoff_5 = cutoff_5, cutoff = 'split_4',
                                                            hi_benchmark = churn_5, lo_benchmark = churn_after_split_3, lo_bound = cutoff_4,
                                                            hi_bound = max_var, suffix = '_split_4')

    #             print(potential_split_4)

            self.assign_value(potential_split_1[self.metricname], threshold,'split_1')
            self.assign_value(potential_split_2[self.metricname], threshold,'split_2')
            self.assign_value(potential_split_3[self.metricname], threshold,'split_3')
            self.assign_value(potential_split_4[self.metricname], threshold,'split_4')

            self.assign_value(potential_split_1['churn_before_split_1'],threshold,'churn_before_split_1')
            self.assign_value(potential_split_2['churn_before_split_2'],threshold,'churn_before_split_2')
            self.assign_value(potential_split_3['churn_before_split_3'],threshold,'churn_before_split_3')
            self.assign_value(potential_split_4['churn_before_split_4'],threshold,'churn_before_split_4')

            self.assign_value(potential_split_1['churn_after_split_1'],threshold,'churn_after_split_1')
            self.assign_value(potential_split_2['churn_after_split_2'],threshold,'churn_after_split_2')
            self.assign_value(potential_split_3['churn_after_split_3'],threshold,'churn_after_split_3')
            self.assign_value(potential_split_4['churn_after_split_4'],threshold,'churn_after_split_4')

    def cutoff_validity_checks(
            self,
            input_df = None,
            num_bins = 4,
            ordered_bin_suffixes = [],
            check_monotonic = True,
            check_pop_size = True,
            min_pop_size = None):
        """Function that checks for the different validity criteria of cutoffs across the universe

        Parameters:
        input_df (df): A table structured in the same way as the cutoff_universe_df
        num_bins (df): The number of bins we are testing for validity. Usually 4 or 8
        ordered_bin_suffixes (df): The suffixes of descriptor fields in the cutoff_universe_df for each bin.
            Must be ordered in descending order by bin's expected churn rate: ['high_churn_bin', 'low_churn_bin']
            Default: []
        check_monotonic (df): Choose whether to perform monotonicity check. Default: True
        check_pop_size (df): Choose whether to perform population size check. Default: True
        min_pop_size (df): The minimum population in each bin. Required for check_pop_size and must be set. Default: None

        Returns:
        An updated input_df. No return call specified in the function itself.

        """

        # assign cutoff_universe_df if not defined
        if not isinstance(input_df, pd.DataFrame):
            input_df = self.cutoff_universe_df

        # check that there are between 4 and 8 bins
        if num_bins > 8 or num_bins < 4:
            raise Exception('This function can only work for 4 to 8 bins')

        # check if population size flag is enabled
        if check_pop_size and min_pop_size == None:
            raise Exception('Please have provide min_pop_size criteria if check_pop_size test is enabled')

        # check that bin suffixes are populated
        if ordered_bin_suffixes == None:
            raise Exception("""Provide bin suffixes (in order of expected churn rate) in df
                            ie [\'_2\', \'_3\']""")

        # check that number of suffixes = number of bins
        if len(ordered_bin_suffixes) != num_bins:
            raise Exception('There is a mismatch between number of suffixes and number of bins')

        # check that monotonicity holds for each bin through churn rate in descending order
        if check_monotonic:
            # clean up if this was run previously (will retain previous results unless flag is tagged)
            print('Remove old monotonic flag...')
            if 'is_monotonic' in input_df.columns:
                input_df.drop(columns = 'is_monotonic', inplace = True)
            print('Remove old monotonic flag...Done.')
            print('Checking each row for monotonicity...')
            # use suffix to find actual churn rate field in the dataframe
            churn_rate_field_list = ('churn_rt_' + bin_i for bin_i in ordered_bin_suffixes)
            # extract the bin churn rate values from df into a list
            churn_rate_list = [input_df[field_name] for field_name in churn_rate_field_list]
            # calculate the ratio between each consecutive bin
            # if churn rate monotonic and ratio is > 1, flag it as 1 and sum, which will always be more than number of bins-1
            ratio_list = [np.where((higher_churn_bin / lower_churn_bin) > 1, 1, 0) for higher_churn_bin, lower_churn_bin in zip(churn_rate_list, churn_rate_list[1:])]
            ratio_flag_list = np.where(sum(ratio_list) > (len(ratio_list) - 1), 'Y', 'N')
            input_df['is_monotonic'] = ratio_flag_list
            print('Checking each row for monotonicity...Done.')

        # check that each bin qualifies for the population size criteria
        if check_pop_size:
            # clean up if this was run previously (will retain previous results unless flag is tagged)
            print('Remove old minimum population flag...')
            if 'is_valid_bin_size' in input_df.columns:
                input_df.drop(columns = 'is_valid_bin_size', inplace = True)
            print('Remove old minimum population flag...Done.')
            print('Checking each row for minimum population criteria...')
            # use suffix to find actual event rate field in the dataframe
            event_size_field_list = ('events_' + bin_i for bin_i in ordered_bin_suffixes)
            # extract the bin event rate values from df into a list
            event_size_list = [input_df[field_name] for field_name in event_size_field_list]
            # validate that each bin's size qualifies against the minimum criteria
            pop_size_validity_list = [np.where(min(row) > int(min_pop_size), 'Y', 'N') for row in np.transpose(event_size_list)]
            pop_size_validity_list_clean = [row.item() for row in pop_size_validity_list]
            input_df['is_valid_bin_size'] = pop_size_validity_list_clean
            print('Checking each row for minimum population criteria...Done.')


    def create_bins(
            self,
            input_df = None,
            bin_definitions = {}):
        """Function that creates bin from all the different events that are available.

        Allows for freedom to choose which bounds are included in each bins (ie upper bound or lower bound)

        Parameters:
        input_df (df): A table structured in the same way as the cutoff_universe_df.
            Requires the event counts to be in 'events_' fields and churner counts to be in 'churners_' fields
        bin_definitions (dict): A dictionary of bin definitions.
            Input the bin name suffix (ie bin_1) as the key and a list of field suffixes as a value.
            ie.     bin_definitions = {
                        'bin_1':['before_2', 'after_2', 'before_4', 'after_4'],
                        'bin_2':['before_2', 'after_2']}

        Returns:
        An updated input_df. No return call specified in the function itself.

        """

        if not isinstance(input_df, pd.DataFrame):
            input_df = self.cutoff_universe_df

        # create bin fields in the input_df - loop over number of bins (split into two loops to group like fields in output)
        for bin_name, suffixes in bin_definitions.items():
            ##################
            ###   EVENTS   ###
            ##################
            # create the field of total events for the bin
            bin_events_field_name = 'events_' + bin_name
            events_fields_names = ['events_' + suffix for suffix in suffixes]
            events_list = [input_df[field_name] for field_name in events_fields_names]
            # sum together the different sections to make the bin
            sum_events_list = [sum(event_i) for event_i in np.transpose(events_list)]
            input_df[bin_events_field_name] = sum_events_list

        for bin_name, suffixes in bin_definitions.items():
            ##################
            ###  CHURNERS  ###
            ##################
            # create the field of total churners for the bin
            bin_churners_field_name = 'churners_' + bin_name
            churners_fields_names = ['churners_' + suffix for suffix in suffixes]
            churners_list = [input_df[field_name] for field_name in churners_fields_names]
            # sum together the different sections to make the bin
            sum_churners_list = [sum(event_i) for event_i in np.transpose(churners_list)]
            input_df[bin_churners_field_name] = sum_churners_list

        for bin_i in bin_definitions:
            ##################
            ### CHURN RATE ###
            ##################
            bin_churn_rate_field_name = 'churn_rt_' + bin_i
            events_fields_names = 'events_' + bin_i
            churners_fields_names = 'churners_' + bin_i
            # calcualte churn rate
            input_df[bin_churn_rate_field_name] = input_df[churners_fields_names] / input_df[events_fields_names]

    def apply_cutoffs(self, direction='proportional'):
        '''DOCSTRING

        TODO: Add a criteria for threshold validity
        '''
        assert len(self.result_table) > 0

        #score_dict = {}
        temp_df = self.input_df.copy(deep=True)
        metricname = self.metricname

        mn = min(self.input_df[metricname])
        mx = max(self.input_df[metricname])

        for threshold in self.result_table['threshold_3']:
            #print('Applying Cutoffs to Calculate Score for Threshold: {}'.format(threshold))
            row = self.result_table.loc[self.result_table['threshold_3'] == threshold,:]

            one = float(row['cutoff_1'])
            two = float(row['cutoff_2'])
            three = float(threshold)
            four = float(row['cutoff_4'])
            five = float(row['cutoff_5'])

            split_1 = float(row['split_1'])
            split_2 = float(row['split_2'])
            split_3 = float(row['split_3'])
            split_4 = float(row['split_4'])

            list_of_cutoffs = [one,two,three,four,five,split_1,split_2,split_3,split_4]

            # Clean the None Values submitted in the list of cutoffs
            cleaned = [i for i in list_of_cutoffs if i] + [mx]

            # Creating a score lookup table
            tbl = pd.DataFrame(cleaned, columns=['cutoffs_ub'])

            if direction == 'proportional':

                tbl.sort_values(by='cutoffs_ub', inplace=True, ascending = True)
                tbl['cutoffs_lb'] = tbl.cutoffs_ub.shift(1).fillna(mn - 0.1)

                tbl['upper_score'] = 100 / (len(tbl) - 2)
                tbl['upper_score'] = tbl.upper_score.cumsum()
                tbl['upper_score'] = tbl['upper_score'].shift(1).fillna(1)
                tbl['lower_score'] = tbl.upper_score.shift(1).fillna(1)


                # Adjusting range values for the values that are found above the last cutoff
                temp_low_score = max(tbl.loc[tbl['lower_score'] != max(tbl['lower_score']),'lower_score'])
                tbl.loc[tbl.lower_score == max(tbl.lower_score), 'lower_score'] = temp_low_score
                tbl.loc[tbl.upper_score == max(tbl.upper_score), 'upper_score'] = 100

                # Adjusting range values for the values that are found below the first cutoff
                temp_high_score = min(tbl.loc[tbl['upper_score'] != min(tbl['upper_score']),'upper_score'])
                tbl.loc[tbl.upper_score == min(tbl.upper_score), 'upper_score'] = temp_high_score

                # Creating the conditino for the range-join
                values = temp_df[self.metricname].values
                ub = tbl.cutoffs_ub.values
                lb = tbl.cutoffs_lb.values

                i, j = np.where((values[:, None] > lb) & (values[:, None] <= ub))

                scored = pd.DataFrame(
                    np.column_stack([temp_df.values[i], tbl.values[j]]),
                    columns=temp_df.columns.append(tbl.columns)
                )

                # Create labels for the score bins
                scored['bin_labels'] = scored.apply(lambda x: str(round(x.lower_score,2)) + ' to ' + str(round(x.upper_score,2)), axis=1)

                # Calculate the score and recast to float
                scored['score'] = scored.lower_score + (scored[metricname] - scored.cutoffs_lb) / (scored.cutoffs_ub - scored.cutoffs_lb)
                scored['score'] = scored['score'].astype(float)

                # Manually adjust scores that should be 1 or 100
                scored.loc[scored[metricname] >= five, 'score'] = 100
                scored.loc[scored[metricname] <= one, 'score'] = 1

                self.score_dict[threshold] = scored

    def define_ranking(
        self,
        ordered_bin_suffixes,
        ranking_formula,
        ranking_column,
        ranking_method = 'dense',
        rounding_factor = None):
        '''Function that creates a ranking column in our cutoff universe table (cutoff_universe_df)

        NOTE: Events and churners exactly at 3 are being considered as 'below threshold'

        Parameters:
        ranking_formula (str): a string passed which specifies the specific ranking formula to apply
            - 'good_v_bad': Calculates rank beased on the weighted churn rate below 3 minus weighted churn rate above 3
            - 'dif_in_extremes': Calculates rank based on weighted churn of the first bin minus weighted churn of the final bin
            - 'min_between_mid_bins': Calculates rank based on the minimum churn delta between the middle bins, i.e. a2 and a3
        ranking_column (str): the column name of the final rank
        ranking_method (str): the pandas method used to determine the rank for groups with identical values
            See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rank.html for more info
            default is 'dense'
        '''

        # First, check to see that the cutoff universe dataframe exists and that it is a pandas dataframe
        assert isinstance(self.cutoff_universe_df, pd.DataFrame), 'Object does not contain a pandas dataframe called cutoff_universe_df'

        # Assign an alias to the output dataframe
        df = self.cutoff_universe_df

        # ranking based on churn rate of the good portion of the curve (above 3) vs bad portion of the curve (below 3)
        if ranking_formula == 'good_v_bad':

            # In order to evenly split bins, the number of bins submitted must be even
            assert len(ordered_bin_suffixes) % 2 == 0, 'Number of bins not equally divisible by two...'

            # Create lists for the events & churner fields using the bin suffixes provided
            bin_events_field_names = ['events_' + bin_name for bin_name in ordered_bin_suffixes]
            bin_churners_field_names = ['churners_' + bin_name for bin_name in ordered_bin_suffixes]

            # Calculate the number of bins to collapse on either side
            nbr_of_bins_to_collapse = int(len(ordered_bin_suffixes) / 2)

            # Sum the list of series associated with the events of the first half of bins
            bad_bins_events = sum([df[field_name] for field_name in bin_events_field_names[:nbr_of_bins_to_collapse]])
            # Sum the list of series associated with the events of the second half of bins
            good_bins_events = sum([df[field_name] for field_name in bin_events_field_names[nbr_of_bins_to_collapse:]])

            # Sum the list of series associated with the churners of the first half of bins
            bad_bin_churners = sum([df[field_name] for field_name in bin_churners_field_names[:nbr_of_bins_to_collapse]])
            # Sum the list of series associated with the churners of the second half of bins
            good_bin_churners = sum([df[field_name] for field_name in bin_churners_field_names[nbr_of_bins_to_collapse:]])

            # Calculate the appropriate churn rates for good & bad
            churn_rt_of_bad_bins = bad_bin_churners / bad_bins_events
            churn_rt_of_good_bins = good_bin_churners / good_bins_events

            # Calculate the churn delta
            df[ranking_formula] = churn_rt_of_bad_bins - churn_rt_of_good_bins
            if rounding_factor  != None:
                df[ranking_formula] = round(df[ranking_formula], rounding_factor)

            # Assign a rank, values ranked in descending order (bigger delta is better)
            df[ranking_column] = df[ranking_formula].rank(method = ranking_method, ascending = False)

        # Ranking method to capture churn delta between extremes
        if ranking_formula == 'dif_in_extremes':

            # In this ranking approach, we're only interested in the first and last bins
            worst_bin = ordered_bin_suffixes[0]
            best_bin = ordered_bin_suffixes[-1]

            # calculate churn delta between extremes
            df[ranking_formula] = df['churn_rt_' + worst_bin] - df['churn_rt_' + best_bin]
            if rounding_factor != None:
                df[ranking_formula] = df[ranking_formula].apply(lambda x: round(x, rounding_factor))

            # Assigning a rank, values ranked in descending order (bigger delta is better)
            df[ranking_column] = df[ranking_formula].rank(method = ranking_method, ascending = False)

        # ranking based on minimizing churn rate difference in the middle of the curve (meant to improve linearity of the curve)
        if ranking_formula == 'min_between_mid_bins':

            # In order to identify middle bins, there must be an even number of bin labels in the submitted list
            nbr_of_bins_submitted = len(ordered_bin_suffixes)
            assert nbr_of_bins_submitted % 2 == 0, 'Number of bins not equally divisible by two...'

            # Reduce the submitted list of bins so that only the middle two remain
            mid_point = int(nbr_of_bins_submitted / 2)
            middle_bin_suffixes = ordered_bin_suffixes[mid_point - 1 : mid_point + 1]

            # Calculate the delta between middle bins, i.e. ABS(a3 - a2)
            df[ranking_formula] = abs(df['churn_rt_' + middle_bin_suffixes[0]] - df['churn_rt_' + middle_bin_suffixes[1]])
            if rounding_factor != None:
                df[ranking_formula] = round(df[ranking_formula], rounding_factor)

            # Assigning a rank, values ranked in ascending order (smaller absolute delta is better)
            df[ranking_column] = df[ranking_formula].rank(method = ranking_method, ascending = True)

        if ranking_formula == 'combined_multiplier':

            # calculate the combined multiplier
            best_bin = ordered_bin_suffixes.pop(-1)
            nbr_of_other_bins = len(ordered_bin_suffixes)
            df[ranking_formula] = sum([df['churn_rt_' + bin_suffix] / df['churn_rt_' + best_bin]
                                    for bin_suffix in ordered_bin_suffixes])
            df[ranking_formula] = df[ranking_formula] / nbr_of_other_bins       # This is for the average

            if rounding_factor != None:
                df[ranking_formula] = round(df[ranking_formula], rounding_factor)

            # Assigning a rank, values ranked in descending order (greater multiplier is better)
            df[ranking_column] = df[ranking_formula].rank(method = ranking_method, ascending = False)

        if ranking_formula == 'pairwise_ratio':

            # Split the bin suffixes into those used as numerators and those used as denominators
            numerator_bins = ['churn_rt_' + bin_suffix for bin_suffix in ordered_bin_suffixes[:-1]]
            denominator_bins = ['churn_rt_' + bin_suffix for bin_suffix in ordered_bin_suffixes[1:]]

            # Store number of bins submitted in order to calculate mean of pairwise multipliers
            nbr_of_bins_submitted = len(ordered_bin_suffixes) - 1

            # calculate the pairwise ratio
            df[ranking_formula] = sum(
                [
                    df[numerator] / df[denominator]
                    for numerator, denominator
                    in zip(numerator_bins,denominator_bins)
                ]
            ) / nbr_of_bins_submitted

            if rounding_factor != None:
                df[ranking_formula] = round(df[ranking_formula], rounding_factor)

            # Assigning a rank, values ranked in descending order (great ratio is better)
            df[ranking_column] = df[ranking_formula].rank(method = ranking_method, ascending = False)

        if ranking_formula == 'sum_square_errors':

            # Split out the best and worst bin to determine straight line of fit, remaining bins will have some deviation from expected values
            worst_bin = ordered_bin_suffixes[0]
            best_bin = ordered_bin_suffixes[-1]
            all_other_bins = ordered_bin_suffixes[1:-1]

            # Calculate the churn increment from worst to best bin expected at each bin in between
            total_nbr_of_bins = len(ordered_bin_suffixes)
            churn_rt_increments = (df['churn_rt_' + worst_bin] - df['churn_rt_' + best_bin]) / total_nbr_of_bins

            # Calculate the expected and actual churn rates for the bin residuals between the extremes
            expected_churn_rates = [df['churn_rt_' + worst_bin] - churn_rt_increments * (bin_nbr + 1)
                                    for bin_nbr in range(len(all_other_bins))]
            actual_churn_rates = [df['churn_rt_' + bin_suffix] for bin_suffix in all_other_bins]

            # Calculate the sum of square errors
            sum_squared_residuals = sum(
                [
                    (actuals - expected) ** 2
                    for actuals, expected
                    in zip(actual_churn_rates, expected_churn_rates)
                ]
            ) * 100000

            # Assign the values to the original df
            df[ranking_formula] = sum_squared_residuals
            if rounding_factor != None:
                df[ranking_formula] = round(df[ranking_formula], rounding_factor)

            # Assign a rank, values ranked in ascending order (the lesser the sum of squared error, the better)
            df[ranking_column] = df[ranking_formula].rank(method = ranking_method, ascending = True)

    def plot_top_scores(self,bins='score_bins',nbr_of_graphs=5):
        '''DOCSTRING
        '''

        if nbr_of_graphs > len(self.result_table['threshold_3']):
            nbr_of_graphs = len(self.result_table['threshold_3'])

        for g in range(nbr_of_graphs):
            ranking = g+1
            threshold = int(self.result_table.loc[self.result_table['rank'] == ranking,'threshold_3'])
            if len(self.score_dict[threshold]['score_bins'].unique()) < 4:
                continue
            self.plot_churn(bins=bins,df=self.score_dict[threshold],metricname='score', plot_type='lines',title='{} Scores\n(Threshold = {} & Rank ={})'.format(self.metricname,threshold,ranking))

    def plot_score(self,threshold,bins='score_bins'):
        '''DOCSTRING
        '''
        self.plot_churn(self.score_dict[threshold],metricname='score',bins=bins, plot_type='lines',title='{} Scores\n(Threshold = {})'.format(self.metricname,threshold))


    def save_object(self,file_name):
        '''DOCSTRING
        '''
        file_tb = open(file_name,'wb')
        pickle.dump(self,file_tb)

        file_tb.close()

        cwd = os.getcwd()

        print('Saved {} in {}'.format(file_name,cwd))

    # TODO: Function needed to run whole process of cutoff creation at once.

class Channel:
    def __init__(self, channel=None):
        self.channel = channel

        global __version__
        self.__version__ = __version__

    def save_object(self,file_name):
        '''DOCSTRING
        '''
        file_obj = open(file_name,'wb')
        pickle.dump(self,file_obj)

        file_obj.close()

        cwd = os.getcwd()

        print('Saved {} in {}'.format(file_name,cwd))

    def print_attributes(self):
        print(self.__dict__)


def plot_churn_workaround(df, title, which_bins='bin_labels'):
    temp_obj = Threshold_Build()

    temp_obj.plot_churn(
        bins=which_bins,
        df = df,
        plot_type = 'lines',
        title = title,
        store_object = True,
        show_results = False
    )

    return temp_obj.bivariate
