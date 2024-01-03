import pandas as pd
import os
import re
import numpy as np
from datetime import datetime, timedelta
# import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Global variables
INPUT_FILEPATH = os.path.join('data', 'input')
EXPED_FILENAME = 'expeditions.csv'

# Import data and drop useless rows
exp_df = pd.read_csv(os.path.join(INPUT_FILEPATH, EXPED_FILENAME))
exp_df.query("peakid == 'EVER'", inplace=True)  # Only everest
exp_df['termdate'] = pd.to_datetime(exp_df['termdate'], errors='coerce')
exp_df['bcdate'] = pd.to_datetime(exp_df['bcdate'], errors='coerce')
exp_df['smtdate'] = pd.to_datetime(exp_df['smtdate'], errors='coerce')
exp_df.dropna(subset=['expid', 'peakid', 'year', 'disputed',
                      'claimed', 'bcdate', 'smtdate', 'termdate',
                      'bcdate', 'termreason', 'highpoint', 'season',
                      'o2used', 'totmembers', 'mdeaths', 'campsites'],
              inplace=True, ignore_index=True)  # Make sure these values are present

# Add "is_considered" and filter to relevant records
exp_df['is_considered'] = (~exp_df['termreason'].isin([0, 2, 11, 12, 13, 14])
                           & ~exp_df['disputed']
                           & ~exp_df['claimed'])
exp_df.query('is_considered == True', inplace=True)
exp_df.reset_index(inplace=True, drop=True)

# Add "has_summit" and update peakid to account for century
exp_df['has_summit'] = exp_df[['success1', 'success2', 'success3', 'success2']].sum(axis=1).gt(0)
exp_df['expid'] = exp_df['expid'] + '-' + exp_df['year'].astype(str)

# Kee only relevant columns
rel_cols = ['expid', 'peakid', 'year', 'season',
            'has_summit', 'bcdate', 'smtdate', 'totdays',
            'termdate', 'termreason', 'termnote', 'highpoint',
            'totmembers', 'smtmembers', 'mdeaths', 'o2used',
            'campsites', 'accidents']
exp_df = exp_df[rel_cols]

# Remove campsites that contain "see route details" (or something of the sort)
exp_df = exp_df.loc[~exp_df['campsites'].str.lower().str.contains('see route'), :]
exp_df.reset_index(inplace=True, drop=True)

# Split campsites by commas that are not in parentheses (remove extra cases after colons)
exp_df['campsites'] = exp_df['campsites'].str.split(';').apply(lambda x: x[0])
exp_df['campsites'] = exp_df['campsites'].str.replace(' ', '')
exp_df['split_camp'] = exp_df['campsites'].str.split(r',\s*(?![^()]*\))')

# Define the regular expression patterns
pattern1 = r'\b\d{2}/\d{2}\b'
pattern2 = r'\b\d{4}m\b'
pattern3 = r'([^\(]*)\('


# Define a function to create tuples for each element in a list of strings
def create_tuples(element_list):
    camp_tuples = []

    for element in element_list:
        match1 = re.search(pattern1, element)
        match2 = re.search(pattern2, element)
        match3 = re.search(pattern3, element)

        if match1 and (match2 or match3):
            result1 = match1.group().replace(" ", "")
            result2 = float(match2.group().replace(" ", "").replace('m', '')) if match2 else None
            result3 = match3.group(1).replace(" ", "") if match3 else None

            camp_tuples.append((result1, result2, result3))

    return camp_tuples


# Create camp tuples (some information may be missing)
exp_df['camp_tuples'] = exp_df['split_camp'].apply(create_tuples)


# Convert dates to datetime and remove problematic cases (can't be converted)
def conv_dt(lst):
    date_tuples = []
    for date, elevation, camp in lst:
        try:
            date_obj = datetime.strptime(date, '%d/%m').strftime('%d/%m')
        except ValueError:
            date_obj = None
        date_tuples.append((date_obj, elevation, camp))
    filtered_tuples = [(date, elevation, camp) for date, elevation, camp in date_tuples if date is not None]
    return filtered_tuples


exp_df['camp_tuples'] = exp_df['camp_tuples'].apply(conv_dt)

# Create map of campsite to elevation
camp_df = pd.DataFrame([(date, elevation, campsite_name) for lst in exp_df['camp_tuples']
                        for (date, elevation, campsite_name) in lst],
                       columns=['date', 'elevation', 'campsite'])
camp_df = camp_df.dropna(subset=['elevation', 'campsite'], how='any')
camp_df = camp_df.groupby('campsite')['elevation'].max().reset_index()
camp_df.query('campsite != ""', inplace=True)
camp_dict = camp_df.set_index('campsite')['elevation'].to_dict()

camp_dict['BC'] = 5360.0
camp_dict['Bc'] = camp_dict['BC']
camp_dict['Sherpas-BC'] = camp_dict['BC.Sherpas']
camp_dict['BC.Sherpa'] = camp_dict['BC.Sherpas']
camp_dict['Smt'] = 8849.0

# Remove records that don't start at BC
exp_df = exp_df.loc[exp_df['campsites'].str.lower().str.startswith('bc'), :]
exp_df.reset_index(inplace=True, drop=True)

# Fill in missing elevations using the camp map
exp_df['camp_tuples'] = exp_df['camp_tuples']\
    .apply(lambda x: [(d, e, c) if e is not None else (d, camp_dict.get(c), c) for (d, e, c) in x])

# Remove tuples where elevation is still None
exp_df['camp_tuples'] = exp_df['camp_tuples'].apply(lambda x: [ele for ele in x if ele[1] is not None])

# Remove camp names from tuples (no longer needed)
exp_df['camp_tuples'] = exp_df['camp_tuples'].apply(lambda x: [(ele[0], ele[1]) for ele in x])

# Add highpoint tuple
exp_df['hp_tuple'] = pd.Series(zip(exp_df['smtdate'].dt.strftime('%d/%m'), exp_df['highpoint']))
exp_df['camp_tuples'] = [
    tuples_list + [additional_tuple]
    for tuples_list, additional_tuple in zip(exp_df['camp_tuples'], exp_df['hp_tuple'])
]

# Add BC tuple
exp_df['bc_tuple'] = pd.Series(zip(exp_df['bcdate'].dt.strftime('%d/%m'),
                                   [camp_dict['BC']] * len(exp_df['bcdate'])))
exp_df['camp_tuples'] = [
    [additional_tuple] + tuples_list
    for tuples_list, additional_tuple in zip(exp_df['camp_tuples'], exp_df['bc_tuple'])
]


# Custom function to filter the tuples based on unique elevations
def filter_unique_elevations(tuples_list):
    unique_elevations = set()
    result_tuples = []

    for tup in tuples_list:
        elevation = tup[1]
        if elevation not in unique_elevations:
            result_tuples.append(tup)
            unique_elevations.add(elevation)

    return result_tuples


# Apply the function to each element in the series
exp_df['camp_tuples'] = exp_df['camp_tuples'].apply(filter_unique_elevations)


# Add year information to records
def flip_date_format_and_add_year(row):
    return [(f'{row["year"]}/{datetime.strptime(date, "%d/%m").strftime("%m/%d")}', elevation)
            for (date, elevation) in row['camp_tuples']]


exp_df['camp_tuples'] = exp_df.apply(flip_date_format_and_add_year, axis=1)


def remove_duplicates_by_date(tuples_list):
    seen_dates = set()
    result_list = []

    for date, elevation in tuples_list:
        if date not in seen_dates:
            result_list.append((date, elevation))
            seen_dates.add(date)

    return result_list


exp_df['camp_tuples'] = exp_df['camp_tuples'].apply(remove_duplicates_by_date)


def adjust_year_by_previous_date(tuples_list):
    result_list = []

    for i, (date, elevation) in enumerate(tuples_list):
        if i > 0:
            prev_date = tuples_list[i - 1][0]
            current_datetime = datetime.strptime(date, '%Y/%m/%d')
            prev_datetime = datetime.strptime(prev_date, '%Y/%m/%d')

            if current_datetime < prev_datetime:
                current_datetime += timedelta(days=365)  # Adding one year

            date = current_datetime.strftime('%Y/%m/%d')

        result_list.append((date, elevation))

    return result_list


exp_df['camp_tuples'] = exp_df['camp_tuples'].apply(adjust_year_by_previous_date)

# Make main dataframe for o2 plot
o2_df = exp_df[['expid', 'camp_tuples']]

# Explode dataframe
o2_df = o2_df.explode('camp_tuples')
o2_df[['date', 'elevation']] = o2_df['camp_tuples'].apply(lambda x: pd.Series([x[0], x[1]]))
o2_df = o2_df.drop('camp_tuples', axis=1)
o2_df.sort_values(by=['expid', 'date'], ascending=True, inplace=True, ignore_index=True)

# Remove rows where elevation is above 8849 or below 5360
o2_df = o2_df.loc[o2_df.elevation.between(5360, 8849), :]
o2_df.reset_index(inplace=True, drop=True)


# Remove rows that ruin the "ascending" of elevation (in order top to bottom by expid)
def mark_rows(group):
    group['is_remove'] = False
    last_accepted_elevation = float('-inf')

    for idx, row in group.iterrows():
        if row['elevation'] < last_accepted_elevation:
            group.at[idx, 'is_remove'] = True
        else:
            last_accepted_elevation = row['elevation']

    return group


o2_df = o2_df.groupby('expid').apply(mark_rows).reset_index(drop=True)
o2_df = o2_df[~o2_df['is_remove']].drop(columns=['is_remove'])

print(o2_df.head())
