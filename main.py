from datetime import datetime, timedelta
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Global variables
INPUT_FOLDER = os.path.join('data', 'input')
OUTPUT_FOLDER = os.path.join('data', 'output')
EXPED_FILENAME = 'expeditions.csv'
BLACK = "#020202"
YELLOW = "#f5d364"

# Import data and drop useless rows
exp_df = pd.read_csv(os.path.join(INPUT_FOLDER, EXPED_FILENAME))
exp_df.query("peakid == 'EVER'", inplace=True)  # Only everest
exp_df['bcdate'] = pd.to_datetime(exp_df['bcdate'], errors='coerce')
exp_df['smtdate'] = pd.to_datetime(exp_df['smtdate'], errors='coerce')
exp_df.dropna(subset=['expid', 'peakid', 'year', 'disputed',
                      'claimed', 'bcdate', 'smtdate', 'bcdate',
                      'termreason', 'highpoint', 'season', 'o2used',
                      'totmembers', 'mdeaths', 'campsites'],
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
            'termreason', 'termnote', 'highpoint', 'totmembers',
            'smtmembers', 'mdeaths', 'o2used', 'campsites',
            'accidents']
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

# Fill in missing elevations using the camp map (override basecamp)
exp_df['camp_tuples'] = exp_df['camp_tuples']\
    .apply(lambda x: [(d, e, c) if e is not None else (d, camp_dict.get(c), c) for (d, e, c) in x])
exp_df['camp_tuples'] = exp_df['camp_tuples']\
    .apply(lambda x: [(d, e, c) if c.lower() != 'bc' else (d, camp_dict.get(c), c) for (d, e, c) in x])

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


# Add year information to records
def flip_date_format_and_add_year(row):
    return [(f'{row["year"]}/{datetime.strptime(date, "%d/%m").strftime("%m/%d")}', elevation)
            for (date, elevation) in row['camp_tuples']]


exp_df['camp_tuples'] = exp_df.apply(flip_date_format_and_add_year, axis=1)


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


def remove_duplicates_by_date(tuples_list):
    seen_dates = set()
    result_list = []

    for date, elevation in tuples_list:
        if date not in seen_dates:
            result_list.append((date, elevation))
            seen_dates.add(date)

    return result_list


exp_df['camp_tuples'] = exp_df['camp_tuples'].apply(remove_duplicates_by_date)

# Make main dataframe for o2 plot
profile_df = exp_df[['expid', 'camp_tuples']]

# Explode dataframe
profile_df = profile_df.explode('camp_tuples')
profile_df[['date', 'elevation']] = profile_df['camp_tuples'].apply(lambda x: pd.Series([x[0], x[1]]))
profile_df = profile_df.drop('camp_tuples', axis=1)
profile_df.sort_values(by=['expid', 'date'], ascending=True, inplace=True, ignore_index=True)

# Remove rows where elevation is above 8849 or below 5360
profile_df = profile_df.loc[profile_df.elevation.between(5360, 8849, inclusive='both'), :]
profile_df.reset_index(inplace=True, drop=True)


# Remove rows that ruin the "ascending" of elevation (in order top to bottom by expid, first should never be removed)
def mark_rows(group):
    group['is_remove'] = False
    last_accepted_elevation = float('-inf')

    for idx, row in group.iterrows():
        if row['elevation'] < last_accepted_elevation:
            group.at[idx, 'is_remove'] = True
        else:
            last_accepted_elevation = row['elevation']

    return group


profile_df = profile_df.groupby('expid').apply(mark_rows).reset_index(drop=True)
profile_df = profile_df[~profile_df['is_remove']].drop(columns=['is_remove'])

# Add timedeltas
profile_df['date'] = pd.to_datetime(profile_df['date'], errors='coerce')  # Make sure the 'date' column is in datetime format


def add_time_delta(group):
    group['time_delta_days'] = (group['date'] - group['date'].iloc[0]).dt.days
    return group


profile_df = profile_df.groupby('expid').apply(add_time_delta).reset_index(drop=True)

# Keep only expeditions that lasted at least 2 days and at most 80 (to the summit)
keep_df = profile_df.groupby(by='expid')['time_delta_days'].max().reset_index()\
    .query('time_delta_days >= 2 and time_delta_days <= 80')[['expid']]\
    .reset_index(drop=True)
profile_df = keep_df.merge(profile_df, how='left', on='expid')
profile_df.rename(columns={'time_delta_days': 'x', 'elevation': 'y'}, inplace=True)

# Keep only expeditions that start at BC  # TODO: why is this issue cropping up? shouldn't this have been taken care of?
keep_df = profile_df.groupby(by='expid')['y'].min().reset_index()\
    .query('y == 5360')[['expid']]\
    .reset_index(drop=True)
profile_df = keep_df.merge(profile_df, how='left', on='expid')
profile_df.reset_index(inplace=True, drop=True)

# Save
profile_df.to_csv(os.path.join(OUTPUT_FOLDER, 'profile_df.csv'), index=False)

# Interpolate
key_exped = ["EVER53101-1953", "EVER19157-2019", "EVER86201-1986", "EVER75101-1975"]


def interpolate_y(df):
    expid = df.expid.iloc[0]
    x_old = df['x']
    y_old = df['y']

    new_x = np.linspace(min(x_old), max(x_old), 1000)

    interpolator = PchipInterpolator(x_old, y_old)
    new_y = interpolator(new_x)

    result_dict = {'x_old': x_old.values[-1],
                   'y_old': y_old.values[-1],
                   'x_new': new_x,
                   'y_new': new_y,
                   'color': YELLOW if expid in key_exped else BLACK,
                   'linewidth': 2 if expid in key_exped else 1,
                   'alpha': 1 if expid in key_exped else 0.2}

    return result_dict


df_lst = []
for _, group_df in profile_df.groupby('expid'):

    df_lst.append(interpolate_y(group_df))

# Make plot
for interp_dict in df_lst:
    x_new = interp_dict['x_new']
    y_new = interp_dict['y_new']
    plt.plot(x_new, y_new, '-', label='Interpolated Data',
             linewidth=interp_dict['linewidth'],
             color=interp_dict['color'], alpha=interp_dict['alpha'])

for interp_dict in df_lst:
    plt.plot(interp_dict['x_old'], interp_dict['y_old'], 'o',
             label='Original Points',
             markersize=2,
             color=interp_dict['color'],
             alpha=1)

plt.plot(26, 8849, 'x', color='r')
plt.plot(38, 8849, 'x', color='r')
plt.plot(51, 8849, 'x', color='r')

plt.savefig(os.path.join(OUTPUT_FOLDER, 'output_plot.svg'), format='svg')
plt.show()
