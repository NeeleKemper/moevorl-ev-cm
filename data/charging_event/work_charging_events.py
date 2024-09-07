import os
import requests
import pandas as pd
from charging_events_utils import process_dataframe, gmm_fitting

# API token and base URL for fetching data from the EV charging platform
token = 'cPQtlwTQK30x8uOdC9DSwx9O_IQe6yzQ6rA25mUJByU'  # Add your own API token
base_url = 'https://ev.caltech.edu/api/v1/sessions/'  # Replace this with the actual base URL

# API endpoints for the different locations
# 'caltech': Research university in Pasadena, CA, with data from 54 EVSEs mainly used by faculty, staff and students.
# 'jpl': National research lab in La Canada, CA, with 50 EVSEs used by employees, indicating a normal workplace schedule.
# 'office001': Office building in the Silicon Valley area with 8 EVSEs used by employees.

# Headers for API requests
headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
}


def fetch_data() -> None:
    """
    Fetches the EV charging data from the specified API endpoints and stores it locally.

    :return: None
    """
    df = pd.DataFrame(columns=['arrival_time', 'departure_time', 'energy'])

    # Convert string to datetime
    start_date = pd.to_datetime('Wed, 05 Sep 2018 00:00:00 GMT')

    # Generate timeseries
    dates = pd.date_range(start=start_date, periods=5 * 365, freq='D')

    # Convert back to the required string format
    dates_str = dates.strftime('%a, %d %b %Y %H:%M:%S GMT')

    for endpoint in ['jpl', 'office001']: # 'caltech',
        for date in dates_str:
            print(f'{endpoint}: {date}')
            # make the GET request
            response = requests.get(
                base_url + endpoint + f'?where=connectionTime>="{date}"',
                headers=headers)

            # if the request was successful, the status code will be 200
            if response.status_code == 200:
                data = response.json()
                for item in data['_items']:
                    print(item)
                    arrival_time = pd.to_datetime(item['connectionTime'], format='%a, %d %b %Y %H:%M:%S %Z')
                    departure_time = pd.to_datetime(item['disconnectTime'], format='%a, %d %b %Y %H:%M:%S %Z')
                    energy = item['kWhDelivered']
                    df.loc[len(df)] = [arrival_time, departure_time, energy]
            else:
                print(f'Request failed with status code {response.status_code}')

    path = 'work'
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(f'{path}/acn_data.csv')


def get_work_data() -> pd.DataFrame:
    """
    Load and process the workplace charging data.

    :return: A DataFrame containing the processed workplace charging data.
    """
    df = pd.read_csv('work/acn_data.csv')
    df['arrival_time'] = pd.to_datetime(df['arrival_time']).dt.tz_convert('America/Los_Angeles')
    df['departure_time'] = pd.to_datetime(df['departure_time']).dt.tz_convert('America/Los_Angeles')
    return process_dataframe(df)


def main():
    fetch_data()
    df = get_work_data()
    gmm_fitting(df, weekday=True, name='work')
    gmm_fitting(df, weekday=False, name='work')


if __name__ == '__main__':
    main()
