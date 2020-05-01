from datetime import date,datetime
import urllib.request
import json

import pandas as pd

API_URL = "http://www.gdacs.org/export.aspx?profile=ARCHIVE&type=geojson&from={}&to={}"

RUN_NUM = '00'
START_DATE = date(2008, 1, 1)
END_DATE = date(2020, 1, 1)
DATE_DANGE = pd.date_range(START_DATE, END_DATE)
COUNTRY = 'Zimbabwe'


df = pd.DataFrame(columns=['Title', 'Summary', 'Id', 'Copyright', 'PublishedDate', 'LastUpdate', 'Category', 'Author', 'Content', 'LinkURl', '_georss_feature_number'
                           'gdacs_country', 'gdacs_cap', 'gdacs_alertlevel', 'gdacs_episodeid', 'gdacs_eventid', 'gdacs_eventname', 'gdacs_eventtype', 'gdacs_fromdate', 'gdacs_gtslink',
                           'gdacs_population', 'gdacs_severity', 'gdacs_todate', 'gdacs_version', 'gdacs_vulnerability', 'gdacs_year', 'gdacs_glide', '_x', '_y', '_z'])

for single_date in DATE_DANGE:
    single_date = single_date.strftime("%Y-%m-%d")
    api_call = API_URL.format(single_date, single_date)
    try:
        with urllib.request.urlopen(api_call) as url:
            data = json.loads(url.read().decode())['features']
            data = [feature for feature in data if feature['properties']['countrylist'] == COUNTRY]
            print(single_date,' Events: ',len(data))
            for feature in data:
                properties = feature['properties']
                gdacs_event = dict()
                fromdate=datetime.strptime(properties['fromdate'], '%d/%b/%Y %H:%M:%S')
                todate=datetime.strptime(properties['todate'], '%d/%b/%Y %H:%M:%S')
                gdacs_event['Title'] = properties['htmldescription']
                gdacs_event['Id'] = '{}{}'.format(properties['eventtype'], properties['eventid'])
                gdacs_event['LinkURl'] = properties['link']
                gdacs_event['gdacs_country'] = properties['countrylist']
                gdacs_event['gdacs_alert_level'] = properties['alertlevel']
                gdacs_event['gdacs_episodeid'] = properties['episodeid']
                gdacs_event['gdacs_eventid'] = properties['eventid']
                gdacs_event['gdacs_eventname'] = properties['eventname']
                gdacs_event['gdacs_eventtype'] = properties['eventtype']
                gdacs_event['gdacs_fromdate'] = fromdate.strftime("%Y-%m-%d")
                gdacs_event['gdacs_todate'] = todate.strftime("%Y-%m-%d")
                gdacs_event['_x'] = feature['geometry']['coordinates'][0]
                gdacs_event['_y'] = feature['geometry']['coordinates'][1]
                # remove old entries and get uupdates
                df = df[df.gdacs_eventid != gdacs_event['gdacs_eventid']]
                df = df.append(gdacs_event, ignore_index=True)
                df.to_csv(f'output/Historical_GDACS_data_{RUN_NUM}.csv', index=False)
    except:
        print('Encountered error, going to next date')
        continue
df.to_excel('output/Historical_GDACS_data.xlsx')
