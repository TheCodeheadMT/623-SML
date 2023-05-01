import datetime

# Given a date time str, convert it to unixtime
def util_fix_date(datetime_str):
    # 1601-01-01T00:00:00.307030+00:00
    if datetime_str.split("-")[0] == "0000":
        return 0.0
    utc_dt = datetime.strptime(datetime_str.split("+")[0], '%Y-%m-%dT%H:%M:%S.%f')
    timestamp = (utc_dt - datetime(1970, 1, 1)).total_seconds()
    if timestamp < 0:
        return 0.0
    return timestamp

# Given datetime column, apply a mapping to column to convert it to unix time
def clean_datetimes(datetime_col):
    return datetime_col.apply(util_fix_date)