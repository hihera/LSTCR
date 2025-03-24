from datetime import datetime

def get_date_info(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    is_holiday = 0  # 根据日期判断是否是假期，这里简化处理
    month = date_obj.month
    weekday = date_obj.weekday()  # 0: Monday, 6: Sunday
    season = get_season(date_obj)
    time_of_day = get_time_of_day(date_obj)
    return is_holiday, month, weekday, season, time_of_day, date_obj

def get_season(date_obj):
    month = date_obj.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def get_time_of_day(date_obj):
    hour = date_obj.hour
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'
