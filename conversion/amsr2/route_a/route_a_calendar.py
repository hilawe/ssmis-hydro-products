"""The 72 fitting days of the AMSR2 conversion maps: the 5th, 15th and 25th of each month from April 2022 to March
2024. The fit refuses any other list of days, and the committed maps record these days as their own."""
import datetime

FIT_DAYS = [datetime.date(y, m, d).isoformat() for y, m in
            [(2022, mm) for mm in range(4, 13)] + [(2023, mm) for mm in range(1, 13)] + [(2024, mm) for mm in (1, 2, 3)]
            for d in (5, 15, 25)]
