from datetime import datetime
import pandas as pd

start = datetime(2018, 1, 1)
end = datetime(2020, 1, 1)
time_series = {k: None for k in ["a", "b", "c"]}

for cpc_group in time_series.keys():
    indicators = {"patent_count": None,
                  "emergingness": None,
                  "citations": None}
    time_series[cpc_group] = {k: indicators for k in range(start.year, end.year + 1)}

time_series["b"][2018]["patent_count"] = 10000

print(time_series)

def dataframe():
    var1 = 2
    var2 = 10

    var1, var2 = 10, 5
    print(var1, var2)

    patents = {'1':10, '3':50, '2':60}
    df = pd.DataFrame({'d': [1, 2, 2], 'e': [111, 2222, 2222]}, index=['1', '2', '3'])
    print(df)
    print(df[df.index.isin(["2"])])
    df.drop_duplicates(inplace=True)

    print(df)



def test():
    a = 0b1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    b = 0b0000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000

    c = 1
    for i in range(400):
        c <<= 1

    print(c)

    if (a & b) > 0:
        print("yes!")
    else:
        print("No...")
