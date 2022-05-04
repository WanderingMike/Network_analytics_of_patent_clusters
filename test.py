dic = {1234: 5, 444: 10, 123: 101}
import pandas as pd
df = pd.DataFrame(list(dic.items()), columns = ['Products','Prices'])
print(df)