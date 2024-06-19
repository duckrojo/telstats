import re

import pandas as pd
from matplotlib import pyplot as plt

# 2019 data from https://arxiv.org/pdf/1908.02584
data = """
1 USA 59,495 326.6 2,824 31 China 8,583 1,379.3 663
2 Japan 38,550 126.5 728 32 Taiwan 24,227 23.5 75
3 France 39,673 67.1 856 33 Argentina 14,061 44.3 148
4 Germany 44,184 80.6 654 34 Australia 56,135 23.2 328
5 India 1,852 1,281.9 281 35 Austria 46,436 8.8 64
6 Indonesia 3,859 260.6 17 36 Belgium 43,243 11.5 145
7 Iran 5,252 82.0 40 37 Brazil 10,020 207.4 204
8 Israel 39,974 8.3 97 38 Canada 44,773 35.6 307
9 Italy 31,619 62.1 670 39 Chile 14,315 17.8 115
10 Kazakhstan 8,585 18.6 10 40 Czech Rep. 19,818 10.7 125
11 Korea, DPR 1,360 25.2 18 41 Denmark 56,335 5.6 90
12 Korea Rep. 29,730 51.2 158 42 Egypt 3,685 97.0 67
13 Mexico 9,249 124.6 147 43 Finland 45,693 5.5 80
14 Mongolia 3,553 3.1 6 44 Norway 73,615 5.3 41
15 Netherlands 48,272 17.1 228 45 Romania 10,372 21.5 33
16 New Zealand 41,629 4.5 35 46 Serbia 5,600 7.1 51
17 Poland 13,429 38.5 162 47 Slovak Rep. 17,491 5.4 46
18 Portugal 20,575 10.8 69 48 Venezuela 6,850 31.3 22
19 Russian Fed. 10,248 142.3 436 49 Uruguay 17,252 3.4 5
20 Greece 18,945 10.8 121 50 Algeria 4,225 41.0 2
21 Hungary 13,460 9.9 72 51 Armenia 3,690 3.0 28
22 South Africa 6,089 54.8 122 52 Azerbaijan 4,098 10.0 10
23 Spain 28,212 49.0 378 53 Bulgaria 7,924 7.1 67
24 Sweden 53,248 10.0 145 54 Colombia 6,238 47.7 27
25 Switzerland 80,837 8.2 138 55 Estonia 19,618 1.3 33
26 Thailand 6,336 68.4 33 56 Ireland 68,604 5.0 50
27 Turkey 10,434 80.8 80 57 Malaysia 9,660 31.4 10
28 Ukraine 2,459 44.0 152 58 Nigeria 2,092 190.6 10
29 UK 38,847 64.8 724 59 Philippines 3,022 104.3 5
30 Vietnam 2,306 96.2 13 60 Singapore 53,880 5.9 2
"""

pattern = r"\d+ ([a-zA-Z .,]+) ([0-9,]+) ([0-9.,]+) ([0-9,]+)"
repat = re.compile(pattern + " " + pattern)

name = []
gdp = []
population = []
astronomers = []


def float_from_comma(string):
    ff = string.split(',')
    if len(ff) > 1:
        return int(ff[0])*1000 + float(ff[1])
    else:
        return float(ff[0])


for line in data.split("\n"):
    if len(line) < 5:
        continue
    match = re.match(repat, line)
    name.append(match[1])
    gdp.append(float_from_comma(match[2]))
    population.append(float_from_comma(match[3]))
    astronomers.append(float_from_comma(match[4]))

    name.append(match[5])
    gdp.append(float_from_comma(match[6]))
    population.append(float_from_comma(match[7]))
    astronomers.append(float_from_comma(match[8]))

df = pd.DataFrame({'name': name, 'gdp': gdp, 'population': population, 'astronomers': astronomers})
df = df.set_index('name')

cono_sur = df.loc[['Chile', 'Argentina', 'Uruguay']]
latinoamerica = df.loc[['Colombia', "Venezuela", ]]  # "Peru", "Equator", 'Bolivia']]

plt.plot(df['population'], df['astronomers'], 'bo')
plt.plot(cono_sur['population'], cono_sur['astronomers'], 'r^')
plt.plot(latinoamerica['population'], latinoamerica['astronomers'], 'gv')

plt.ylim([0, 150])
plt.xlim([0, 70])

plt.xlabel('population')
plt.ylabel('astronomers')

plt.show()

plt.clf()
plt.plot(df['population'], df['astronomers'] / df['population'], 'bo')
plt.plot(cono_sur['population'], cono_sur['astronomers'] / cono_sur['population'], 'r^')
plt.plot(latinoamerica['population'], latinoamerica['astronomers'] / latinoamerica['population'], 'gv')
plt.show()
