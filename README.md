# telstats
A Class to do quick worldwide telescope statistics. Optical telescope data is read directly
from wikipedia tables. Code works with table format of May 13th, 2020. If table's format changes, code would need to be adapted.

Try:

```
import telstats

ts = telstats.TelStats(site_ref="Chile")
ts.plot_fraction_region()
ts.plot_area_time()
```
