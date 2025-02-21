import telstats

ts = telstats.TelStats()
ts.plot_fraction_region(site_ref="Chile", min_diameter=3)
ts.plot_area_time()
ts.plot_show()

