# telstats.py  - Quick stats and plots on telescopes worldwide
#
# Copyright (C) 2020 Patricio Rojo <pato@das.uchile.cl>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import re

seaborn.set()


class TelStats:
    def __init__(self,
                 site_ref="Chile",
                 min_diameter=3):
        self.col_built = ["First light", "Built"]

        self.site_ref = site_ref
        self.min_diameter = min_diameter
        self.real_min_diameter = 0

        self.all_telescopes = pd.DataFrame()
        self.add_wikipedia_tels()
        self.add_future_tels()
        self.add_mm_tels()

    def add_mm_tels(self,
                    clear_list=False,
                    verbose=False,
                    ):

        submm = {"Name": ["ALMA", "SMA", "IRAM30", "SPT",
                          "SMT", "JCMT", "Herschel", "APEX",
                          "Greenland", "NOEMA", "ARO12", ],
                 "Site": ["Chile", "USA", "Spain", "Antarctica",
                          "USA", "USA", "Space", "Chile",
                          "Greenland", "France", "USA"],
                 self.col_built[0]: [2014, 2003, 1984, 2007,
                                     1993, 1987, 2009, 2005,
                                     2017, 2020, 2013],
                 "area": [54 * np.pi * 6 ** 2 + 12 * np.pi * 3.5 ** 2, 8 * np.pi * 3 ** 2, np.pi * 15 ** 2,
                          np.pi * 5 ** 2,
                          np.pi * 5 ** 2, np.pi * 7.5 ** 2, np.pi * 1.75 ** 2, np.pi * 7.5 ** 2,
                          np.pi * 6 ** 2, np.pi * 10 * 7.5 ** 2, np.pi * 6 ** 2
                          ],
                 }

        tels_mm = pd.DataFrame(submm)
        tels_mm['range'] = "mm"

        if verbose:
            print("Millimeter telescopes considered:")
            print(tels_mm)
        if clear_list:
            self.all_telescopes = pd.DataFrame()
        self.all_telescopes = pd.concat([self.all_telescopes, tels_mm], ignore_index=True)

    def add_future_tels(self,
                        clear_list=False,
                        verbose=False,
                        ):
        future = {"Name": ["E-ELT", "GMT", "TMT"],
                  self.col_built[0]: [2025, 2029, 2033],
                  "area": [np.pi * (39.3 / 2) ** 2, 6 * np.pi * (8.4 / 2) ** 2, np.pi * 15 ** 2],
                  "Site": ["Chile", "Chile", "Spain"],
                  "range": ["optical"] * 3,
                  }
        tels_fut = pd.DataFrame(future)

        if verbose:
            print("Future telescopes considered:")
            print(tels_fut)
        if clear_list:
            self.all_telescopes = pd.DataFrame()
        self.all_telescopes = pd.concat([self.all_telescopes, tels_fut], ignore_index=True)

    def add_wikipedia_tels(self,
                           largest_url='https://en.wikipedia.org/wiki/List_of_largest_optical_reflecting_telescopes',
                           large_url='https://en.wikipedia.org/wiki/List_of_large_optical_reflecting_telescopes',
                           clear_list=False,
                           verbose=False,
                           ):
        largest = pd.read_html(largest_url)
        large = pd.read_html(large_url)

        # right now, the "largest" page has only one table, while the "large" has several: greather than
        # 2m (exhaustive), greater than 1m (selection), smaller than 1m (selection, not considered)
        g3m = largest[1][:-1]
        g2m = large[0]
        g1m = large[1]

        all_telescopes = pd.concat([g3m, g2m, g1m], ignore_index=True)
        if self.min_diameter < 2:
            print("WARNING: Wikipedia list of telescopes smaller than 2m diameter is not complete")

        # numerizing and computing area
        tels = all_telescopes.replace(
            regex={r'\d+[–-](\d+)': r'\1', r'\d/(\d)': r'\1', r'.ref..+./ref.': '', r'TBA': np.nan,
                   r's$': '', r'.+(\d\d\d\d)$': r'\1', r'\[\d+\]': ''})
        diameter = tels['Effective aperture'].str.extract(r'(?P<inches>[0-9.]+).in', expand=False)
        diameter = diameter.fillna(tels['Aperture'].str.extract(r'(?P<inches>[0-9.]+).in', expand=False))
        if 'Aper. in' in tels.columns:
            diameter = diameter.fillna(tels['Aper. in'].str.extract(r'(?P<inches>[0-9.]+)″', expand=False))
        area = 3.14159 * (diameter.astype(float) * 0.0254 / 2) ** 2
        tels = tels.join(diameter).join(area.rename('area'))
        tels['range'] = 'optical'

        def first_int(toconvert):
            ret = int(re.search(r'\d+', toconvert).group())
            return ret

        # extracting relevant columns
        to_fill = tels[self.col_built[0]].isna()

        for col in self.col_built[1:]:
            available = ~tels[col].isna()
            tels.loc[to_fill & available, self.col_built[0]] = tels[col][to_fill & available]

        tels = tels[['Name', self.col_built[0], 'area', 'range', 'Site']]
        dropping = tels.loc[(tels.isna()).sum(1) > 0]
        tels = tels.dropna()
        tels[self.col_built[0]] = tels[self.col_built[0]].apply(first_int)
        tels = tels.sort_values(by=self.col_built[0])
        if len(dropping):
            nl = '\n'
            texts = dropping[['Name', 'First light', 'area']].agg(lambda x: '{} [{}, {}m]'.format(*x), axis=1)
            print(f"WARNING: {len(dropping)} row(s) were discarded from wikipedia (name [built, area]):\n - "
                  f"{f'{nl} - '.join(texts)}")

        if verbose:
            print("Wikipedia tables parsed:")
            print(tels)
        if clear_list:
            self.all_telescopes = pd.DataFrame()
        self.all_telescopes = pd.concat([self.all_telescopes, tels], ignore_index=True)

    def filter_diameter(self, data=None, min_diameter=None):
        if min_diameter is None:
            min_diameter = self.min_diameter
        if data is None:
            data = self.all_telescopes
        df = data[data['area'] > np.pi * (min_diameter / 2) ** 2]

        self.real_min_diameter = np.min(np.sqrt(df['area'] / 3.14159) * 2)
        self.min_diameter = min_diameter

        return df

    def filter_region(self, data=None, site_ref=None):
        if site_ref is None:
            site_ref = self.site_ref
        if data is None:
            data = self.all_telescopes.copy()
        df = data.join(data['Site'].str.lower().str.extract(f'(?P<in_region>{site_ref.lower()})')
                       == site_ref.lower())

        self.site_ref = site_ref

        return df

    @staticmethod
    def select_range_region(tels):
        tels_opt = tels[tels['range'] == 'optical']
        tels_mm = tels[tels['range'] == 'mm']
        return tels_opt, tels_mm, tels_opt[tels_opt['in_region']], tels_mm[tels_mm['in_region']]

    def plot_area_time(self,
                       xlabel="year",
                       ylabel='Area ($m^2$)',
                       title="Only for telescopes with diameters larger than {real_min_diameter:.1f}m",
                       axes=None,
                       min_diameter=2,
                       site_ref=None
                       ):
        tels = self.filter_diameter(min_diameter=min_diameter)
        tels = self.filter_region(data=tels, site_ref=site_ref)

        tels_opt, tels_mm, tels_opt_region, tels_mm_region = self.select_range_region(tels)

        if axes is None:
            f, axes = plt.subplots(figsize=(10, 8))
        axes.semilogy(tels_mm[self.col_built[0]], tels_mm['area'], 'kv', label="mm")
        axes.semilogy(tels_opt[self.col_built[0]], tels_opt['area'], 'k^', label="opt")
        axes.semilogy(tels_mm_region[self.col_built[0]], tels_mm_region['area'], 'rv', label=f"mm@{self.site_ref}")
        axes.semilogy(tels_opt_region[self.col_built[0]], tels_opt_region['area'], 'r^', label=f"opt@{self.site_ref}")
        axes.legend()

        self._set_custom_titles(axes, xlabel, ylabel, title)

        return axes

    def _set_custom_titles(self, ax, xlabel, ylabel, title):
        if xlabel is not None:
            ax.set_xlabel(xlabel.format(**{name: getattr(self, name)
                                           for name in re.findall('{(.+?)(?::.+)?}', xlabel)}))
        if ylabel is not None:
            ax.set_ylabel(ylabel.format(**{name: getattr(self, name)
                                           for name in re.findall('{(.+?)(?::.+)?}', ylabel)}))
        if title is not None:
            ax.set_title(title.format(**{name: getattr(self, name)
                                         for name in re.findall('{(.+?)(?::.+)?}', title)}))

    def get_cumulatives(self, bins,
                        min_diameter=None, site_ref=None):
        """
        Get cumulative areas grouped by years binned

        Parameters
        ----------
        min_diameter: float
        site_ref : str
           Reference site. Default none is to use the one specified at object creation
        bins: list
           years in which to bin

        Returns
        -------
        tuple
           Returns the cumulative value of areas in
        [opt_total, mm_total, opt_in_region, mm_in_region]

        """
        tels = self.filter_diameter(min_diameter=min_diameter)
        tels = self.filter_region(data=tels, site_ref=site_ref)
        tels_split = self.select_range_region(tels)

        digs = [np.digitize(tels_sub[self.col_built[0]], bins, right=True) for tels_sub in tels_split]
        cums = [np.array([tels_sub['area'].loc[digs_sub <= i].sum() for i in range(len(list(bins)))])
                for tels_sub, digs_sub in zip(tels_split, digs)]

        return cums

    @staticmethod
    def _fill_defaults(params, defaults):
        for prop in defaults.keys():
            for param, default in zip(params, defaults[prop]):
                for idx, d in enumerate(default):
                    if param[idx] is None:
                        param[idx] = {}
                    if prop not in param[idx]:
                        param[idx][prop] = d

    def plot_fraction_region(self,
                             dbin=5,
                             from_year=1960,
                             until_year=2035,
                             xlabel="year",
                             ylabel="Percentage of area in ",
                             ylabel_right="Total telescope area",
                             title="Only for telescopes with diameters larger than {real_min_diameter:.1f}m",
                             axes=None,
                             opt_style="bar", mm_style="line", both_style="line",
                             opt_params=None, mm_params=None,   both_params=None,
                             opt_total_style="line", mm_total_style="none", total_style="none",
                             opt_total_params=None, mm_total_params=None,   total_params=None,
                             site_ref=None, min_diameter=None,
                             ):
        """
        Plot percentage of telescope area in region

        Parameters
        ----------
        min_diameter: float
           Minimum diameter to consider
        site_ref: str
           region to filter
        ylabel_right: str
           label for right axis
        dbin: float
           Bin every this many years. Default is 5
        from_year: int
           Default is 1960
        until_year: int
           Default is 2030
        xlabel: str
        ylabel: str
        title: str
        axes: matplotlib.axes
        mm_style: ["none"|"line"|"bar"]
        opt_style: ["none"|"line"|"bar"]
        both_style: ["none"|"line"|"bar"]
        total_style: ["none"|"line"|"bar"]
        mm_total_style: ["none"|"line"|"bar"]
        opt_total_style: ["none"|"line"|"bar"]
        mm_params: dict
           Plotting parameters for mm fraction data
        opt_params: dict
           Plotting parameters for optical fraction data
        both_params: dict
           Plotting parameters for optical+mm fraction data
        total_params: dict
           Plotting parameters for optical+mm total data
        mm_total_params: dict
           Plotting parameters for mm total data
        opt_total_params: dict
           Plotting parameters for optical total data

        Returns
        -------
           matplotlib.axes.__class__
        instance used

        """
        if site_ref is None:
            site_ref = self.site_ref
        ylabel += str(site_ref)

        bins = np.arange(from_year, until_year, dbin)
        cums = self.get_cumulatives(bins,
                                    site_ref=site_ref,
                                    min_diameter=min_diameter)

        if axes is None:
            f, axes = plt.subplots(figsize=(10, 8))
        labels = [["% opt", "% mm", "% opt+mm"],
                  ["total opt", "total mm", "total opt+mm"]]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".+true_divide")
            fracs = [[cums[2] / cums[0], cums[3] / cums[1], (cums[2] + cums[3]) / (cums[0] + cums[1])],
                     [cums[0], cums[1], cums[0]+cums[1]]]
        styles = [[opt_style, mm_style, both_style],
                  [opt_total_style, mm_total_style, total_style]]
        params = [[opt_params, mm_params, both_params],
                  [opt_total_params, mm_total_params, total_params]]
        defaults = {'color': [['blue', 'red', 'black'], ['green', 'yellow', 'pink']],
                    'ls': [['-', '-', '-'], ['--', '--', '--']]
                    }
        self._fill_defaults(params, defaults)

        all_axes = [axes]
        if not all(st == "none" for st in styles[1]):
            all_axes.append(axes.twinx())

        for idx, ax in enumerate(all_axes):
            for style, param, frac, label in zip(styles[idx], params[idx], fracs[idx], labels[idx]):
                if style == 'bar':
                    ax.bar(bins - dbin / 2, 100 * frac, label=label, width=dbin * 0.9, align="center", **param)
                elif style == 'line':
                    ax.plot(bins - dbin / 2, 100 * frac, label=label, **param)
                elif style == 'none':
                    pass
                else:
                    print(f"WARNING: plotting style '{style}' unrecognized for '{label}' ")
            ax.legend(loc=2-idx)
        self._set_custom_titles(axes, xlabel, ylabel, title)
        if len(all_axes) == 2:
            self._set_custom_titles(all_axes[1], None, ylabel_right, None)

        return axes

    @staticmethod
    def plot_show():
        plt.show()

    @staticmethod
    def plot_ion():
        plt.ion()
