package python

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterPythonModules([]problems.CourseModule{
		{
			ID:          2219,
			Title:       "Python Data Science and Scientific Computing",
			Description: "Master NumPy, Pandas, Matplotlib, SciPy, data manipulation, statistical analysis, and visualization techniques.",
			Order:       19,
			Lessons: []problems.Lesson{
				{
					Title: "NumPy Pandas Matplotlib SciPy and Data Analysis",
					Content: `Python's data science ecosystem is built on NumPy, Pandas, Matplotlib, and SciPy, forming the foundation for scientific computing and data analysis.

**NumPy — Numerical Python:**

ndarray (N-dimensional array):
  Core data structure
  Homogeneous (all elements same type)
  Contiguous memory layout (cache-friendly)
  Vectorized operations (no Python loops)
  
Creation:
  np.array([1, 2, 3]) — from list
  np.zeros((3, 4)) — zero-filled
  np.ones((2, 3)) — one-filled
  np.empty((3, 3)) — uninitialized
  np.arange(0, 10, 2) — evenly spaced (like range)
  np.linspace(0, 1, 100) — evenly spaced (N points)
  np.eye(4) — identity matrix
  np.random.rand(3, 3) — uniform random
  np.random.randn(3, 3) — normal random
  np.random.randint(0, 10, (3, 3)) — integer random
  np.full((3, 3), 7) — filled with value

Attributes:
  arr.shape — dimensions tuple
  arr.ndim — number of dimensions
  arr.size — total elements
  arr.dtype — data type
  arr.itemsize — bytes per element
  arr.nbytes — total bytes

Data Types:
  np.int8, int16, int32, int64
  np.uint8, uint16, uint32, uint64
  np.float16, float32, float64
  np.complex64, complex128
  np.bool_, np.str_, np.object_

Indexing and Slicing:
  arr[0] — first element
  arr[-1] — last element
  arr[1:5] — slice
  arr[::2] — every other element
  arr[:, 0] — first column (2D)
  arr[1, :] — second row (2D)
  
  Fancy indexing:
    arr[[0, 2, 4]] — select specific indices
    arr[arr > 5] — boolean indexing
    
  Views vs Copies:
    Slicing creates a view (shares memory)
    Fancy indexing creates a copy
    arr.copy() — explicit copy

Reshaping:
  arr.reshape(3, 4) — new shape
  arr.ravel() — flatten to 1D (view)
  arr.flatten() — flatten to 1D (copy)
  arr.T — transpose
  arr.transpose(1, 0, 2) — custom axes
  np.expand_dims(arr, axis=0) — add dimension
  np.squeeze(arr) — remove single dimensions

Operations (Vectorized):
  Element-wise: +, -, *, /, **, %
  Comparison: ==, !=, <, >, <=, >=
  Logical: &, |, ~, np.logical_and/or/not
  
  Universal Functions (ufuncs):
    np.sin, np.cos, np.tan
    np.exp, np.log, np.log2, np.log10
    np.sqrt, np.abs
    np.maximum, np.minimum
    np.clip(arr, min, max)
    np.where(cond, x, y) — conditional

Aggregation:
  np.sum, np.prod, np.cumsum, np.cumprod
  np.mean, np.median, np.std, np.var
  np.min, np.max, np.argmin, np.argmax
  np.percentile, np.quantile
  
  Axis parameter:
    arr.sum(axis=0) — sum along columns
    arr.sum(axis=1) — sum along rows

Broadcasting:
  Rules for operations on arrays of different shapes:
  1. Dimensions compared from trailing edge
  2. Dimensions are compatible if equal or one is 1
  3. Missing dimensions padded with 1
  
  Example: (3, 4) + (4,) → (3, 4) + (1, 4) → broadcasts

Linear Algebra:
  np.dot(A, B) or A @ B — matrix multiply
  np.linalg.det(A) — determinant
  np.linalg.inv(A) — inverse
  np.linalg.eig(A) — eigenvalues/vectors
  np.linalg.solve(A, b) — solve Ax = b
  np.linalg.svd(A) — SVD
  np.linalg.norm(v) — vector/matrix norm

**Pandas — Data Analysis:**

Core Data Structures:
  Series: 1D labeled array
  DataFrame: 2D labeled table
  Index: Row/column labels

Creation:
  pd.Series([1, 2, 3], index=['a', 'b', 'c'])
  pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
  pd.read_csv('file.csv')
  pd.read_excel('file.xlsx')
  pd.read_json('file.json')
  pd.read_sql('SELECT * FROM table', conn)
  pd.read_parquet('file.parquet')

Selection:
  df['column'] or df.column — single column (Series)
  df[['col1', 'col2']] — multiple columns (DataFrame)
  df.loc[label] — by label
  df.iloc[index] — by integer position
  df.loc[rows, cols] — label-based slicing
  df.iloc[rows, cols] — integer-based slicing
  df.at[label, col] — single value (fast)
  df.iat[row, col] — single value by position

Filtering:
  df[df['col'] > 5] — boolean mask
  df[df['col'].isin([1, 2, 3])]
  df[df['col'].between(1, 10)]
  df[df['col'].str.contains('pattern')]
  df.query('col > 5 and col2 == "a"')

Missing Data:
  df.isna() / df.isnull() — detect
  df.notna() / df.notnull() — inverse
  df.dropna() — drop rows with NaN
  df.dropna(axis=1) — drop columns with NaN
  df.fillna(value) — fill NaN
  df.fillna(method='ffill') — forward fill
  df.fillna(method='bfill') — backward fill
  df.interpolate() — linear interpolation

Aggregation:
  df.describe() — statistical summary
  df.mean(), df.median(), df.std()
  df.sum(), df.count(), df.nunique()
  df.value_counts() — frequency counts
  df.agg({'col1': 'mean', 'col2': 'sum'})

GroupBy:
  df.groupby('col').mean()
  df.groupby(['col1', 'col2']).agg({...})
  df.groupby('col').transform(func) — same shape
  df.groupby('col').apply(func) — flexible
  df.groupby('col').filter(func) — filter groups

Pivot and Reshape:
  df.pivot_table(values, index, columns, aggfunc)
  df.melt(id_vars, value_vars) — wide to long
  pd.crosstab(df.col1, df.col2) — cross-tabulation
  df.stack() / df.unstack() — reshape multi-index

Merge and Join:
  pd.merge(df1, df2, on='key') — inner join
  pd.merge(df1, df2, on='key', how='left')
  pd.merge(df1, df2, left_on='a', right_on='b')
  pd.concat([df1, df2]) — stack vertically
  pd.concat([df1, df2], axis=1) — side by side
  df1.join(df2, on='key') — index-based join

Time Series:
  pd.to_datetime(column) — parse dates
  df.set_index('date') — datetime index
  df.resample('M').mean() — monthly mean
  df.resample('W').sum() — weekly sum
  df.rolling(7).mean() — 7-period rolling mean
  df.expanding().sum() — expanding window
  df.shift(1) — lag by 1 period
  df.diff() — first difference
  df.pct_change() — percentage change

String Operations:
  df['col'].str.lower()
  df['col'].str.upper()
  df['col'].str.strip()
  df['col'].str.replace('old', 'new')
  df['col'].str.split('delimiter')
  df['col'].str.extract(r'pattern')
  df['col'].str.contains(r'pattern')

Performance:
  Use vectorized operations (avoid apply())
  Use appropriate dtypes (category for low cardinality)
  Use chunking for large files: pd.read_csv(chunksize=1000)
  Use eval() and query() for complex expressions
  Use Parquet format (faster than CSV)

**Matplotlib — Visualization:**

Basic Plots:
  plt.plot(x, y) — line plot
  plt.scatter(x, y) — scatter plot
  plt.bar(x, height) — bar chart
  plt.barh(x, width) — horizontal bar
  plt.hist(data, bins=30) — histogram
  plt.boxplot(data) — box plot
  plt.pie(sizes, labels) — pie chart
  plt.imshow(data) — heatmap/image
  plt.contour(X, Y, Z) — contour plot

Figure and Axes:
  fig, ax = plt.subplots() — single
  fig, axes = plt.subplots(2, 3) — grid
  fig, ax = plt.subplots(figsize=(10, 6))
  
  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_title('Title')
  ax.legend()
  ax.set_xlim(0, 10)
  ax.set_ylim(0, 100)
  ax.grid(True)

Customization:
  Colors: 'r', 'g', 'b', '#FF5733', (0.1, 0.2, 0.5)
  Markers: 'o', 's', '^', 'D', '*', '+'
  Line styles: '-', '--', '-.', ':'
  Line width: linewidth=2
  Transparency: alpha=0.7
  
Saving:
  plt.savefig('plot.png', dpi=300, bbox_inches='tight')
  plt.savefig('plot.pdf')
  plt.savefig('plot.svg')

Seaborn (statistical visualization):
  sns.histplot(data, x='col', hue='category')
  sns.boxplot(data, x='cat', y='value')
  sns.violinplot(data, x='cat', y='value')
  sns.heatmap(correlation_matrix, annot=True)
  sns.pairplot(df, hue='species')
  sns.regplot(data, x='x', y='y')
  sns.countplot(data, x='category')
  sns.kdeplot(data, x='col')

**SciPy — Scientific Computing:**

Optimization:
  scipy.optimize.minimize — general optimization
  scipy.optimize.curve_fit — curve fitting
  scipy.optimize.root — root finding
  scipy.optimize.linprog — linear programming

Statistics:
  scipy.stats.norm — normal distribution
  scipy.stats.ttest_ind — t-test
  scipy.stats.chi2_contingency — chi-square test
  scipy.stats.pearsonr — correlation
  scipy.stats.kstest — KS test
  scipy.stats.mannwhitneyu — Mann-Whitney U

Signal Processing:
  scipy.signal.fft — Fourier transform
  scipy.signal.butter — Butterworth filter
  scipy.signal.filtfilt — zero-phase filtering
  scipy.signal.find_peaks — peak detection

Interpolation:
  scipy.interpolate.interp1d — 1D interpolation
  scipy.interpolate.griddata — N-D interpolation
  scipy.interpolate.UnivariateSpline — spline

Spatial:
  scipy.spatial.distance — distance metrics
  scipy.spatial.KDTree — spatial indexing
  scipy.spatial.ConvexHull — convex hull

Integration:
  scipy.integrate.quad — single integral
  scipy.integrate.dblquad — double integral
  scipy.integrate.odeint — ODE solver`,
					CodeExamples: `# Python data science and scientific computing examples

import math
import statistics
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict

# ============================================================
# NumPy-like Array Operations
# ============================================================

class NDArray:
    """Simplified N-dimensional array implementation."""
    
    def __init__(self, data, shape=None, dtype=float):
        if isinstance(data, list):
            self._data = [dtype(x) for x in self._flatten(data)]
            if shape:
                self._shape = tuple(shape)
            else:
                self._shape = self._infer_shape(data)
        else:
            self._data = [dtype(data)]
            self._shape = (1,)
        self._dtype = dtype
    
    @staticmethod
    def _flatten(data):
        if isinstance(data, (list, tuple)):
            for item in data:
                yield from NDArray._flatten(item)
        else:
            yield data
    
    @staticmethod
    def _infer_shape(data):
        if not isinstance(data, (list, tuple)):
            return ()
        shape = [len(data)]
        if data and isinstance(data[0], (list, tuple)):
            inner = NDArray._infer_shape(data[0])
            shape.extend(inner)
        return tuple(shape)
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def ndim(self):
        return len(self._shape)
    
    @property
    def size(self):
        result = 1
        for s in self._shape:
            result *= s
        return result
    
    @property
    def dtype(self):
        return self._dtype
    
    def __getitem__(self, index):
        if isinstance(index, int):
            if self.ndim == 1:
                return self._data[index]
            # Return a sub-array
            stride = self.size // self._shape[0]
            start = index * stride
            return NDArray(self._data[start:start + stride],
                          shape=self._shape[1:])
        return self._data[index]
    
    def __setitem__(self, index, value):
        self._data[index] = value
    
    def __len__(self):
        return self._shape[0] if self._shape else 0
    
    def __repr__(self):
        return f"NDArray(shape={self._shape}, dtype={self._dtype.__name__})"
    
    def __add__(self, other):
        if isinstance(other, NDArray):
            return NDArray([a + b for a, b in zip(self._data, other._data)],
                          shape=self._shape)
        return NDArray([a + other for a in self._data], shape=self._shape)
    
    def __sub__(self, other):
        if isinstance(other, NDArray):
            return NDArray([a - b for a, b in zip(self._data, other._data)],
                          shape=self._shape)
        return NDArray([a - other for a in self._data], shape=self._shape)
    
    def __mul__(self, other):
        if isinstance(other, NDArray):
            return NDArray([a * b for a, b in zip(self._data, other._data)],
                          shape=self._shape)
        return NDArray([a * other for a in self._data], shape=self._shape)
    
    def __truediv__(self, other):
        if isinstance(other, NDArray):
            return NDArray([a / b for a, b in zip(self._data, other._data)],
                          shape=self._shape)
        return NDArray([a / other for a in self._data], shape=self._shape)
    
    def sum(self, axis=None):
        if axis is None:
            return sum(self._data)
        # Simplified axis sum for 2D
        if self.ndim == 2 and axis == 0:
            rows, cols = self._shape
            result = []
            for j in range(cols):
                s = sum(self._data[i * cols + j] for i in range(rows))
                result.append(s)
            return NDArray(result, shape=(cols,))
        elif self.ndim == 2 and axis == 1:
            rows, cols = self._shape
            result = []
            for i in range(rows):
                s = sum(self._data[i * cols + j] for j in range(cols))
                result.append(s)
            return NDArray(result, shape=(rows,))
        return sum(self._data)
    
    def mean(self, axis=None):
        if axis is None:
            return sum(self._data) / len(self._data)
        s = self.sum(axis)
        n = self._shape[1 - axis] if self.ndim == 2 else len(self._data)
        return NDArray([x / n for x in s._data], shape=s._shape)
    
    def std(self, axis=None):
        m = self.mean(axis)
        if axis is None:
            variance = sum((x - m) ** 2 for x in self._data) / len(self._data)
            return math.sqrt(variance)
        return m  # Simplified
    
    def min(self, axis=None):
        if axis is None:
            return min(self._data)
        return self._data[0]  # Simplified
    
    def max(self, axis=None):
        if axis is None:
            return max(self._data)
        return self._data[-1]  # Simplified
    
    def argmin(self):
        return self._data.index(min(self._data))
    
    def argmax(self):
        return self._data.index(max(self._data))
    
    def reshape(self, *shape):
        new_size = 1
        for s in shape:
            new_size *= s
        if new_size != self.size:
            raise ValueError(f"Cannot reshape {self._shape} to {shape}")
        return NDArray(self._data[:], shape=shape)
    
    def transpose(self):
        if self.ndim != 2:
            raise ValueError("Transpose only for 2D arrays")
        rows, cols = self._shape
        new_data = []
        for j in range(cols):
            for i in range(rows):
                new_data.append(self._data[i * cols + j])
        return NDArray(new_data, shape=(cols, rows))
    
    @property
    def T(self):
        return self.transpose()
    
    def dot(self, other):
        """Matrix multiplication."""
        if self.ndim == 1 and other.ndim == 1:
            return sum(a * b for a, b in zip(self._data, other._data))
        
        if self.ndim == 2 and other.ndim == 2:
            m, k1 = self._shape
            k2, n = other._shape
            if k1 != k2:
                raise ValueError(f"Shape mismatch: {self._shape} @ {other._shape}")
            
            result = []
            for i in range(m):
                for j in range(n):
                    val = sum(
                        self._data[i * k1 + p] * other._data[p * n + j]
                        for p in range(k1)
                    )
                    result.append(val)
            return NDArray(result, shape=(m, n))
        
        raise ValueError("Unsupported dimensions for dot product")
    
    def tolist(self):
        return list(self._data)
    
    @staticmethod
    def zeros(shape):
        size = 1
        for s in shape:
            size *= s
        return NDArray([0.0] * size, shape=shape)
    
    @staticmethod
    def ones(shape):
        size = 1
        for s in shape:
            size *= s
        return NDArray([1.0] * size, shape=shape)
    
    @staticmethod
    def eye(n):
        data = []
        for i in range(n):
            for j in range(n):
                data.append(1.0 if i == j else 0.0)
        return NDArray(data, shape=(n, n))
    
    @staticmethod
    def arange(start, stop=None, step=1):
        if stop is None:
            start, stop = 0, start
        data = []
        val = start
        while val < stop:
            data.append(float(val))
            val += step
        return NDArray(data, shape=(len(data),))
    
    @staticmethod
    def linspace(start, stop, num):
        if num <= 1:
            return NDArray([float(start)], shape=(1,))
        step = (stop - start) / (num - 1)
        data = [start + i * step for i in range(num)]
        return NDArray(data, shape=(num,))


# ============================================================
# DataFrame Implementation
# ============================================================

class Series:
    """Pandas-like Series (1D labeled data)."""
    
    def __init__(self, data, index=None, name=None):
        self._data = list(data)
        self._index = list(index) if index else list(range(len(self._data)))
        self.name = name
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, key):
        if isinstance(key, (list, Series)):
            # Boolean or fancy indexing
            if isinstance(key, Series):
                key = key._data
            return Series(
                [self._data[i] for i, k in enumerate(key) if k],
                [self._index[i] for i, k in enumerate(key) if k],
                self.name
            )
        if isinstance(key, slice):
            return Series(self._data[key], self._index[key], self.name)
        # Label-based
        if key in self._index:
            idx = self._index.index(key)
            return self._data[idx]
        return self._data[key]
    
    def __repr__(self):
        lines = []
        for idx, val in zip(self._index, self._data):
            lines.append(f"{idx}    {val}")
        if self.name:
            lines.append(f"Name: {self.name}")
        return '\n'.join(lines)
    
    def __gt__(self, other):
        return Series([v > other for v in self._data], self._index)
    
    def __lt__(self, other):
        return Series([v < other for v in self._data], self._index)
    
    def __eq__(self, other):
        return Series([v == other for v in self._data], self._index)
    
    def __add__(self, other):
        if isinstance(other, Series):
            return Series([a + b for a, b in zip(self._data, other._data)],
                         self._index, self.name)
        return Series([a + other for a in self._data], self._index, self.name)
    
    def __mul__(self, other):
        if isinstance(other, Series):
            return Series([a * b for a, b in zip(self._data, other._data)],
                         self._index, self.name)
        return Series([a * other for a in self._data], self._index, self.name)
    
    def mean(self):
        numeric = [v for v in self._data if isinstance(v, (int, float))]
        return sum(numeric) / len(numeric) if numeric else 0
    
    def sum(self):
        numeric = [v for v in self._data if isinstance(v, (int, float))]
        return sum(numeric)
    
    def std(self):
        numeric = [v for v in self._data if isinstance(v, (int, float))]
        if len(numeric) < 2:
            return 0
        return statistics.stdev(numeric)
    
    def min(self):
        return min(self._data)
    
    def max(self):
        return max(self._data)
    
    def value_counts(self):
        counts = defaultdict(int)
        for v in self._data:
            counts[v] += 1
        sorted_items = sorted(counts.items(), key=lambda x: -x[1])
        return Series(
            [c for _, c in sorted_items],
            [v for v, _ in sorted_items]
        )
    
    def isna(self):
        return Series([v is None or (isinstance(v, float) and math.isnan(v))
                       for v in self._data], self._index)
    
    def fillna(self, value):
        return Series(
            [value if (v is None or (isinstance(v, float) and math.isnan(v))) else v
             for v in self._data],
            self._index, self.name
        )
    
    def apply(self, func):
        return Series([func(v) for v in self._data], self._index, self.name)
    
    def unique(self):
        seen = set()
        result = []
        for v in self._data:
            if v not in seen:
                seen.add(v)
                result.append(v)
        return result
    
    def nunique(self):
        return len(set(self._data))
    
    def sort_values(self, ascending=True):
        pairs = sorted(zip(self._index, self._data),
                      key=lambda x: x[1], reverse=not ascending)
        return Series(
            [v for _, v in pairs],
            [i for i, _ in pairs],
            self.name
        )
    
    @property
    def values(self):
        return self._data[:]
    
    @property
    def index(self):
        return self._index[:]


class DataFrame:
    """Pandas-like DataFrame (2D labeled data)."""
    
    def __init__(self, data=None, columns=None, index=None):
        if isinstance(data, dict):
            self._columns = list(data.keys())
            max_len = max(len(v) for v in data.values()) if data else 0
            self._data = {}
            for col in self._columns:
                vals = list(data[col])
                if len(vals) < max_len:
                    vals.extend([None] * (max_len - len(vals)))
                self._data[col] = vals
            self._index = list(index) if index else list(range(max_len))
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                self._columns = list(data[0].keys()) if not columns else list(columns)
                self._data = {col: [row.get(col) for row in data] for col in self._columns}
                self._index = list(index) if index else list(range(len(data)))
            else:
                self._columns = list(columns) if columns else [f"col_{i}" for i in range(len(data[0]) if data else 0)]
                self._data = {}
                for i, col in enumerate(self._columns):
                    self._data[col] = [row[i] if i < len(row) else None for row in data]
                self._index = list(index) if index else list(range(len(data)))
        else:
            self._columns = list(columns) if columns else []
            self._data = {col: [] for col in self._columns}
            self._index = list(index) if index else []
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return Series(self._data[key], self._index, name=key)
        if isinstance(key, list):
            return DataFrame({k: self._data[k] for k in key},
                           index=self._index)
        if isinstance(key, Series):
            # Boolean indexing
            mask = key._data
            new_data = {}
            new_index = []
            for col in self._columns:
                new_data[col] = []
            for i, m in enumerate(mask):
                if m:
                    for col in self._columns:
                        new_data[col].append(self._data[col][i])
                    new_index.append(self._index[i])
            return DataFrame(new_data, index=new_index)
        return None
    
    def __setitem__(self, key, value):
        if isinstance(value, Series):
            self._data[key] = value._data[:]
        elif isinstance(value, list):
            self._data[key] = value
        else:
            self._data[key] = [value] * len(self._index)
        if key not in self._columns:
            self._columns.append(key)
    
    def __len__(self):
        return len(self._index)
    
    def __repr__(self):
        lines = ['  '.join([''] + [str(c)[:10] for c in self._columns])]
        for i, idx in enumerate(self._index[:10]):
            row = [str(idx)]
            for col in self._columns:
                val = self._data[col][i] if i < len(self._data[col]) else ''
                row.append(str(val)[:10])
            lines.append('  '.join(row))
        if len(self._index) > 10:
            lines.append(f"... ({len(self._index)} rows)")
        return '\n'.join(lines)
    
    @property
    def shape(self):
        return (len(self._index), len(self._columns))
    
    @property
    def columns(self):
        return self._columns[:]
    
    @property
    def index(self):
        return self._index[:]
    
    def head(self, n=5):
        new_data = {col: self._data[col][:n] for col in self._columns}
        return DataFrame(new_data, index=self._index[:n])
    
    def tail(self, n=5):
        new_data = {col: self._data[col][-n:] for col in self._columns}
        return DataFrame(new_data, index=self._index[-n:])
    
    def describe(self):
        stats = {}
        for col in self._columns:
            values = [v for v in self._data[col] if isinstance(v, (int, float))]
            if values:
                stats[col] = {
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    '25%': sorted(values)[len(values) // 4],
                    '50%': statistics.median(values),
                    '75%': sorted(values)[3 * len(values) // 4],
                    'max': max(values),
                }
        return stats
    
    def groupby(self, column):
        return GroupBy(self, column)
    
    def merge(self, other, on=None, how='inner'):
        if on is None:
            on = list(set(self._columns) & set(other._columns))[0]
        
        result = {col: [] for col in set(self._columns + other._columns)}
        
        if how == 'inner':
            for i, key in enumerate(self._data[on]):
                for j, other_key in enumerate(other._data[on]):
                    if key == other_key:
                        for col in self._columns:
                            result[col].append(self._data[col][i])
                        for col in other._columns:
                            if col != on:
                                result[col].append(other._data[col][j])
        
        return DataFrame(result)
    
    def sort_values(self, by, ascending=True):
        indices = list(range(len(self._index)))
        indices.sort(key=lambda i: self._data[by][i], reverse=not ascending)
        
        new_data = {}
        for col in self._columns:
            new_data[col] = [self._data[col][i] for i in indices]
        new_index = [self._index[i] for i in indices]
        
        return DataFrame(new_data, index=new_index)
    
    def drop(self, columns=None, index=None):
        if columns:
            new_cols = [c for c in self._columns if c not in columns]
            return DataFrame({c: self._data[c] for c in new_cols},
                           index=self._index)
        return self
    
    def rename(self, columns=None):
        if columns:
            new_data = {}
            new_cols = []
            for col in self._columns:
                new_name = columns.get(col, col)
                new_data[new_name] = self._data[col]
                new_cols.append(new_name)
            df = DataFrame(new_data, index=self._index)
            df._columns = new_cols
            return df
        return self
    
    def apply(self, func, axis=0):
        if axis == 0:  # Apply to each column
            result = {}
            for col in self._columns:
                result[col] = func(Series(self._data[col], self._index, col))
            return Series(list(result.values()), list(result.keys()))
        else:  # Apply to each row
            results = []
            for i in range(len(self._index)):
                row = {col: self._data[col][i] for col in self._columns}
                results.append(func(Series(list(row.values()), list(row.keys()))))
            return Series(results, self._index)
    
    def to_dict(self, orient='dict'):
        if orient == 'dict':
            return {col: dict(zip(self._index, self._data[col]))
                    for col in self._columns}
        elif orient == 'records':
            return [
                {col: self._data[col][i] for col in self._columns}
                for i in range(len(self._index))
            ]
        elif orient == 'list':
            return {col: self._data[col][:] for col in self._columns}
        return {}


class GroupBy:
    """GroupBy operations."""
    
    def __init__(self, df: DataFrame, column: str):
        self._df = df
        self._column = column
        self._groups = self._compute_groups()
    
    def _compute_groups(self):
        groups = defaultdict(list)
        for i, val in enumerate(self._df._data[self._column]):
            groups[val].append(i)
        return dict(groups)
    
    def mean(self):
        result = {}
        for col in self._df._columns:
            if col == self._column:
                continue
            values = self._df._data[col]
            if not values or not isinstance(values[0], (int, float)):
                continue
            col_means = {}
            for group_key, indices in self._groups.items():
                group_vals = [values[i] for i in indices]
                col_means[group_key] = sum(group_vals) / len(group_vals)
            result[col] = col_means
        
        group_keys = list(self._groups.keys())
        data = {}
        for col, means in result.items():
            data[col] = [means.get(k, 0) for k in group_keys]
        
        return DataFrame(data, index=group_keys)
    
    def sum(self):
        result = {}
        for col in self._df._columns:
            if col == self._column:
                continue
            values = self._df._data[col]
            if not values or not isinstance(values[0], (int, float)):
                continue
            col_sums = {}
            for group_key, indices in self._groups.items():
                group_vals = [values[i] for i in indices]
                col_sums[group_key] = sum(group_vals)
            result[col] = col_sums
        
        group_keys = list(self._groups.keys())
        data = {}
        for col, sums in result.items():
            data[col] = [sums.get(k, 0) for k in group_keys]
        
        return DataFrame(data, index=group_keys)
    
    def count(self):
        group_keys = list(self._groups.keys())
        counts = [len(self._groups[k]) for k in group_keys]
        return Series(counts, group_keys, name='count')
    
    def agg(self, funcs):
        result = {}
        group_keys = list(self._groups.keys())
        
        for col, func_name in funcs.items():
            values = self._df._data.get(col, [])
            col_result = []
            
            for key in group_keys:
                indices = self._groups[key]
                group_vals = [values[i] for i in indices
                             if isinstance(values[i], (int, float))]
                
                if func_name == 'mean':
                    col_result.append(sum(group_vals) / len(group_vals) if group_vals else 0)
                elif func_name == 'sum':
                    col_result.append(sum(group_vals))
                elif func_name == 'count':
                    col_result.append(len(group_vals))
                elif func_name == 'min':
                    col_result.append(min(group_vals) if group_vals else None)
                elif func_name == 'max':
                    col_result.append(max(group_vals) if group_vals else None)
            
            result[col] = col_result
        
        return DataFrame(result, index=group_keys)


# ============================================================
# Statistical Functions
# ============================================================

class Statistics:
    """Statistical analysis utilities."""
    
    @staticmethod
    def correlation(x: list, y: list) -> float:
        n = len(x)
        if n != len(y) or n < 2:
            return 0.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (n - 1)
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / (n - 1))
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / (n - 1))
        
        if std_x == 0 or std_y == 0:
            return 0.0
        
        return cov / (std_x * std_y)
    
    @staticmethod
    def linear_regression(x: list, y: list) -> Tuple[float, float]:
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        denom = n * sum_x2 - sum_x ** 2
        if denom == 0:
            return 0.0, 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        
        return slope, intercept
    
    @staticmethod
    def r_squared(x: list, y: list) -> float:
        slope, intercept = Statistics.linear_regression(x, y)
        mean_y = sum(y) / len(y)
        
        ss_res = sum((yi - (slope * xi + intercept)) ** 2
                     for xi, yi in zip(x, y))
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        
        if ss_tot == 0:
            return 1.0
        
        return 1 - ss_res / ss_tot
    
    @staticmethod
    def t_test(sample1: list, sample2: list) -> Tuple[float, float]:
        n1, n2 = len(sample1), len(sample2)
        mean1 = sum(sample1) / n1
        mean2 = sum(sample2) / n2
        
        var1 = sum((x - mean1) ** 2 for x in sample1) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in sample2) / (n2 - 1)
        
        se = math.sqrt(var1 / n1 + var2 / n2)
        if se == 0:
            return 0.0, 1.0
        
        t_stat = (mean1 - mean2) / se
        
        # Approximate p-value using normal approximation
        df = n1 + n2 - 2
        p_value = 2 * (1 - Statistics._normal_cdf(abs(t_stat)))
        
        return t_stat, p_value
    
    @staticmethod
    def _normal_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    @staticmethod
    def percentile(data: list, p: float) -> float:
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
    
    @staticmethod
    def moving_average(data: list, window: int) -> list:
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            window_data = data[start:i + 1]
            result.append(sum(window_data) / len(window_data))
        return result
    
    @staticmethod
    def exponential_moving_average(data: list, alpha: float = 0.3) -> list:
        result = [data[0]]
        for i in range(1, len(data)):
            ema = alpha * data[i] + (1 - alpha) * result[-1]
            result.append(ema)
        return result`,
				},
			},
		},
	})
}
