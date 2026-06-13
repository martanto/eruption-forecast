Seismic data adapters that yield a single day of ObsPy `Stream` to the rest of the pipeline. Two concrete sources implement the same `SeismicDataSource.get(date)` interface so callers don't care whether the bytes came from disk or the network.

Files:
- base.py — `SeismicDataSource` ABC; defines `get(date) -> Stream` and a log-prefix helper.
- sds.py — `SDS` local adapter: reads miniSEED from a SeisComP Data Structure tree (`{sds}/{year}/{net}/{sta}/{cha}.D/{nslc}.D.{year}.{julian}`).
- fdsn.py — `FDSN` remote adapter: queries an FDSN client and transparently caches each fetched day back into a local SDS hierarchy so re-runs are offline.
