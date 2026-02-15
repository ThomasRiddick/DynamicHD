import warnings
import configparser
import csv

class Dataset:

    def __init__(self,name,filename,filetype,datatype,
                 desc,number_of_dimensions,date_range):
        self.name = name
        self.filename = filename
        self.filetype = filetype
        self.datatype = datatype
        self.desc = desc
        self.number_of_dimensions = number_of_dimensions
        self.date_range = date_range
        self.data = None
        self.dimensions = None
        self.loaded = False
        if not self.filetype in ["nc","csv"]:
            raise RuntimeError("Filetype not recognised")
        if self.filetype == "nc" and not self.filename.endswith(".nc"):
            warnings.warn("Netcdf file doesn't have normal file extension (nc)")
        if self.filetype == "csv" and self.number_of_dimensions > 2:
            raise RuntimeError("Can't read more than two dimension from CSV file")
        if self.number_of_dimensions > 3:
            raise RuntimeError("Can't read more than three dimensions")

    def load(self):
        if self.filetype == "csv":
            with open(self.filename,"r") as f:
                self.data = []
                self.dimensions = []
                data_reader = csv.reader(f)
                if self.number_of_dimensions == 1:
                    for i,row in enumerate(data_reader):
                        if i == 0:
                            self.dimensions = row
                        elif i == 1:
                            self.data = row
                        else:
                            RuntimeError("Too many lines in csv file for 1d data")
                elif self.number_of_dimension == 2:
                    second_dimension = []
                    for i,row in enumerate(data_reader):
                        if i == 0:
                            self.dimensions = row
                        else:
                            second_dimension.append(row[0])
                            self.data.append(row[1:])
                    self.dimensions.append(second_dimension)
                else:
                    raise RuntimeError("Wrong number of dimension in csv datafile")
        elif self.filetype == "nc":
            raise RuntimeError("Reading netcdf datasets not yet implementeD")
        self.loaded = True

    def get_data(self):
        if not self.loaded:
            self.load()
        return self.dimensions,self.data

class DatasetManager:

    def __init__(self,catalog_filename=None):
        self.datasets = {}
        if catalog_filename is not None:
            self.read_dataset_catalog(catalog_filename)

    def read_dataset_catalog(self,catalog_filename):
        config = configparser.ConfigParser()
        config.read(catalog_filename)
        for section in config.sections():
            config_section = config[section]
            self.datasets[section] = \
                Dataset(name=section.split(":")[1],
                        filename=config_section.get("filename"),
                        filetype=config_section.get("filetype"),
                        datatype=section.split(":")[0],
                        desc=config_section.get("desc"),
                        number_of_dimensions=
                        int(config_section.get("ndims")),
                        date_range=[int(date) for date in
                                    config_section.get("data_range").split(",")])

    def get_dataset(self,datatype,name):
        return self.datasets[f'{datatype}:{name}']

    def get_datasets_by_type(self,datatype):
        return [dataset.name for dataset in self.datasets.values()
                if dataset.datatype == datatype]

# if __name__ == '__main__':
#     dsm = DatasetManager("/Users/thomasriddick/Documents/data/temp/lakes_datasets.ini")
#     ds = dsm.get_dataset("agassizoutlet-time","test1")
#     print(ds.get_data())




