import pandas as pd
import numpy as np


from mpi4py import MPI
from pathlib import Path

# customized modules
from deflate_data_generator import DeflateDataGenerator as deflate

from utils.compression import Compression
from utils.filenaming import FileNaming
from utils.validations import Validations
from utils.saving import Saving
from utils.distribution_generator import DistributionGenerator as distribution

# type hinting
from typing import Any, Dict, List, Callable, Union


class DataGenerator:
    ACCEPTED_COMPRESSION_TYPES = ["npz", "jpg"]

    def __init__(
        self,
        data_path: str,
        num_files: Union[str, None, int],
        compression_type: str,
        save_path: Path,
        analyze: Dict,
    ):

        # whether or not to run the analysis or not

        self.analyze: bool = analyze["analyze"]
        self.analysis_save_dir: Union[None, Path] = analyze["analysis_save_dir"]
        self.analysis_fname: Union[None, str] = analyze["analysis_fname"]

        self.cr_name = "{}_compression_ratio".format(compression_type)

        self.data_path = data_path
        self.num_files: Union[str, None, int] = num_files

        # not static
        self.total_files = None
        self.compression_type: str = compression_type
        self.save_path = save_path

        # load the deflate models
        self.deflate_models: Dict = deflate.load_models(
            deflate.inverse_path, deflate.nls_path
        )

    def _validations(self) -> int:
        Validations.validate_compression_types(
            DataGenerator.ACCEPTED_COMPRESSION_TYPES, [self.compression_type]
        )

        return 0

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)

        return df

    def _analyze_deflate(self) -> int:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            df = self._load_data()
            if isinstance(self.num_files, int):

                params = distribution.get_params(
                    df, self.num_files, cr_type=self.compression_type
                )
                num_rows = min(self.num_files, len(df))
            else:

                params = distribution.get_params(
                    df, len(df), cr_type=self.compression_type
                )
                num_rows = len(df)
            crs = params[self.cr_name].astype(np.float32)
            xdims = params["dimensions"].astype(np.uint32)

        else:
            df = None
            params = None
            crs = None
            xdims = None
            num_rows = None

        self.total_rows = comm.bcast(num_rows, root=0)
        crs = comm.bcast(crs, root=0)
        xdims = comm.bcast(xdims, root=0)

        num_rows = self.total_rows
        rows_per_process = num_rows // size
        start = rank * rows_per_process
        end = start + rows_per_process if rank != size - 1 else num_rows

        results: List[Dict[str, Any]] = []
        for i in range(self.total_rows)[start:end]:

            xside = xdims[i]
            ocratio = crs[i]
            sdimensions = (xside, xside, 3)
            ssize = (xside**2) * 3

            # synthetics
            sfile_name: str = "synthetic-" + str(i)

            synthetic_image = deflate.generate_synthetic_image(
                ocratio, xside, self.deflate_models
            )

            result = Compression.compress_and_calculate(
                sfile_name, synthetic_image, sdimensions, ["npz"], self.save_path
            )

            result["uncompressed_size"] = ssize
            result["file_name"] = sfile_name

            results.append(result)

        gathered_results = comm.gather(results, root=0)

        if rank == 0 and gathered_results is not None:

            # get number of images to be processed
            num_rows = self.num_files

            if self.analysis_save_dir is None:
                raise ValueError("Analysis save directory cannot be None")
            if self.analysis_fname is None:
                raise ValueError("Analysis file name cannot be None")

            filepath = FileNaming.generate_filename(
                self.analysis_save_dir, self.analysis_fname, "csv"
            )

            # merge results from all processes
            flat_results = [
                result for sublist in gathered_results for result in sublist
            ]

            # save to csv here

            Saving.save_to_csv(flat_results, filepath)

        return 0

    def _generate_deflate(self) -> int:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            df = self._load_data()
            params = distribution.get_params(
                df, self.num_files, cr_type=self.compression_type
            )
            crs = params[self.cr_name].astype(np.float32)
            xdims = params["dimensions"].astype(np.uint32)

        else:
            df = None
            params = None
            crs = None
            xdims = None

        crs = comm.bcast(crs, root=0)
        xdims = comm.bcast(xdims, root=0)

        num_rows = self.num_files
        rows_per_process = num_rows // size
        start = rank * rows_per_process
        end = start + rows_per_process if rank != size - 1 else num_rows

        for i in range(self.num_files)[start:end]:

            xside = xdims[i]
            ocratio = crs[i]

            # synthetics
            sfile_name: str = "synthetic-" + str(i)

            synthetic_image = deflate.generate_synthetic_image(
                ocratio, xside, self.deflate_models
            )

            Compression.compress(
                sfile_name, synthetic_image, ["npz"], self.save_path, remove=False
            )

        return 0

    def _run_deflate(self) -> int:
        """
        Run the deflate compression depending on whether or not want analysis
        """

        if self.analyze:
            self._analyze_deflate()
        else:
            self._generate_deflate()

        return 0

    def run(self):

        COMPRESSION_ALGORITHMS: Dict[str, Callable] = {"npz": self._run_deflate}
        COMPRESSION_ALGORITHMS[self.compression_type]()

        return 0
