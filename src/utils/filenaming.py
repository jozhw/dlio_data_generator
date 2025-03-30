import datetime
import os
from pathlib import Path


class FileNaming:
    """
    This class of methods would be used for all instances of saving into files
    for the synthetic based operations.
    """

    @staticmethod
    def _get_currenttime() -> str:
        """
        Generate a timestamp in the format YYYYMMDDTHHMMSS.

        The 'T' in the format is a seperator that separates the date and the
        time.

        Returns:
            str: A timestamp string that consists of the year, month, date, hour, minute, and seconds
        """
        current_time = datetime.datetime.now()

        date_str = current_time.strftime("%Y%m%d")
        time_str = current_time.strftime("%H%M%S")

        time = "T".join([date_str, time_str])

        return time

    @staticmethod
    def _create_dir(path: Path) -> None:
        """
        Create the path if the path does not exist.

        Args:
            path<Path>: takes a path

        """

        os.makedirs(path, exist_ok=True)

        print("The following path was created: {}".format(path))

    @staticmethod
    def generate_filename(path: Path, filename: str, fext: str) -> Path:
        """
        Generate the filename for a file of interest. The naming scheme is the
        datetime + source + filename + fext.

        The format is as follows: YYYYMMDDTHHMMSS--<filename><fext>

        Args:
            path<str>: the relative path to the directory where the file will be saved
            filename<str>: the name of the file
            fext<str>: is the file type extension of the file
            source<str>: the origin of the data used to generate the file

        Returns:
            Path: of the full relative file path.
        """

        dir_path = Path(path)

        # verify if path exists and it is a directory
        # if path does not exist prompt the user to create the path
        if not dir_path.exists():
            print(
                "The following path that was given does not exist: {}".format(dir_path)
            )
            user_input = input(
                "Would you like to create the path: \n{}\nEnter 0 (no) or 1 (yes): ".format(
                    dir_path
                )
            )
            if user_input == 1:
                FileNaming._create_dir(dir_path)

            elif user_input == 0:
                print("The path was not created")
            else:
                print("Invalid input: {}. Please enter 0 or 1".format(user_input))

        elif not dir_path.is_dir():
            raise ValueError(
                "The following path is not a directory: {}".format(dir_path)
            )

        # get the datetime portion for file naming
        fdatetime = FileNaming._get_currenttime()

        # concat all of the portions to get complete file name
        complete_filename = fdatetime + "--" + filename + "." + fext

        file_path = dir_path.joinpath(complete_filename)

        return file_path

    @staticmethod
    def generate_raw_filename(path: str, filename: str, fext: str) -> Path:
        """
        Generate the filename for a file of interest. The naming scheme is the
        filename + fext.

        The format is as follows: <filename><fext>

        Args:
            path<str>: the relative path to the directory where the file will be saved
            filename<str>: the name of the file
            fext<str>: is the file type extension of the file

        Returns:
            Path: of the full relative file path.
        """

        dir_path = Path(path)

        # verify if path exists and it is a directory
        # if path does not exist create the path
        if not dir_path.exists():
            print(
                "The following path that was given does not exist: {}".format(dir_path)
            )
            FileNaming._create_dir(dir_path)

            print("The following path was created: {}\n".format(dir_path))

        elif not dir_path.is_dir():
            raise ValueError(
                "The following path is not a directory: {}".format(dir_path)
            )

        # get the datetime portion for file naming
        # fdatetime = FileNaming._get_currenttime()

        # concat all of the portions to get complete file name
        complete_filename = filename + "." + fext

        file_path = dir_path.joinpath(complete_filename)

        return file_path
