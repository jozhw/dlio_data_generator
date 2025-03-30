from typing import List


class Validations:

    @staticmethod
    def validate_compression_types(accepted_types: List[str], input_types: List[str]):

        for input_type in input_types:
            if input_type not in accepted_types:
                raise ValueError(
                    f"Invalid file type - {input_type}. Accepted file types are: {', '.join(accepted_types)}"
                )

    @staticmethod
    def validate_json_extension(json_img_path: str):
        image_path_type = json_img_path.split(".")[-1].lower()
        if image_path_type != "json":
            raise ValueError(
                "json_img_path is supposed to be a json file, but got {}".format(
                    image_path_type
                )
            )
