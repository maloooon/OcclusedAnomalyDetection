from enum import Enum
# VisaDatasetCategories is an Enum class that contains the categories of the Visa dataset.
#       candle
#     capsules
#     cashew
#     chewinggum
#     fryum
#     macaroni1
#     macaroni2
#     pcb1
#     pcb2
#     pcb3
#     pcb4
#     pipe_fryum
class VisaDatasetCategory(Enum):
    candle = 'candle'
    capsules = 'capsules'
    cashew = 'cashew'
    chewinggum = 'chewinggum'
    fryum = 'fryum'
    macaroni1 = 'macaroni1'
    macaroni2 = 'macaroni2'
    pcb1 = 'pcb1'
    pcb2 = 'pcb2'
    pcb3 = 'pcb3'
    pcb4 = 'pcb4'
    pipe_fryum = 'pipe_fryum'

class VisaDatasetCategoryEncoder:
    """
    This class is used to encode the categories of the Visa dataset.
    """
    def __init__(self):
        self.category_encoder = {
            'candle': 0,
            'capsules': 1,
            'cashew': 2,
            'chewinggum': 3,
            'fryum': 4,
            'macaroni1': 5,
            'macaroni2': 6,
            'pcb1': 7,
            'pcb2': 8,
            'pcb3': 9,
            'pcb4': 10,
            'pipe_fryum': 11
        }

    def encode(self, category: str) -> int:
        """
        This method encodes the category.
        Args:
            category (str): category to be encoded
        Returns:
            int: encoded category
        """
        return self.category_encoder[category]

    def decode(self, category: int) -> str:
        """
        This method decodes the category.
        Args:
            category (int): category to be decoded
        Returns:
            str: decoded category
        """
        for key, value in self.category_encoder.items():
            if value == category:
                return key
        return None


