from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ExtractionResult:
    """Data class that holds the blueprint for the text extraction layer's results.
    
    Attributes: 
        text (str): The extracted text.
        confidence (float | None): The confidence level of the extraction.
        engine (str): The engine used for extraction.
        meta (dict): Additional metadata about the extraction.
    """
    text: str = ''
    confidence: float | None = None
    engine: str = ''
    meta: dict = field(default_factory=dict)


class BaseTextExtractor(ABC):
    
    name: str = "base"
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> ExtractionResult:
        """An abstract method that defines the interface for text extraction from an image.

        Args:
            image (np.ndarray): The input image from which text needs to be extracted.

        Returns:
            ExtractionResult: An instance of ExtractionResult containing the extracted text, confidence level, engine used, and any additional metadata.
        """
        pass

class EasyOcrExtractor(BaseTextExtractor):
    
    name: str = "easyocr"
    
    def __init__(self, languages: tuple = ('en', ), gpu: bool = False):
        """Initializes the EasyOcrExtractor with specified languages and GPU usage while lazy-loading the EasyOCR reader.
        
        Args: 
            languages (tuple): A tuple of language codes for text extraction. Defaults to ('en',).
            gpu (bool): A flag indicating whether to use GPU for extraction. Defaults to False.
        """
        
        import easyocr
        self.reader = easyocr.Reader(list(languages), gpu = gpu)
        
    def extract():
        pass
    
class PaddleOcrExtractor(BaseTextExtractor):
    pass

class TesseractExtractor(BaseTextExtractor):
    pass

class VLMExtractor(BaseTextExtractor):
    pass

