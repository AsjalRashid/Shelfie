from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from matplotlib import image
import numpy as np
from typing import Literal, Any 
import cv2

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
    bbox: list | None = None
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
 
    def __init__(self, languages: tuple[str, ...] = ("en",), use_gpu: bool = False):
        """Initializes the EasyOcrExtractor with specified languages and GPU usage while lazy-loading the EasyOCR reader.
 
        Args:
            languages: Language codes for text extraction. Defaults to ('en',).
            use_gpu: Whether to run on the GPU. Defaults to False.
        """
        import easyocr
 
        self.version = easyocr.__version__
        self.engine = f"{self.name}-{self.version}"
        self._reader = easyocr.Reader(list(languages), gpu=use_gpu)
 
    def _empty(self) -> ExtractionResult:
        return ExtractionResult(
            text="", confidence=0.0, bbox=[], engine=self.engine,
            meta={"version": self.version, "n_regions": 0},
        )
 
    def extract(self, image: np.ndarray) -> ExtractionResult:
        """Extracts text from the provided image using EasyOCR and returns an ExtractionResult.

        Args:
            image (np.ndarray): Input image from which text needs to be extracted.

        Returns:
            ExtractionResult: An instance of ExtractionResult containing the extracted text, confidence level, engine used.
        """
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
        results = self._reader.readtext(bgr)          # list of (box, text, confidence)
        if not results:
            return self._empty()
 
        items = [
            (text.strip(), float(conf), [[int(x), int(y)] for x, y in box])
            for box, text, conf in results
            if text.strip()
        ]
        if not items:
            return self._empty()
 
        texts, scores, boxes = zip(*items)

        return ExtractionResult(
            text="\n".join(texts),
            confidence=sum(scores) / len(scores),
            bbox=list(boxes),
            engine=self.engine,
            meta={"version": self.version, "n_regions": len(items)},
        )

class PaddleOcrExtractor(BaseTextExtractor):
    name: str = "paddleocr"
    def __init__(
        self,
        version: str = "PP-OCRv6",
        language: str = "en",
        use_orientation: bool = False,
        device: str | None = None,
        enable_mkldnn: bool | None = None,
    ):
        """Initializes the PaddleOcrExtractor with specified version, language, orientation selection, selection type, and device while lazy-loading the PaddleOCR reader.

        Args:
            version (str, optional): The version of PaddleOCR to use. Defaults to "PP-OCRv6".
            language (str, optional): The language code for text extraction. Defaults to 'en'.
            use_orientation (bool, optional): Whether to detect orientation of the scanned image and correct it before text extraction. Defaults to False.
            device (str | None, optional): The device to use for extraction. Defaults to None.
            enable_mkldnn (bool | None, optional): CPU only. None keeps Paddle's default; False works
                around the oneDNN NotImplementedError on some CPUs.
        """
        
        from paddleocr import PaddleOCR
        
        self.version = version
        self.engine = f"{self.name}-{version}"
        
        extra: dict[str, Any] = {}
        if enable_mkldnn is not None:
            extra["enable_mkldnn"] = enable_mkldnn
 
        self._reader = PaddleOCR(
            ocr_version=version,
            lang=language,
            device=device,
            use_doc_orientation_classify=use_orientation,
            use_textline_orientation=use_orientation,
            use_doc_unwarping=False, 
            **extra)
    def _empty(self) -> ExtractionResult:
        return ExtractionResult(
            text="", confidence=0.0, bbox=[], engine=self.engine,
            meta={"version": self.version, "n_regions": 0},
        )
 
    def extract(self, image: np.ndarray) -> ExtractionResult:
        """Extracts text from the provided image using PaddleOCR and returns an ExtractionResult.
     
             Args:
                 image (np.ndarray): Input image from which text needs to be extracted.
     
             Returns:
                 ExtractionResult: An instance of ExtractionResult containing the extracted text, confidence level, engine used.
        """
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # PaddleOCR expects BGR images
        outputs = self._reader.predict(bgr)          
        if not outputs:
            return self._empty()
 
        res = outputs[0]
        items = [
            (text.strip(), float(score), np.asarray(poly).tolist())
            for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"])
            if text.strip()
        ]
        
        if not items:
            return self._empty()
 
        texts, scores, boxes = zip(*items)
        
        return ExtractionResult(
            text="\n".join(texts),
            confidence=sum(scores) / len(scores),
            bbox=list(boxes),
            engine=self.engine,
            meta={"version": self.version, "n_regions": len(items)},
        )        
        
class TesseractExtractor(BaseTextExtractor):
    pass

class VLMExtractor(BaseTextExtractor):
    pass

