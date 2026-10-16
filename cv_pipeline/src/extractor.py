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
    """Base class for text extractors"""
    
    name: str = "base"
    
    def _empty(self, reason: str = "", **meta) -> ExtractionResult:
        """A method that returns ExtractionResults for cases where no text was found by an extractor.

        Args:
            reason (str, optional): A string describing the reason an empty result was returned. Defaults to "".

        Returns:
            ExtractionResult: An instance of ExtractionResult with empty text, bbox, confidence score as 0.0, name of the extractor engine , reason for empty result and other metadata
        """
        return ExtractionResult(
            text="",
            confidence=0.0,
            bbox=[],
            engine=self.engine or self.name,    
            meta={"n_regions": 0, "reason": reason, **meta},
        )
        
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
        """Initializes the EasyOcrExtractor with the provided parameters and lazy-loads the EasyOCR library.
 
        Args:
            languages: Language codes for text extraction. Defaults to ('en',).
            use_gpu: Whether to run on the GPU. Defaults to False.
        """
        import easyocr
 
        self.version = easyocr.__version__
        self.engine = f"{self.name}-{self.version}"
        self._reader = easyocr.Reader(list(languages), gpu=use_gpu)
  
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
            return self._empty("no_detections",version = self.version, )
 
        items = [
            (text.strip(), float(conf), [[int(x), int(y)] for x, y in box])
            for box, text, conf in results
            if text.strip()
        ]
        
        if not items:
            return self._empty("all_text_blank",version = self.version)
 
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
        """Initializes the PaddleOcrExtractor with the provided parameters and lazy-loads the PaddleOCR library.

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
            return self._empty("no_output", version = self.version)
 
        res = outputs[0]
        items = [
            (text.strip(), float(score), np.asarray(poly).tolist())
            for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"])
            if text.strip()
        ]
        
        if not items:
            return self._empty("all_text_blank", version = self.version)
 
        texts, scores, boxes = zip(*items)
        
        return ExtractionResult(
            text="\n".join(texts),
            confidence=sum(scores) / len(scores),
            bbox=list(boxes),
            engine=self.engine,
            meta={"version": self.version, "n_regions": len(items)},
        )        
        
class TesseractExtractor(BaseTextExtractor):
    name: str = "tesseract"

    def __init__(self,
                 language: str = 'eng',
                 psm: int = 7,
                 oem: int = 3,
                 min_confidence: float = 30.0,
                 tesseract_cmd: str | None = None):
        """Initializes the TesseractExtractor with the provided parameters and lazy-loads the pytesseract library.

        Args:
            language (str, optional): The language to use for text extraction. Defaults to 'eng'.
            psm (int, optional): The page segmentation mode to use for pytesseract. A flag in pytesseract that specifices the layout of the text for the input image. Defaults to 7.
            oem (int, optional): The OCR engine mode. A flag in pytesseract that specifices the underlying engine to use. Defaults to 3.
            min_confidence (float, optional): The minimum confidence level for text detection. Defaults to 30.0.
            tesseract_cmd (str | None, optional): Path to the tesseract executable if it is not on PATH. Defaults to None.
        """
        
        import pytesseract 
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        try:
            self.version = str(pytesseract.get_tesseract_version())
        except pytesseract.TesseractNotFoundError as e:
            raise RuntimeError(
                "Tesseract binary not found. Install it (Linux: apt-get install "
                "tesseract-ocr; Windows: UB-Mannheim installer) or pass tesseract_cmd."
            ) from e
            
        self.engine = f"{self.name}-{self.version}"
        self.language = language
        self.psm = psm
        self.oem = oem
        self.min_conf = min_confidence
        self._reader = pytesseract
    
    def _read_lines(self, image: np.ndarray) -> list[dict]:
        """Runs the Pytesseract to extract data via the imgae_to_data method and groups together the detailed per word output in a simplified list that has the
        data for the entire text.

        Args:
            image (np.ndarray): The input crop to the Extractor. 
            
        Returns:
            list[dict]: A list containing all extracted words as a string along with the confidence score and bounding box coordianates
        """
        
        data = self._reader(image, lang = self.language, config = f"--oem {self.oem} --psm {self.psm}", output_type = self._reader.Output.DICT)
        
        lines : dict[tuple,dict] = {}
        
        for i, word in enumerate (data['text']):
            word = word.strip()
            conf = float(data['conf'][i])
            
            if not word or conf < self.min_conf:
                continue
            
            x,y,w,h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # Creates key by which each block of words can be identified in the extracted text 
            key = (data['block_num'][i], data['par_num'][i], data['line_num'][i]) 
            
            line = lines.setdefault(key, {"words" : [], "confs" : [], "x1" : x ,"y1" : y, "x2" : x+w ,"y2" : y+h})
            
            # For each new word extends the bbox and text to accomodate it
            line['words'].append(word)
            line["confs"].append(conf / 100.0)
            line["x1"], line["y1"] = min(line["x1"], x), min(line["y1"], y)
            line["x2"], line["y2"] = max(line["x2"], x + w), max(line["y2"], y + h)
            

        return [  
            {
                "text": " ".join(lines[k]["words"]),
                "conf": sum(lines[k]["confs"]) / len(lines[k]["confs"]),
                "box": [
                    [lines[k]["x0"], lines[k]["y0"]], [lines[k]["x1"], lines[k]["y0"]],
                    [lines[k]["x1"], lines[k]["y1"]], [lines[k]["x0"], lines[k]["y1"]],
                ],
            }
            for k in sorted(lines)
        ]            
    
    def extract(self, image: np.ndarray) -> ExtractionResult:
        """Extracts text from the provided image using Pytesseract and returns an ExtractionResult.
    
            Args:
                image (np.ndarray): Input image from which text needs to be extracted.
    
            Returns:
                ExtractionResult: An instance of ExtractionResult containing the extracted text, confidence level, engine used and meta data about the extractor.
        """
        lines = self._read_lines(image)

        return ExtractionResult(
            text="\n".join(line["text"] for line in lines),
            confidence=sum(line["conf"] for line in lines) / len(lines),
            bbox=[line["box"] for line in lines],
            engine=self.engine,
            meta={
                "version": self.version,
                "n_regions": len(lines),
                "psm": self.psm,
                "oem" : self.oem,
            },
        )


class VLMExtractor(BaseTextExtractor):
    pass

