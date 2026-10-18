from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from matplotlib import image
import numpy as np
from typing import Literal, Any 
import cv2
import re
import io
from PIL import Image
import base64

DEFAULT_TITLE_PROMPT = "Read the book title from this image. Return only the title as plain text, with no author name, subtitle, or commentary.If no title is legible, return NONE."

def parse_title(raw_text : str) -> str:
    """ Normalizes a VLM's raw output into bare text.

    Args:
        raw_text (str): The raw vlm query response

    Returns:
        str: Normalized text with no <think> block , markdowns , prefixes or any stray quotations
    """
    
    text = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.DOTALL) # Drops the reasoning block if exists in the text
    text = text.split("</think>")[-1].strip()                        # unclosed reasoning block
    
    if not text:
        return ""
    
    text = text.splitlines()[0].replace("**", "").replace("`", "").strip()  # first line, no markdown
    text = re.sub(r"^(book title|title)\s*[:\-]\s*", "", text, flags=re.I)  # "Title: ..." prefix
    text = text.strip("_\"'“”‘’ ")                                          # stray quotes
    
    if text.upper().strip(" .") in {"NONE", "N/A", "NO TITLE"}:
        return ""
    
    return text.strip()
    
def _resize_max_side(image : np.ndarray, max_side : int ) -> np.ndarray :
    """ Resizes an image so its longest side matches max_side_length, maintaining the original aspect ratio.

    Args:
        image (np.ndarray): The image to resize.
        max_side (int): The max value the image's height or width is allowed to have.

    Returns:
        np.ndarray: The resized image.
    """
    height, width = image[:2]
    
    if max(height, width) <= max_side : 
        return image

    #Determining the scaling ratio for the image 
    if width > height: 
        ratio = max_side/float(width)
        new_width = max_side
        new_height = int(ratio * height)
        
    else: 
        ratio = max_side/float(height)
        new_height = max_side
        new_width = int(ratio * width)
        
    resized_img = cv2.resize(image, (new_height, new_width), interpolation= cv2.INTER_AREA)
    
    return resized_img

def _to_url(image : np.ndarray) -> str:
    """Encodes an RGB array as a PNG 'data URL' so it can travel inside the JSON body."""
    
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f"data:image/png;base64,{encoded}"
    

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

# The current design will only be implementing the following two vlms but if in future if I add more vlm extractors, a seprate 
# BaseVLMExtractor class sould be created to store the duplicated functionalities cross them.

class OpenAiVLMExtractor(BaseTextExtractor):
    name: str = "OpenaiVLM"

    def __init__(self,
                 model: str,
                 base_url: str = "http://localhost:8000/v1",
                 api_key_env: str | None = None,
                 prompt: str = DEFAULT_TITLE_PROMPT,
                 max_tokens: int = 256,
                 temperature: float = 0.0,
                 max_side: int = 1500,
                 timeout: float = 120.0,):
        """Initializes a openai api compatible vlm with the provided parameters and lazy-loads the openai library

        Args:
            model (str): The model to initialize
            base_url (str): Server address including the "/v1" suffix. Defaults to "http://localhost:8000/v1".
            api_key_env (str | None, optional): Name of the environment variable holding the api key. Defaults to None.
            prompt (str, optional): Instruction sent along with the image. Defaults to DEFAULT_TITLE_PROMPT.
            max_tokens (int, optional): Max tokens alowed to be generated for this task. Defaults to 256.
            temperature (float, optional): A hyperparameter that controls the randomness and creativity of the model's text generation. Defaults to 0.0.
            max_side (int, optional): The max height or width an image is allowed to have before it's downscaled. Defaults to 1500.
            timeout (float, optional): Seconds to wait for a response before raising an error.. Defaults to 120.0.
        """
        
        import os
        from openai import OpenAI
        
        api_key = (os.environ.get(api_key_env, "") if api_key_env else "") or "EMPTY"
        self.model = model
        self.engine = f"{self.name}-{self.model}"
        self._reader = OpenAI(base_url= base_url, api_key= api_key, timeout= timeout)
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.max_side = max_side
        self.temprature = temperature
    
    
    def extract(self, image : np.ndarray) -> ExtractionResult:
        image = _resize_max_side(image, self.max_side)
        
        response = self._reader.chat.completions.create(
            model = self.model,
            messages = [
                {
                    'role': 'user',
                    'content' : [
                        {'type': 'text', 'text': self.prompt},
                        {'type': 'image_url', 'image_url':{'url': _to_url(image)}}
                    ],
                }
            ],
            max_tokens= self.max_tokens,
            temperature= self.temprature,
            )
        
        choice = response.choices[0]
        raw = choice.message.content or "" # incase it's None
        
        title = parse_title(raw)
        if not title:
            return self._empty("response_empty", raw_text = raw)

        return ExtractionResult(
            text=title,
            confidence=None,
            bbox=[],  # a VLM answers with text only; it has no boxes
            engine=self.engine,
            meta={"n_regions": 1, "raw": raw,},
        )
        
class Qwen3VlMExtractor(BaseTextExtractor):
    def __init__(self):
        pass
    
    def extract(self, image : np.ndarray) -> ExtractionResult:
        pass