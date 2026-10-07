#TODO: Create a Extraction Factory that handles the different text extraction configurations for the pipeine
#TODO: Create a API call to get book data and log it to the database
#TODO: Create embeddings of the text and calculate cosine similarity to find the most relevant recommendations

  
def get_crops(results,image):
  crops = []
  for result in results:
    boxes = result.boxes
    for box in boxes:
      #Getting the bounding box and then cropping them
      x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
      crop = image[y1:y2, x1:x2]
      crops.append(crop)
  return crops


