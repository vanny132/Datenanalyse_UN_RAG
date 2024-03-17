from .model import GLiNER
import logging
logger = logging.getLogger(__name__)

class GlinerNER():
    
    def __init__(self, labels = ["date","time", "club", "league"]):
        self.model = GLiNER.from_pretrained("urchade/gliner_base")
        self.labels = labels

    def predict_tags(self, texts, verbose = False):
        recognitions = []
        if isinstance(texts, str):
            texts = [texts]
        for text in texts:
            entities = self.model.predict_entities(text, self.labels)
            if verbose:
                pass
                #logger.info("Gliner Text: ", str(text))
            #print(text)
            for entity in entities:
                recognitions.append({
                    "text": entity["text"], 
                    "label": entity["label"], 
                    "start": entity["start"], 
                    "end": entity["end"]
                })
                if verbose:
                    pass
                    #logger.info("Gliner Entity: " , str(entity["text"]), "=>", str(entity["label"]))
        
        return recognitions
           