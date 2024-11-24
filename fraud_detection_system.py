from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import faiss
import numpy as np
import re
import logging
import os
from langchain_ollama import OllamaEmbeddings

HISTORICAL_DATA_DIM = 768  # Nomic embedding dimension
VENDOR_DATA_DIM = 768  # Nomic embedding dimension
UNUSUAL_CHARS_THRESHOLD = 5
INCONSISTENT_CAPS_THRESHOLD = 0.3
VENDOR_SIMILARITY_THRESHOLD = 0.7
HISTORICAL_SIMILARITY_THRESHOLD = 0.6
HIGH_VALUE_THRESHOLD = 10000

class InvoiceFraudDetectionSystem:
    def __init__(self, historical_index_path, vendor_index_path):
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-vision-128k-instruct",device_map="cuda",trust_remote_code=True,torch_dtype="auto")
        self.processor = AutoProcessor.from_pretrained("microsoft/Phi-3-vision-128k-instruct")
        self.historical_data = self.load_index(historical_index_path, HISTORICAL_DATA_DIM)
        self.vendor_details = self.load_index(vendor_index_path, VENDOR_DATA_DIM)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        
    def load_index(self, index_path, dimension):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        return {'index': faiss.read_index(index_path)}
    
    
    def analyze_invoice(self, image_path):
        try:
            image = Image.open(image_path)
        except IOError:
            logging.error(f"Unable to open image file: {image_path}")
            return None, ["Error: Unable to process invoice image"]

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        analysis = self.processor.decode(outputs[0], skip_special_tokens=True)
        anomalies = self.check_anomalies(analysis)
        return analysis, anomalies    
    

    def check_anomalies(self, analysis):
        anomalies = []

        if self.detect_altered_text(analysis):
            anomalies.append("Possible altered text detected")

        if self.detect_inconsistent_formatting(analysis):
            anomalies.append("Inconsistent formatting detected")

        if self.detect_suspicious_vendor(analysis):
            anomalies.append("Suspicious vendor information detected")

        if self.compare_historical_patterns(analysis):
            anomalies.append("Deviates from historical patterns")

        return anomalies
    

    def detect_altered_text(self, analysis):
        unusual_chars = re.findall(r'[^\w\s,.$]', analysis)
        if len(unusual_chars) > UNUSUAL_CHARS_THRESHOLD:
            return True
        
        words = analysis.split()
        inconsistent_caps = sum(1 for word in words if word.istitle() != words[0].istitle())
        if inconsistent_caps > len(words) * INCONSISTENT_CAPS_THRESHOLD:
            return True
        
        return False
    
    
    def detect_inconsistent_formatting(self, analysis):
        date_formats = re.findall(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', analysis)
        if len(set(date_formats)) > 1:
            return True
        
        number_formats = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', analysis)
        decimal_counts = [num.count('.') for num in number_formats]
        if len(set(decimal_counts)) > 1:
            return True
        
        lines = analysis.split('\n')
        left_aligned = sum(1 for line in lines if line.lstrip() == line)
        if left_aligned != 0 and left_aligned != len(lines):
            return True
        
        return False
    
    
    def detect_suspicious_vendor(self, analysis):
        vector = self.get_embedding(analysis)
        
        D, I = self.vendor_details['index'].search(np.array([vector]), k=1)
        
        if D[0][0] > VENDOR_SIMILARITY_THRESHOLD:
            return True
        
        suspicious_keywords = ['urgent', 'confidential', 'immediate payment', 'offshore']
        if any(keyword in analysis.lower() for keyword in suspicious_keywords):
            return True
        
        return False
    
    
    def compare_historical_patterns(self, analysis):
        vector = self.get_embedding(analysis)
        
        D, I = self.historical_data['index'].search(np.array([vector]), k=5)
        
        if np.mean(D[0]) > HISTORICAL_SIMILARITY_THRESHOLD:
            return True
        
        amounts = re.findall(r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', analysis)
        if amounts:
            max_amount = max(float(amount.replace(',', '')) for amount in amounts)
            if max_amount > HIGH_VALUE_THRESHOLD:
                return True
        
        return False
    
    
    def get_embedding(self, text):
        return self.embeddings.embed_query(text)