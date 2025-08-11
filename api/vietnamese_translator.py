"""
Vietnamese to English Translation Module
Sử dụng Google Translate API miễn phí để dịch tiếng Việt sang tiếng Anh
"""

import requests
import urllib.parse
import json
import time
from typing import Optional

class VietnameseTranslator:
    def __init__(self):
        self.base_url = "https://translate.googleapis.com/translate_a/single"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def translate_vi_to_en(self, vietnamese_text: str, max_retries: int = 3) -> Optional[str]:
        """
        Dịch text từ tiếng Việt sang tiếng Anh
        
        Args:
            vietnamese_text: Text tiếng Việt cần dịch
            max_retries: Số lần thử lại nếu failed
            
        Returns:
            Translated English text hoặc None nếu failed
        """
        if not vietnamese_text or not vietnamese_text.strip():
            return vietnamese_text
            
        for attempt in range(max_retries):
            try:
                # Prepare parameters
                params = {
                    'client': 'gtx',
                    'sl': 'vi',  # source language: Vietnamese
                    'tl': 'en',  # target language: English
                    'dt': 't',   # return translation
                    'q': vietnamese_text
                }
                
                # Make request
                response = self.session.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                
                # Extract translated text
                if result and len(result) > 0 and len(result[0]) > 0:
                    translated_parts = []
                    for part in result[0]:
                        if part[0]:
                            translated_parts.append(part[0])
                    
                    translated_text = ''.join(translated_parts).strip()
                    
                    if translated_text:
                        print(f"Translation: '{vietnamese_text}' → '{translated_text}'")
                        return translated_text
                
                print(f"Empty translation result for: {vietnamese_text}")
                return None
                
            except requests.exceptions.RequestException as e:
                print(f"Translation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    
            except (json.JSONDecodeError, IndexError, KeyError) as e:
                print(f"Failed to parse translation response: {e}")
                return None
                
        print(f"All translation attempts failed for: {vietnamese_text}")
        return None
    
    def translate_with_fallback(self, text: str) -> str:
        """
        Dịch text với fallback strategy
        
        Args:
            text: Input text (có thể là tiếng Việt hoặc tiếng Anh)
            
        Returns:
            Translated text hoặc original text nếu dịch failed
        """
        # Detect if text contains Vietnamese characters
        vietnamese_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
        
        has_vietnamese = any(char.lower() in vietnamese_chars for char in text)
        
        if not has_vietnamese:
            # Already in English (or no Vietnamese chars), return as-is
            return text
            
        # Try to translate Vietnamese to English
        translated = self.translate_vi_to_en(text)
        
        if translated:
            return translated
        else:
            # Fallback: return original text if translation fails
            print(f"Translation failed, using original text: {text}")
            return text

# Global translator instance
translator = VietnameseTranslator()

def translate_vietnamese_query(query: str) -> tuple[str, str]:
    """
    Dịch query tiếng Việt sang tiếng Anh
    
    Returns:
        tuple(original_query, translated_query)
    """
    original = query.strip()
    translated = translator.translate_with_fallback(original)
    
    return original, translated

# Test function
if __name__ == "__main__":
    # Test cases
    test_queries = [
        "người đang đi bộ",
        "xe hơi đang chạy", 
        "người đang nói chuyện",
        "cảnh thiên nhiên đẹp",
        "màn hình máy tính",
        "con chó đang chạy",
        "người đang học lập trình",
        "cô gái đang cười"
    ]
    
    print("🔄 Testing Vietnamese Translation...")
    for query in test_queries:
        original, translated = translate_vietnamese_query(query)
        print(f"✅ '{original}' → '{translated}'")
