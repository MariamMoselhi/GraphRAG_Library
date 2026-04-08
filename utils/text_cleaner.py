"""
Text Cleaner for HoloLearn Extractor
Cleans and normalizes extracted text from various sources.
"""

import re
from typing import Optional
import unicodedata


class TextCleaner:
    """Clean and normalize extracted text"""
    
    @staticmethod
    def clean_text(text: str, 
                   remove_urls: bool = False,
                   remove_emails: bool = False,
                   fix_spacing: bool = True,
                   remove_special_chars: bool = False) -> str:
        """
        Main cleaning function - applies all cleaning steps
        
        Args:
            text: Raw text to clean
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            fix_spacing: Fix multiple spaces/newlines
            remove_special_chars: Remove non-alphanumeric characters
        
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 1. Fix encoding issues
        text = TextCleaner._fix_encoding(text)
        
        # 2. Remove URLs if requested
        if remove_urls:
            text = TextCleaner._remove_urls(text)
        
        # 3. Remove emails if requested
        if remove_emails:
            text = TextCleaner._remove_emails(text)
        
        # 4. Fix spacing issues
        if fix_spacing:
            text = TextCleaner._fix_spacing(text)
        
        # 5. Remove special characters if requested
        if remove_special_chars:
            text = TextCleaner._remove_special_chars(text)
        
        # 6. Final cleanup
        text = text.strip()
        
        return text
    
    # Unicode math characters to preserve during encoding cleanup
    _MATH_CHARS = set(
        '±×÷·∗∘∙√∛∜∞∝∠∡∢∫∬∭∮∯∰∱∲∳'
        '∑∏∐∂∆∇∈∉∊∋∌∍∎∀∁∂∃∄∅∆∇'
        '≠≈≡≢≤≥≦≧≨≩≪≫≮≯≰≱≲≳≺≻≼≽'
        '⊂⊃⊄⊅⊆⊇⊈⊉⊊⊋⊎⊏⊐⊑⊒⊓⊔⊕⊖⊗⊘⊙⊚⊛⊜⊝'
        '∪∩∧∨∴∵∶∷∸∹∺∻∼∽∾∿⊥⊦⊧⊨⊩⊪⊫⊬⊭⊮⊯'
        'αβγδεζηθικλμνξοπρσςτυφχψω'
        'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
        'ℂℕℙℚℝℤℤℬℰℱℋℐℒℳℛ'
        '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ'
        '₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔₕₖₗₘₙₚₛₜ'
        '←→↑↓↔↕↖↗↘↙⇐⇒⇑⇓⇔⇕'
        '½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞'
        '°′″‰‱ℓÅ'
    )

    @staticmethod
    def _fix_encoding(text: str) -> str:
        """
        Fix common encoding issues while preserving mathematical symbols.

        Removes only true junk (BOM, zero-width chars, control characters)
        while keeping LaTeX sequences, Unicode math symbols, and Greek letters.
        """
        # Replace problematic invisible/whitespace characters first
        replacements = {
            '\u00a0': ' ',   # Non-breaking space -> regular space
            '\u200b': '',    # Zero-width space -> remove
            '\u200c': '',    # Zero-width non-joiner -> remove
            '\u200d': '',    # Zero-width joiner -> remove
            '\ufeff': '',    # Byte order mark -> remove
            '\r\n': '\n',    # Windows line break -> Unix
            '\r': '\n',      # Old Mac line break -> Unix
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove control characters (C0/C1) except whitespace (\n, \t, space)
        cleaned = []
        for ch in text:
            cat = unicodedata.category(ch)
            # Keep the character if it is:
            #   - NOT a control char (Cc/Cf) OR is tab/newline/space
            #   - OR is in our math character set
            if cat == 'Cc':
                # Control chars: keep only \n and \t
                if ch in ('\n', '\t'):
                    cleaned.append(ch)
            elif cat == 'Cf':
                # Format chars (already handled above, drop any remaining)
                pass
            else:
                cleaned.append(ch)

        text = ''.join(cleaned)

        # Normalize to NFC (composed form) to keep characters like α, ∫, ² intact
        # NFC keeps composed characters together, unlike NFKD which decomposes them
        text = unicodedata.normalize('NFC', text)

        return text
    
    @staticmethod
    def _remove_urls(text: str) -> str:
        """
        Remove URLs from text
        Examples:
            "Visit https://example.com for more" → "Visit  for more"
            "Check www.site.com" → "Check "
        """
        # Pattern matches: http://..., https://..., www....
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        
        # Also remove www. links
        www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),])+'
        text = re.sub(www_pattern, '', text)
        
        return text
    
    @staticmethod
    def _remove_emails(text: str) -> str:
        """
        Remove email addresses
        Example: "Contact john@example.com" → "Contact "
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, '', text)
        return text
    
    @staticmethod
    def _fix_spacing(text: str) -> str:
        """
        Fix spacing issues:
        - Multiple spaces → single space
        - Multiple newlines → double newline (paragraph break)
        - Trailing/leading spaces on lines
        """
        # Remove multiple spaces → single space
        text = re.sub(r' +', ' ', text)
        
        # Remove spaces at start/end of lines
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        # Remove multiple blank lines → max 2 newlines (one blank line)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    @staticmethod
    def _remove_special_chars(text: str) -> str:
        """
        Remove junk special characters while keeping:
        - Letters (a-z, A-Z) and Unicode letters (Greek, etc.)
        - Numbers (0-9) and Unicode digits/superscripts/subscripts
        - Basic punctuation (. , ! ? - ' : ; ( ) [ ] { })
        - Math operators and symbols (+, =, <, >, /, \\, ^, _, $)
        - Spaces and newlines
        """
        def _keep(ch):
            cat = unicodedata.category(ch)
            # Letters (L*), Numbers (N*), Math symbols (Sm), whitespace
            if cat[0] in ('L', 'N'):
                return True
            if cat == 'Sm':  # Math symbols: +, =, <, >, ∫, ∑, etc.
                return True
            if ch in ' \t\n':
                return True
            if ch in '.,!?-\':;()[]{}+=/\\^_$~|@#&*"':
                return True
            return False

        return ''.join(ch for ch in text if _keep(ch))
    
    @staticmethod
    def remove_duplicate_lines(text: str) -> str:
        """
        Remove duplicate consecutive lines
        Example:
            "Hello\nHello\nWorld" → "Hello\nWorld"
        """
        lines = text.split('\n')
        cleaned_lines = []
        prev_line = None
        
        for line in lines:
            if line != prev_line:
                cleaned_lines.append(line)
                prev_line = line
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 10000) -> str:
        """
        Truncate text to maximum length
        Useful for preventing huge files from crashing agents
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length] + "\n\n[TEXT TRUNCATED - TOO LONG]"


# Example usage
if __name__ == "__main__":
    # Test cases
    messy_text = """
    This    is   messy    text.
    
    
    
    With multiple     spaces and blank lines.
    Visit https://example.com and email test@email.com
    café   résumé   naïve
    
    Same line
    Same line
    Different line
    """
    
    print("=== ORIGINAL TEXT ===")
    print(repr(messy_text))
    print("\n")
    
    # Clean with default settings
    cleaned = TextCleaner.clean_text(messy_text)
    print("=== CLEANED (default) ===")
    print(cleaned)
    print("\n")
    
    # Clean and remove URLs/emails
    cleaned_full = TextCleaner.clean_text(
        messy_text,
        remove_urls=True,
        remove_emails=True,
        fix_spacing=True
    )
    print("=== CLEANED (remove URLs & emails) ===")
    print(cleaned_full)
    print("\n")
    
    # Remove duplicate lines
    deduped = TextCleaner.remove_duplicate_lines(cleaned_full)
    print("=== CLEANED (removed duplicates) ===")
    print(deduped)