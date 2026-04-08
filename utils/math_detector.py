"""
Spoken Math Detector for HoloLearn Video Extractor

Detects mathematical expressions in Whisper-transcribed audio text
and converts them to LaTeX notation.

Usage:
    detector = SpokenMathDetector()
    result = detector.detect("so x squared plus y squared equals z squared")
    print(result['latex'])        # "x^{2} + y^{2} = z^{2}"
    print(result['has_math'])     # True
    print(result['equations'])    # list of detected equation strings
"""

import re
from typing import Dict, List, Tuple


# ── Greek letter mapping (spoken name -> LaTeX) ──────────────────────────────
GREEK_MAP = {
    'alpha': r'\alpha', 'beta': r'\beta', 'gamma': r'\gamma',
    'delta': r'\delta', 'epsilon': r'\epsilon', 'zeta': r'\zeta',
    'eta': r'\eta', 'theta': r'\theta', 'iota': r'\iota',
    'kappa': r'\kappa', 'lambda': r'\lambda', 'mu': r'\mu',
    'nu': r'\nu', 'xi': r'\xi', 'pi': r'\pi',
    'rho': r'\rho', 'sigma': r'\sigma', 'tau': r'\tau',
    'upsilon': r'\upsilon', 'phi': r'\phi', 'chi': r'\chi',
    'psi': r'\psi', 'omega': r'\omega',
}

# ── Spoken operator / relation -> LaTeX ──────────────────────────────────────
OPERATOR_MAP = {
    'plus': '+',
    'minus': '-',
    'times': r'\times',
    'multiplied by': r'\times',
    'divided by': r'\div',
    'over': r'\div',
    'equals': '=',
    'equal to': '=',
    'is equal to': '=',
    'not equal to': r'\neq',
    'not equal': r'\neq',
    'greater than': '>',
    'less than': '<',
    'greater than or equal to': r'\geq',
    'greater than or equal': r'\geq',
    'less than or equal to': r'\leq',
    'less than or equal': r'\leq',
    'approximately': r'\approx',
    'approximately equal to': r'\approx',
    'proportional to': r'\propto',
    'infinity': r'\infty',
    'plus or minus': r'\pm',
}

# ── Spoken math patterns (regex pattern, replacement function/template) ──────
# Each entry: (compiled_regex, replacement_string_or_callable)
# Patterns are applied in order; more specific patterns first.

def _build_patterns() -> List[Tuple[re.Pattern, str]]:
    """Build the ordered list of spoken-math -> LaTeX substitution patterns."""
    patterns = []

    # --- Functions with arguments ---
    # "square root of X" -> \sqrt{X}
    patterns.append((
        re.compile(r'\bsquare\s+root\s+of\s+(\w+)', re.IGNORECASE),
        r'\\sqrt{\1}'
    ))
    # "cube root of X" -> \sqrt[3]{X}
    patterns.append((
        re.compile(r'\bcube\s+root\s+of\s+(\w+)', re.IGNORECASE),
        r'\\sqrt[3]{\1}'
    ))
    # "the nth root of X" -> \sqrt[n]{X}
    patterns.append((
        re.compile(r'\bthe\s+(\w+)\s+root\s+of\s+(\w+)', re.IGNORECASE),
        r'\\sqrt[\1]{\2}'
    ))

    # --- Integrals ---
    # "integral of X dx" or "integral of X"
    patterns.append((
        re.compile(r'\bintegral\s+of\s+(.+?)\s+d([a-zA-Z])\b', re.IGNORECASE),
        r'\\int \1 \\, d\2'
    ))
    patterns.append((
        re.compile(r'\bintegral\s+from\s+(\w+)\s+to\s+(\w+)\s+of\s+(.+?)\s+d([a-zA-Z])\b', re.IGNORECASE),
        r'\\int_{\1}^{\2} \3 \\, d\4'
    ))
    patterns.append((
        re.compile(r'\bintegral\s+of\s+(\w+)', re.IGNORECASE),
        r'\\int \1'
    ))

    # --- Summation ---
    # "sum from i equals 1 to n of X" or "summation of X"
    patterns.append((
        re.compile(
            r'\b(?:sum|summation)\s+from\s+(\w+)\s*(?:equals|=)\s*(\w+)\s+to\s+(\w+)\s+of\s+(.+?)(?:\.|,|$)',
            re.IGNORECASE
        ),
        r'\\sum_{\1=\2}^{\3} \4'
    ))
    patterns.append((
        re.compile(r'\b(?:sum|summation)\s+of\s+(\w+)', re.IGNORECASE),
        r'\\sum \1'
    ))

    # --- Limits ---
    # "limit as x approaches a of f(x)"
    patterns.append((
        re.compile(
            r'\blimit\s+as\s+(\w+)\s+(?:approaches|goes\s+to|tends\s+to)\s+(\w+)\s+of\s+(.+?)(?:\.|,|$)',
            re.IGNORECASE
        ),
        r'\\lim_{\1 \\to \2} \3'
    ))
    patterns.append((
        re.compile(
            r'\blimit\s+as\s+(\w+)\s+(?:approaches|goes\s+to|tends\s+to)\s+(\w+)',
            re.IGNORECASE
        ),
        r'\\lim_{\1 \\to \2}'
    ))

    # --- Derivatives ---
    # "partial derivative" must come BEFORE "derivative" to match first
    # "partial derivative of f with respect to x" -> \frac{\partial f}{\partial x}
    patterns.append((
        re.compile(
            r'\bpartial\s+derivative\s+of\s+(\w+)\s+with\s+respect\s+to\s+(\w+)',
            re.IGNORECASE
        ),
        r'\\frac{\\partial \1}{\\partial \2}'
    ))
    # "derivative of f with respect to x" -> \frac{df}{dx}
    patterns.append((
        re.compile(
            r'\bderivative\s+of\s+(\w+)\s+with\s+respect\s+to\s+(\w+)',
            re.IGNORECASE
        ),
        r'\\frac{d\1}{d\2}'
    ))
    # "d y d x" or "dy dx" -> \frac{dy}{dx}
    patterns.append((
        re.compile(r'\bd\s*([a-zA-Z])\s+d\s*([a-zA-Z])\b', re.IGNORECASE),
        r'\\frac{d\1}{d\2}'
    ))

    # --- Fractions ---
    # "X over Y" or "X divided by Y" -> \frac{X}{Y}
    patterns.append((
        re.compile(r'\b(\w+)\s+over\s+(\w+)\b', re.IGNORECASE),
        r'\\frac{\1}{\2}'
    ))
    patterns.append((
        re.compile(r'\b(\w+)\s+divided\s+by\s+(\w+)\b', re.IGNORECASE),
        r'\\frac{\1}{\2}'
    ))

    # --- Exponents ---
    # "X squared" -> X^{2}
    patterns.append((
        re.compile(r'\b(\w+)\s+squared\b', re.IGNORECASE),
        r'\1^{2}'
    ))
    # "X cubed" -> X^{3}
    patterns.append((
        re.compile(r'\b(\w+)\s+cubed\b', re.IGNORECASE),
        r'\1^{3}'
    ))
    # "X to the power of Y" or "X to the Y" -> X^{Y}
    patterns.append((
        re.compile(r'\b(\w+)\s+to\s+the\s+(?:power\s+of\s+)?(\w+)\b', re.IGNORECASE),
        r'\1^{\2}'
    ))

    # --- Subscripts ---
    # "X sub Y" or "X subscript Y" -> X_{Y}
    patterns.append((
        re.compile(r'\b(\w+)\s+(?:sub|subscript)\s+(\w+)\b', re.IGNORECASE),
        r'\1_{\2}'
    ))

    # --- Trig / log functions ---
    for fn in ['sin', 'cos', 'tan', 'cot', 'sec', 'csc',
               'arcsin', 'arccos', 'arctan', 'log', 'ln', 'exp']:
        patterns.append((
            re.compile(rf'\b{fn}\s+of\s+(\w+)', re.IGNORECASE),
            rf'\\{fn}({{\1}})'
        ))
        # Also match bare "sin x", "log x"
        patterns.append((
            re.compile(rf'\b{fn}\s+(\w+)\b', re.IGNORECASE),
            rf'\\{fn} \1'
        ))

    # --- Absolute value ---
    patterns.append((
        re.compile(r'\babsolute\s+value\s+of\s+(\w+)', re.IGNORECASE),
        r'|\1|'
    ))

    # --- Function notation ---
    # "f of x" -> f(x)
    patterns.append((
        re.compile(r'\b([a-zA-Z])\s+of\s+([a-zA-Z])\b'),
        r'\1(\2)'
    ))

    return patterns


SPOKEN_MATH_PATTERNS = _build_patterns()


class SpokenMathDetector:
    """Detect and convert spoken mathematical expressions to LaTeX."""

    # Phrases that signal the surrounding text is mathematical
    MATH_INDICATOR_PHRASES = [
        'squared', 'cubed', 'to the power', 'square root', 'cube root',
        'integral', 'derivative', 'summation', 'sum from', 'sum of',
        'limit as', 'approaches', 'tends to',
        'divided by', 'over', 'multiplied by',
        'equals', 'equal to', 'not equal',
        'greater than', 'less than',
        'plus or minus',
        'f of x', 'g of x',
        'with respect to',
        'partial derivative',
        'absolute value',
        'log of', 'ln of', 'sin of', 'cos of', 'tan of',
        'theta', 'alpha', 'beta', 'gamma', 'delta', 'epsilon',
        'lambda', 'sigma', 'pi', 'omega', 'phi', 'mu',
    ]

    def detect(self, text: str) -> Dict:
        """
        Detect spoken math in text and convert to LaTeX.

        Args:
            text: Raw transcript text from Whisper.

        Returns:
            {
                'has_math': bool,
                'original': str,
                'latex': str,           # Full text with math portions converted
                'equations': list[str], # Extracted LaTeX equation snippets
            }
        """
        if not text or not isinstance(text, str):
            return {
                'has_math': False,
                'original': text or '',
                'latex': '',
                'equations': [],
            }

        has_math = self._contains_math(text)
        if not has_math:
            return {
                'has_math': False,
                'original': text,
                'latex': '',
                'equations': [],
            }

        latex_text = self._convert_to_latex(text)
        equations = self._extract_equation_snippets(text, latex_text)

        return {
            'has_math': True,
            'original': text,
            'latex': latex_text,
            'equations': equations,
        }

    def _contains_math(self, text: str) -> bool:
        """Check if text contains spoken math indicators."""
        text_lower = text.lower()

        for phrase in self.MATH_INDICATOR_PHRASES:
            if phrase in text_lower:
                return True

        # Check for Greek letter names
        for name in GREEK_MAP:
            if re.search(rf'\b{name}\b', text_lower):
                return True

        return False

    def _convert_to_latex(self, text: str) -> str:
        """Convert spoken math phrases in text to LaTeX notation."""
        result = text

        # Apply structural patterns first (integrals, sums, limits, etc.)
        for pattern, replacement in SPOKEN_MATH_PATTERNS:
            result = pattern.sub(replacement, result)

        # Replace spoken operators
        # Sort by length (longest first) to avoid partial matches
        for spoken, latex_str in sorted(OPERATOR_MAP.items(), key=lambda x: -len(x[0])):
            # Use lambda to avoid re.sub interpreting backslashes in replacement
            result = re.sub(
                rf'\b{re.escape(spoken)}\b',
                lambda m, r=latex_str: r,
                result,
                flags=re.IGNORECASE
            )

        # Replace Greek letter names
        for name, latex_str in GREEK_MAP.items():
            result = re.sub(
                rf'\b{name}\b',
                lambda m, r=latex_str: r,
                result,
                flags=re.IGNORECASE
            )

        return result

    def _extract_equation_snippets(self, original: str, latex_text: str) -> List[str]:
        """
        Extract the math-converted portions as standalone equation snippets.

        Compares original vs converted text to find regions that changed,
        then returns those LaTeX snippets.
        """
        equations = []

        # Strategy: find LaTeX commands in the converted text
        # These are the portions that were actually converted
        latex_command_pattern = re.compile(
            r'(?:'
            r'\\(?:frac|sqrt|int|sum|lim|partial|alpha|beta|gamma|delta|epsilon|'
            r'zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|'
            r'upsilon|phi|chi|psi|omega|times|div|neq|geq|leq|approx|propto|'
            r'infty|pm|sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|log|ln|exp|to)'
            r'[^.!?\n]*'       # capture the rest of the expression until sentence end
            r'|'
            r'\w+\^{[^}]+}'   # X^{2} style
            r'[^.!?\n]*'
            r'|'
            r'\w+_{[^}]+}'    # X_{i} style
            r'[^.!?\n]*'
            r')'
        )

        for match in latex_command_pattern.finditer(latex_text):
            snippet = match.group(0).strip()
            if snippet and len(snippet) > 2:
                # Clean trailing punctuation/whitespace
                snippet = snippet.rstrip(' ,.')
                if snippet not in equations:
                    equations.append(snippet)

        return equations


# ── Module-level convenience function ────────────────────────────────────────
_detector = None

def detect_spoken_math(text: str) -> Dict:
    """Module-level convenience function for spoken math detection."""
    global _detector
    if _detector is None:
        _detector = SpokenMathDetector()
    return _detector.detect(text)


# ── Testing ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        "so x squared plus y squared equals z squared",
        "the derivative of f with respect to x",
        "take the integral of x squared dx",
        "the limit as x approaches infinity of one over x",
        "we have alpha plus beta equals gamma",
        "square root of two times pi",
        "sum from i equals 1 to n of x sub i",
        "f of x equals x cubed minus two x plus one",
        "this is a normal sentence with no math",
        "a divided by b plus c over d",
        "the partial derivative of f with respect to y",
        "x to the power of n minus one",
        "log of x plus ln of y",
    ]

    detector = SpokenMathDetector()

    print("=" * 70)
    print("SPOKEN MATH DETECTOR - TEST RESULTS")
    print("=" * 70)

    for text in test_cases:
        result = detector.detect(text)
        print(f"\nInput:    {text}")
        print(f"Has math: {result['has_math']}")
        if result['has_math']:
            print(f"LaTeX:    {result['latex']}")
            if result['equations']:
                print(f"Snippets: {result['equations']}")
        print("-" * 70)
