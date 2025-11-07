#!/usr/bin/env python3
"""
Kokoro Phonemizer - Converts text to IPA phonemes for Kokoro TTS.

Uses g2p-en (ARPABET) and converts to IPA phonemes expected by Kokoro.
"""

import json
from typing import List
from g2p_en import G2p


class KokoroPhonemizer:
    """
    Phonemizer for Kokoro TTS.

    Converts English text → ARPABET (via g2p-en) → IPA (for Kokoro).
    """

    # ARPABET to IPA mapping
    # Based on CMU Dictionary and standard IPA conventions
    ARPABET_TO_IPA = {
        # Vowels
        'AA': 'ɑ',    # odd
        'AE': 'æ',    # at
        'AH': 'ʌ',    # hut
        'AO': 'ɔ',    # ought
        'AW': 'aʊ',   # cow
        'AY': 'aɪ',   # hide
        'EH': 'ɛ',    # Ed
        'ER': 'ɜr',   # hurt (rhotic)
        'EY': 'eɪ',   # ate
        'IH': 'ɪ',    # it
        'IY': 'i',    # eat
        'OW': 'oʊ',   # oat
        'OY': 'ɔɪ',   # toy
        'UH': 'ʊ',    # hood
        'UW': 'u',    # two

        # Consonants
        'B': 'b',     # be
        'CH': 'ʧ',    # cheese
        'D': 'd',     # dee
        'DH': 'ð',    # thee
        'F': 'f',     # fee
        'G': 'g',     # green
        'HH': 'h',    # he
        'JH': 'ʤ',    # gee
        'K': 'k',     # key
        'L': 'l',     # lee
        'M': 'm',     # me
        'N': 'n',     # knee
        'NG': 'ŋ',    # ping
        'P': 'p',     # pee
        'R': 'r',     # read
        'S': 's',     # sea
        'SH': 'ʃ',    # she
        'T': 't',     # tea
        'TH': 'θ',    # theta
        'V': 'v',     # vee
        'W': 'w',     # we
        'Y': 'y',     # yield
        'Z': 'z',     # zee
        'ZH': 'ʒ',    # seizure
    }

    def __init__(self, vocab_path: str = "kokoro_phoneme_vocab.json"):
        """
        Initialize the phonemizer.

        Args:
            vocab_path: Path to Kokoro phoneme vocabulary JSON
        """
        # Initialize g2p-en
        self.g2p = G2p()

        # Load Kokoro vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        self.phoneme_to_id = vocab_data['phoneme_to_id']
        self.id_to_phoneme = {int(k): v for k, v in vocab_data['id_to_phoneme'].items()}
        self.vocab_size = vocab_data['metadata']['vocab_size']

        # Special tokens
        self.pad_token = 0
        self.space_token = self.phoneme_to_id.get(' ', 16)

        print(f"KokoroPhonemizer initialized")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  G2P backend: g2p-en (ARPABET → IPA)")

    def _arpabet_to_ipa(self, arpabet: str) -> str:
        """
        Convert ARPABET phoneme to IPA.

        Args:
            arpabet: ARPABET phoneme (may include stress digits)

        Returns:
            IPA phoneme string
        """
        # Remove stress markers (0, 1, 2)
        clean_arpabet = ''.join(c for c in arpabet if not c.isdigit())

        # Convert to IPA
        if clean_arpabet in self.ARPABET_TO_IPA:
            ipa = self.ARPABET_TO_IPA[clean_arpabet]

            # Add stress markers if present
            if '1' in arpabet:  # Primary stress
                ipa = 'ˈ' + ipa
            elif '2' in arpabet:  # Secondary stress
                ipa = 'ˌ' + ipa

            return ipa
        else:
            # Return as-is if no mapping (e.g., punctuation, space)
            return arpabet

    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to IPA phonemes.

        Args:
            text: Input text string

        Returns:
            List of IPA phoneme strings
        """
        # Get ARPABET phonemes from g2p-en
        arpabet_phonemes = self.g2p(text)

        # Convert each ARPABET phoneme to IPA
        ipa_phonemes = []
        for arp in arpabet_phonemes:
            if arp == ' ':
                # Keep spaces
                ipa_phonemes.append(' ')
            elif len(arp) == 1 and not arp.isalpha():
                # Keep punctuation
                ipa_phonemes.append(arp)
            else:
                # Convert ARPABET to IPA
                ipa = self._arpabet_to_ipa(arp)

                # Split diphthongs/complex phonemes into individual characters
                # (Kokoro vocab has individual IPA symbols)
                for char in ipa:
                    if char in self.phoneme_to_id:
                        ipa_phonemes.append(char)
                    elif char == 'ˈ' or char == 'ˌ':
                        # Stress markers
                        ipa_phonemes.append(char)
                    else:
                        # Unknown character - try to find closest match
                        # or skip
                        pass

        return ipa_phonemes

    def encode(self, phonemes_or_text) -> List[int]:
        """
        Convert phonemes or text to token IDs.

        Args:
            phonemes_or_text: Either a list of IPA phoneme strings OR text string

        Returns:
            List of token IDs
        """
        # Check if input is a list (phonemes) or string (text)
        if isinstance(phonemes_or_text, list):
            # Input is already phonemes
            phonemes = phonemes_or_text
        else:
            # Input is text - convert to phonemes first
            phonemes = self.text_to_phonemes(phonemes_or_text)

        # Convert to token IDs
        token_ids = []
        for phoneme in phonemes:
            if phoneme in self.phoneme_to_id:
                token_ids.append(self.phoneme_to_id[phoneme])
            else:
                # Unknown phoneme - use space token as fallback
                print(f"Warning: Unknown phoneme '{phoneme}' (U+{ord(phoneme):04X})")
                token_ids.append(self.space_token)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text (phonemes).

        Args:
            token_ids: List of token IDs

        Returns:
            Phoneme string
        """
        phonemes = []
        for token_id in token_ids:
            if token_id in self.id_to_phoneme:
                phonemes.append(self.id_to_phoneme[token_id])
            elif token_id != self.pad_token:
                phonemes.append(f"<{token_id}>")

        return ''.join(phonemes)


def test_phonemizer():
    """Test the phonemizer."""
    print("=" * 70)
    print("TESTING KOKORO PHONEMIZER")
    print("=" * 70)

    # Initialize
    phonemizer = KokoroPhonemizer()

    # Test texts
    test_texts = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "How are you doing today?",
        "This is a test of the phonemization system.",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: \"{text}\"")

        # Convert to phonemes
        phonemes = phonemizer.text_to_phonemes(text)
        print(f"   Phonemes: {phonemes}")
        print(f"   Phoneme count: {len(phonemes)}")

        # Encode to token IDs
        token_ids = phonemizer.encode(text)
        print(f"   Token IDs: {token_ids}")
        print(f"   Token count: {len(token_ids)}")

        # Decode back
        decoded = phonemizer.decode(token_ids)
        print(f"   Decoded: \"{decoded}\"")

    print("\n" + "=" * 70)
    print("PHONEMIZER TEST COMPLETE")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Integrate into kokoro_xdna2_runtime.py")
    print("  2. Replace ASCII character mapping with phonemizer")
    print("  3. Test with real TTS synthesis")
    print("  4. Compare audio quality before/after")


if __name__ == "__main__":
    import os
    os.chdir('/home/ccadmin/CC-1L/npu-services/unicorn-orator/xdna2')
    test_phonemizer()
