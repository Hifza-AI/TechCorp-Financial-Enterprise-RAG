"""Tests STAGE_1 ke modules ko seedha import karte hain (project flat scripts hai, package nahi)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "STAGE_1"))