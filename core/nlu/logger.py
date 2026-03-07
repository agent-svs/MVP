"""
NLU Dataset Logger
Logs all NLU turn results to a JSONL file for offline analysis and threshold tuning.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from config import cfg

logger = logging.getLogger(__name__)

# Ensure logs directory exists
_LOGS_DIR = cfg.project_root / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
_NLU_LOG_FILE = _LOGS_DIR / "nlu_data.jsonl"


class NLULogger:
    """Handles structured logging of NLU interactions."""

    def __init__(self, log_file: Path = _NLU_LOG_FILE):
        self.log_file = log_file

    def log_turn(self, turn_data: Dict[str, Any]) -> None:
        """
        Log a single NLU turn result.
        
        Expected structure in turn_data (from DST result + router):
        - text: User utterance
        - intent: Classified intent
        - confidence: Overall NLU confidence
        - slots: Extracted slots
        - state: Current DST state
        - route: Selected action path (e.g., 'A', 'B', 'C', 'clarify')
        """
        try:
            # Add timestamp
            record = {
                "timestamp": datetime.now().isoformat(),
                **turn_data
            }
            
            # Append to JSONL
            with open(self.log_file, "a", encoding="utf-8") as f:
                json.dump(record, f)
                f.write("\n")
                
        except Exception as e:
            logger.error(f"Failed to log NLU turn: {e}")

# Global instance
nlu_logger = NLULogger()
