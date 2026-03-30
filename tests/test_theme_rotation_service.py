# -*- coding: utf-8 -*-
"""Tests for theme rotation / relay candidate helpers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.theme_rotation_service import (
    analyze_theme_rotation,
    annotate_rotation_candidates,
)


class TestThemeRotationService(unittest.TestCase):
    def test_annotate_rotation_candidates_prefers_pullback_relay(self) -> None:
        candidates = [
            {
                "code": "002361",
                "name": "神剑股份",
                "price": 7.77,
                "change_pct": 5.6,
                "main_net": 4200,
                "support": 7.06,
                "bias5": 2.6,
                "ma_trend": "多头排列",
                "vol_pattern": "温和放量",
                "tech_score": 92,
            },
            {
                "code": "600999",
                "name": "副龙候选",
                "price": 7.10,
                "change_pct": 0.8,
                "main_net": 3600,
                "support": 7.05,
                "bias5": 0.2,
                "ma_trend": "多头排列",
                "vol_pattern": "缩量回踩",
                "tech_score": 88,
                "candidate_source": "sub_dragon",
            },
        ]

        annotated = annotate_rotation_candidates(candidates, dominant_themes=["商业航天"], limit=2)

        self.assertEqual(annotated[0]["code"], "600999")
        self.assertEqual(annotated[0]["entry_state"], "pullback_ready")
        self.assertEqual(annotated[0]["relay_role"], "副龙/补涨候选")
        self.assertTrue(annotated[1]["missed_entry"])
        self.assertEqual(annotated[1]["entry_state"], "overextended")

    def test_analyze_theme_rotation_detects_active_switch(self) -> None:
        sectors = [
            {"name": "商业航天", "change_pct": 3.6, "main_net": 9800, "sector_type": "gn"},
            {"name": "军工", "change_pct": 2.8, "main_net": 7600, "sector_type": "gn"},
            {"name": "光伏", "change_pct": 1.0, "main_net": 2200, "sector_type": "hy"},
        ]
        candidates = [
            {
                "code": "002361",
                "name": "神剑股份",
                "price": 7.77,
                "change_pct": 9.1,
                "main_net": 5200,
                "support": 7.20,
                "ma_trend": "多头排列",
            },
            {
                "code": "300600",
                "name": "副龙一号",
                "price": 7.08,
                "change_pct": 1.2,
                "main_net": 3300,
                "support": 7.02,
                "ma_trend": "多头排列",
                "vol_pattern": "缩量回踩",
                "candidate_source": "sub_dragon",
            },
        ]

        result = analyze_theme_rotation(sectors, candidates)

        self.assertEqual(result["switch_signal"], "active")
        self.assertEqual(result["dominant_themes"][0], "商业航天")
        self.assertEqual(result["leader"]["code"], "002361")
        self.assertEqual(result["relay_candidates"][0]["code"], "300600")


if __name__ == "__main__":
    unittest.main()
