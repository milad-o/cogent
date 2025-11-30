"""
Tests for agent reasoning module.
"""

import pytest
from agenticflow.agent.reasoning import (
    ReasoningConfig,
    ReasoningStyle,
    ThinkingStep,
    ReasoningResult,
    build_reasoning_prompt,
    extract_thinking,
    estimate_confidence,
)


class TestReasoningStyle:
    """Tests for ReasoningStyle enum."""
    
    def test_analytical_value(self):
        assert ReasoningStyle.ANALYTICAL.value == "analytical"
    
    def test_exploratory_value(self):
        assert ReasoningStyle.EXPLORATORY.value == "exploratory"
    
    def test_critical_value(self):
        assert ReasoningStyle.CRITICAL.value == "critical"
    
    def test_creative_value(self):
        assert ReasoningStyle.CREATIVE.value == "creative"


class TestReasoningConfig:
    """Tests for ReasoningConfig dataclass."""
    
    def test_default_config(self):
        config = ReasoningConfig()
        assert config.enabled is True
        assert config.max_thinking_rounds == 3
        assert config.thinking_budget is None
        assert config.show_thinking is False
        assert config.style == ReasoningStyle.ANALYTICAL
        assert config.require_confidence is None
        assert config.self_correct is True
    
    def test_quick_preset(self):
        config = ReasoningConfig.quick()
        assert config.max_thinking_rounds == 1
        assert config.self_correct is False
    
    def test_standard_preset(self):
        config = ReasoningConfig.standard()
        assert config.max_thinking_rounds == 3
        assert config.style == ReasoningStyle.ANALYTICAL
        assert config.self_correct is True
    
    def test_deep_preset(self):
        config = ReasoningConfig.deep()
        assert config.max_thinking_rounds == 5
        assert config.style == ReasoningStyle.EXPLORATORY
        assert config.show_thinking is True
        assert config.require_confidence == 0.7
    
    def test_critical_preset(self):
        config = ReasoningConfig.critical()
        assert config.max_thinking_rounds == 4
        assert config.style == ReasoningStyle.CRITICAL
        assert config.show_thinking is True
    
    def test_validation_min_rounds(self):
        with pytest.raises(ValueError, match="max_thinking_rounds must be at least 1"):
            ReasoningConfig(max_thinking_rounds=0)
    
    def test_validation_max_rounds(self):
        with pytest.raises(ValueError, match="max_thinking_rounds cannot exceed 10"):
            ReasoningConfig(max_thinking_rounds=11)
    
    def test_validation_confidence_low(self):
        with pytest.raises(ValueError, match="require_confidence must be between"):
            ReasoningConfig(require_confidence=-0.1)
    
    def test_validation_confidence_high(self):
        with pytest.raises(ValueError, match="require_confidence must be between"):
            ReasoningConfig(require_confidence=1.5)
    
    def test_valid_confidence(self):
        config = ReasoningConfig(require_confidence=0.8)
        assert config.require_confidence == 0.8


class TestThinkingStep:
    """Tests for ThinkingStep dataclass."""
    
    def test_create_step(self):
        step = ThinkingStep(
            round=1,
            thought="Analyzing the problem...",
            reasoning_type="analysis",
            confidence=0.8,
        )
        assert step.round == 1
        assert step.thought == "Analyzing the problem..."
        assert step.reasoning_type == "analysis"
        assert step.confidence == 0.8
    
    def test_to_dict(self):
        step = ThinkingStep(
            round=2,
            thought="Breaking it down...",
            reasoning_type="plan",
            confidence=0.7,
        )
        d = step.to_dict()
        assert d["round"] == 2
        assert d["thought"] == "Breaking it down..."
        assert d["reasoning_type"] == "plan"
        assert d["confidence"] == 0.7
    
    def test_optional_confidence(self):
        step = ThinkingStep(
            round=1,
            thought="Thinking...",
            reasoning_type="analysis",
        )
        assert step.confidence is None


class TestReasoningResult:
    """Tests for ReasoningResult dataclass."""
    
    def test_empty_result(self):
        result = ReasoningResult()
        assert result.thinking_steps == []
        assert result.final_plan is None
        assert result.total_thinking_tokens == 0
        assert result.thinking_rounds == 0
    
    def test_thinking_summary_empty(self):
        result = ReasoningResult()
        assert result.thinking_summary == ""
    
    def test_thinking_summary_with_steps(self):
        result = ReasoningResult(
            thinking_steps=[
                ThinkingStep(1, "First thought about the problem...", "analysis"),
                ThinkingStep(2, "Second thought refining approach...", "refinement"),
            ]
        )
        summary = result.thinking_summary
        assert "[Round 1]" in summary
        assert "[Round 2]" in summary
        assert "analysis" in summary
        assert "refinement" in summary
    
    def test_to_dict(self):
        result = ReasoningResult(
            thinking_steps=[
                ThinkingStep(1, "Thought", "analysis", 0.8),
            ],
            final_plan="The plan is...",
            total_thinking_tokens=100,
            thinking_rounds=1,
        )
        d = result.to_dict()
        assert len(d["thinking_steps"]) == 1
        assert d["final_plan"] == "The plan is..."
        assert d["total_thinking_tokens"] == 100
        assert d["thinking_rounds"] == 1


class TestExtractThinking:
    """Tests for extract_thinking function."""
    
    def test_extract_single_block(self):
        response = "Some text <thinking>My thoughts here</thinking> more text"
        thinking, cleaned = extract_thinking(response)
        assert thinking == "My thoughts here"
        assert "<thinking>" not in cleaned
        assert "My thoughts here" not in cleaned
        assert "Some text" in cleaned
        assert "more text" in cleaned
    
    def test_extract_multiple_blocks(self):
        response = "<thinking>First</thinking> middle <thinking>Second</thinking>"
        thinking, cleaned = extract_thinking(response)
        assert "First" in thinking
        assert "Second" in thinking
        assert "middle" in cleaned
    
    def test_no_thinking_block(self):
        response = "Just a regular response without thinking"
        thinking, cleaned = extract_thinking(response)
        assert thinking is None
        assert cleaned == response
    
    def test_multiline_thinking(self):
        response = "<thinking>\nLine 1\nLine 2\nLine 3\n</thinking>"
        thinking, cleaned = extract_thinking(response)
        assert "Line 1" in thinking
        assert "Line 2" in thinking
        assert "Line 3" in thinking


class TestEstimateConfidence:
    """Tests for estimate_confidence function."""
    
    def test_high_confidence_phrases(self):
        thinking = "I'm confident this will work. The solution is clearly correct."
        confidence = estimate_confidence(thinking)
        assert confidence > 0.6  # Above baseline
    
    def test_low_confidence_phrases(self):
        thinking = "I'm not sure about this. It might work, but it's unclear."
        confidence = estimate_confidence(thinking)
        assert confidence < 0.6  # Below baseline
    
    def test_neutral_text(self):
        thinking = "The algorithm processes data in sequential order."
        confidence = estimate_confidence(thinking)
        assert 0.5 <= confidence <= 0.7  # Near baseline
    
    def test_confidence_clamped_high(self):
        thinking = "definitely confident clearly obvious straightforward simple"
        confidence = estimate_confidence(thinking)
        assert confidence <= 0.95  # Clamped
    
    def test_confidence_clamped_low(self):
        thinking = "not sure uncertain unclear might maybe could be risky"
        confidence = estimate_confidence(thinking)
        assert confidence >= 0.1  # Clamped


class TestBuildReasoningPrompt:
    """Tests for build_reasoning_prompt function."""
    
    def test_basic_prompt(self):
        config = ReasoningConfig()
        prompt = build_reasoning_prompt(config, "Analyze this data")
        assert "Task: Analyze this data" in prompt
        assert "think through this carefully" in prompt
    
    def test_prompt_with_context(self):
        config = ReasoningConfig()
        context = {"key": "value", "data": "info"}
        prompt = build_reasoning_prompt(config, "Do something", context)
        assert "Context:" in prompt
        assert "key: value" in prompt
        assert "data: info" in prompt
    
    def test_analytical_style(self):
        config = ReasoningConfig(style=ReasoningStyle.ANALYTICAL)
        prompt = build_reasoning_prompt(config, "Task")
        # Just check it doesn't crash with analytical style
        assert "Task:" in prompt
    
    def test_exploratory_style(self):
        config = ReasoningConfig(style=ReasoningStyle.EXPLORATORY)
        prompt = build_reasoning_prompt(config, "Task")
        assert "Task:" in prompt
