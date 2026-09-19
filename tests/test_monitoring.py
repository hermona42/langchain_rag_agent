from src.monitoring.cost_tracker import calculate_cost, CostTrackerCallback

def test_calculate_cost_gpt4o_mini():
    """
    Requirement 3: Cost tracking must accurately calculate model usage costs.
    """
    # gpt-4o-mini rates: $0.150 per 1M input tokens, $0.600 per 1M output tokens
    cost = calculate_cost(
        model_name="gpt-4o-mini",
        prompt_tokens=1000,
        completion_tokens=500
    )
    # Expected: (1000 * 0.00000015) + (500 * 0.0000006) = 0.00015 + 0.0003 = 0.00045
    assert round(cost, 6) == 0.00045

def test_cost_tracker_callback():
    """
    Requirement 3: LangChain callback handler tracks cumulative tokens and total cost.
    """
    tracker = CostTrackerCallback(model_name="gpt-4o-mini")
    
    # Simulate first LLM call run
    tracker.on_llm_end_tokens(prompt_tokens=2000, completion_tokens=1000)
    assert tracker.total_tokens == 3000
    assert round(tracker.total_cost_usd, 5) == 0.0009