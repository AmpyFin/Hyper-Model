# this is an interface for a strategy agent

# each strategy agent must implement the following methods:
# get_strategy_name() -> str - we will always use snake case for strategy names
# run_strategy(pd.dataframe, current_price) -> float value between -1 and 1 with 1 being the most bullish and -1 being the most bearish
# get_ideal_period() -> str - we will 