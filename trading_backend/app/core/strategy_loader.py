# app/core/strategy_loader.py
import os
import importlib.util
import inspect
import logging
from typing import Dict, Type, List
from typing import List, Optional, Dict, Any, Literal, Union # Added Union

# Assuming BaseStrategy is in app/strategies/base_strategy.py
# Adjust path if necessary.
from ..strategies.base_strategy import BaseStrategy # Relative import

# Configure logging
logger = logging.getLogger(__name__)

# Path to the strategies directory
# Assumes this file (strategy_loader.py) is in app/core/
# and strategies are in app/strategies/
STRATEGIES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "strategies")

# Global cache for loaded strategies to avoid reloading on every call
_LOADED_STRATEGIES: Dict[str, Type[BaseStrategy]] = {}
_STRATEGIES_LOADED_FLAG = False

def load_strategies() -> Dict[str, Type[BaseStrategy]]:
    """
    Scans the STRATEGIES_DIR, imports Python modules, and discovers classes
    that inherit from BaseStrategy.
    Returns a dictionary mapping strategy_id to strategy class.
    """
    global _LOADED_STRATEGIES, _STRATEGIES_LOADED_FLAG
    if _STRATEGIES_LOADED_FLAG:
        return _LOADED_STRATEGIES

    logger.info(f"Scanning for strategies in: {STRATEGIES_DIR}")
    if not os.path.isdir(STRATEGIES_DIR):
        logger.error(f"Strategies directory not found: {STRATEGIES_DIR}")
        return {}

    for filename in os.listdir(STRATEGIES_DIR):
        if filename.endswith(".py") and filename != "__init__.py" and filename != "base_strategy.py":
            module_name = filename[:-3] # Remove .py
            file_path = os.path.join(STRATEGIES_DIR, filename)
            
            try:
                # Dynamically import the module
                spec = importlib.util.spec_from_file_location(f"app.strategies.{module_name}", file_path)
                if spec is None or spec.loader is None:
                    logger.warning(f"Could not create module spec for {file_path}")
                    continue
                
                strategy_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(strategy_module)
                
                # Inspect the module for classes inheriting from BaseStrategy
                for name, obj in inspect.getmembers(strategy_module):
                    if inspect.isclass(obj) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                        try:
                            strategy_id = obj.strategy_id # Access class variable
                            if strategy_id in _LOADED_STRATEGIES:
                                logger.warning(f"Duplicate strategy_id '{strategy_id}' found in {filename}. Overwriting.")
                            _LOADED_STRATEGIES[strategy_id] = obj
                            logger.info(f"Successfully loaded strategy: {strategy_id} from {filename}")
                        except AttributeError:
                            logger.warning(f"Class {name} in {filename} inherits BaseStrategy but lacks 'strategy_id' attribute.")
                        except Exception as e_inner:
                             logger.error(f"Error processing strategy class {name} in {filename}: {e_inner}")

            except ImportError as e_import:
                logger.error(f"Could not import strategy module {module_name} from {file_path}: {e_import}")
            except Exception as e_general:
                logger.error(f"An unexpected error occurred while loading strategy from {file_path}: {e_general}", exc_info=True)
                
    _STRATEGIES_LOADED_FLAG = True
    if not _LOADED_STRATEGIES:
        logger.warning("No strategies were loaded.")
    else:
        logger.info(f"Finished loading strategies. Total loaded: {len(_LOADED_STRATEGIES)}")
    return _LOADED_STRATEGIES

def get_strategy_class(strategy_id: str) -> Optional[Type[BaseStrategy]]:
    """Returns the strategy class for a given strategy_id, loading if necessary."""
    if not _STRATEGIES_LOADED_FLAG:
        load_strategies()
    return _LOADED_STRATEGIES.get(strategy_id)

def get_available_strategies_info() -> List[Dict]: # Should return List[models.StrategyInfo]
    """Returns a list of StrategyInfo objects for all loaded strategies."""
    if not _STRATEGIES_LOADED_FLAG:
        load_strategies()
    
    info_list = []
    for strategy_id, strategy_class in _LOADED_STRATEGIES.items():
        try:
            # Assuming get_info() is a classmethod that returns a Pydantic model or dict
            info = strategy_class.get_info() 
            # If info is a Pydantic model, you might want to convert to dict for some use cases:
            # info_list.append(info.model_dump() if hasattr(info, 'model_dump') else info)
            info_list.append(info) # Assuming get_info() returns a dict or Pydantic model
        except Exception as e:
            logger.error(f"Could not get info for strategy {strategy_id}: {e}")
    return info_list

# Example of how to call it at startup (e.g., in app/main.py)
# if __name__ == "__main__":
#     # Configure basic logging for testing this module directly
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
#     loaded_strategies = load_strategies()
#     print("\nLoaded Strategies:")
#     for sid, s_class in loaded_strategies.items():
#         print(f"  ID: {sid}, Class: {s_class.__name__}")
#         try:
#             print(f"    Info: {s_class.get_info()}")
#         except:
#             pass
#     print("\nAvailable Strategies Info:")
#     for info_item in get_available_strategies_info():
#         print(f"  {info_item}")
