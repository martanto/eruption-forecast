# Standard library imports
import functools
import inspect
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

# Project imports
from eruption_forecast.decorators.decorator_class import (
    AutoSaveDict,
    SerializationWrapper,
)


def save_parameters(
    filepath: str | None = None,
    save_as: Literal["json", "yaml"] = "json",
    include_self: bool = False,
    append_timestamp: bool = False,
):
    """
    Decorator to save method parameters to a file

    Args:
        filepath: Path to save file (default: method_name_params.json)
        save_as: 'json' or 'yaml'
        include_self: Whether to include 'self' parameter for instance methods
        append_timestamp: Whether to append timestamp to filename

    Example:
        @save_parameters('my_params.json', format='json')
        def my_function(x, y, z=10):
            return x + y + z
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Prepare data to save
            params_dict = dict(bound_args.arguments)

            # Remove 'self' or 'cls' if not requested
            if not include_self:
                params_dict.pop("self", None)
                params_dict.pop("cls", None)

            data = {
                "function": func.__name__,
                "timestamp": datetime.now().isoformat(),
                "parameters": params_dict,
            }

            # Determine filepath
            if filepath:
                output_path = filepath
            else:
                output_path = f"{func.__name__}_params.{format}"

            if append_timestamp:
                path = Path(output_path)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{path.stem}_{timestamp}{path.suffix}"

            # Save to file
            SerializationWrapper.save_to_file(data, output_path, save_as)

            # Execute original function
            return func(*args, **kwargs)

        return wrapper

    return decorator


def save_properties(
    filepath: str | None = None,
    save_as: Literal["json", "yaml"] = "json",
    include_private: bool = False,
    properties: list | None = None,
):
    """
    Class decorator to add methods for saving class properties

    Args:
        filepath: Default path to save file
        save_as: 'json' or 'yaml'
        include_private: Whether to include private attributes (starting with _)
        properties: Specific properties to save (None = all public properties)

    Example:
        @save_properties('my_class.json', save_as='json')
        class MyClass:
            def __init__(self, x, y):
                self.x = x
                self.y = y
    """

    def decorator(cls):
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._save_format = save_as
            self._save_filepath = filepath or f"{cls.__name__}_properties.{save_as}"

        def save_properties_method(
            self,
            output_path: str | None = None,
            output_format: Literal["json", "yaml"] | None = None,
        ):
            """Save current properties to file"""
            path = output_path or self._save_filepath
            fmt = output_format or self._save_format

            # Collect properties
            props = {}
            for key, value in self.__dict__.items():
                # Skip internal save configuration
                if key.startswith("_save_"):
                    continue

                # Handle private attributes
                if key.startswith("_") and not include_private:
                    continue

                # Handle specific properties filter
                if properties is not None and key not in properties:
                    continue

                props[key] = value

            data = {
                "class": cls.__name__,
                "timestamp": datetime.now().isoformat(),
                "properties": props,
            }

            SerializationWrapper.save_to_file(data, path, fmt)
            return path

        def save_on_change(self, attr_name: str):
            """Enable auto-save when specific attribute changes"""
            original_value = getattr(self, attr_name)

            def setter(new_value):
                setattr(self, f"_{attr_name}_internal", new_value)
                self.save_properties()

            def getter():
                return getattr(self, f"_{attr_name}_internal", original_value)

            setattr(self, f"_{attr_name}_internal", original_value)
            setattr(cls, attr_name, property(getter, setter))  # type: ignore

        # Add methods to class
        cls.__init__ = new_init
        cls.save_properties = save_properties_method
        cls.enable_auto_save = save_on_change

        return cls

    return decorator


def snapshot(filepath: str | None = None, save_as: Literal["json", "yaml"] = "json"):
    """
    Decorator to save method parameters and return value

    Example:
        @snapshot('calculation.json')
        def calculate(x, y):
            return x * y
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Capture parameters
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            params_dict = dict(bound_args.arguments)
            params_dict.pop("self", None)
            params_dict.pop("cls", None)

            # Execute function
            result = func(*args, **kwargs)

            # Save snapshot
            data = {
                "function": func.__name__,
                "timestamp": datetime.now().isoformat(),
                "parameters": params_dict,
                "return_value": result,
            }

            output_path = filepath or f"{func.__name__}_snapshot.{save_as}"
            SerializationWrapper.save_to_file(data, output_path, save_as)

            return result

        return wrapper

    return decorator


def timer(name: str | None = None, verbose: bool = True):
    """
    Decorator factory for timing functions.

    Args:
        name: Custom name for the operation (defaults to function name)
        verbose: Whether to print timing results

    Returns:
        Decorated function

    Example:
        @timer()
        def my_function():
            time.sleep(1)
    """

    def decorator(func: Callable) -> Callable:
        operation_name = name if name else func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed = timedelta(seconds=end - start).seconds
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            if verbose:
                print("==" * 50)
                print(
                    f"|| {operation_name}: took {hours:02d} hours, {minutes:02d} minutes, {seconds:02d} seconds"
                )
                print("==" * 50)
            return result

        return wrapper

    return decorator
