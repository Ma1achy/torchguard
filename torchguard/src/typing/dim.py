from __future__ import annotations
from typing import Any, List

__all__ = ['Dim', 'Broadcast']


class _AttributeRef:
    """Represents a reference to an instance attribute for dimension specs."""
    
    def __init__(self, obj_name: str, attr: str) -> None:
        """
        Initialize attribute reference.

        Args:
            obj_name (str): Object name (typically "self")
            attr (str): Attribute name to reference
        """
        self.obj_name = obj_name
        self.attr = attr
    
    def __getattr__(self, name: str) -> _AttributeRef:
        """
        Support chained access: Dim.config.hidden_size.

        Args:
            name (str): Next attribute in the chain

        Returns:
            (_AttributeRef): New attribute reference with chained path
        """
        return _AttributeRef(f"{self.obj_name}.{self.attr}", name)
    
    def __parse_attribute_path(self) -> List[str]:
        """
        Parse the full attribute path into a list of attribute names.

        Returns:
            (List[str]): List of attribute names to navigate
        """
        if '.' in self.obj_name:
            path_parts = self.obj_name.split('.')[1:]  # Remove 'self'
            path_parts.append(self.attr)
        else:
            path_parts = [self.attr]
        return path_parts
    
    def __navigate_attribute_path(self, instance: Any, path_parts: List[str]) -> Any:
        """
        Navigate the attribute path on the instance object.

        Args:
            instance (Any): Instance object to navigate attributes on
            path_parts (List[str]): List of attribute names to navigate

        Returns:
            (Any): Final resolved value from the attribute path

        Raises:
            AttributeError: If any attribute in the path doesn't exist
        """
        value = instance
        for part in path_parts:
            if not hasattr(value, part):
                raise AttributeError(
                    f"Instance {type(instance).__name__} has no attribute '{part}' "
                    f"while resolving '{self}'"
                )
            value = getattr(value, part)
        return value
    
    def __validate_resolved_value(self, value: Any) -> None:
        """
        Validate that the resolved value is an integer.

        Args:
            value (Any): Resolved value to validate

        Raises:
            TypeError: If value is not an integer
        """
        if not isinstance(value, int):
            raise TypeError(
                f"Dimension '{self}' resolved to {type(value).__name__}, expected int. "
                f"Got value: {value}"
            )
    
    def __check_instance_not_none(self, instance: Any) -> None:
        """
        Check that instance is not None.

        Args:
            instance (Any): Instance object to check

        Raises:
            ValueError: If instance is None
        """
        if instance is None:
            raise ValueError(
                f"Cannot resolve '{self}' without instance. "
                f"Ensure @tensorcheck is used on instance methods."
            )
    
    def resolve(self, instance: Any) -> int:
        """
        Resolve attribute path to actual integer value.

        Args:
            instance (Any): Instance object to resolve attributes from

        Returns:
            (int): Resolved integer value from the attribute path

        Raises:
            ValueError: If instance is None
            AttributeError: If any attribute in the path doesn't exist
            TypeError: If resolved value is not an integer
        """
        self.__check_instance_not_none(instance)
        path_parts = self.__parse_attribute_path()
        value = self.__navigate_attribute_path(instance, path_parts)
        self.__validate_resolved_value(value)
        return value
    
    def __repr__(self) -> str:
        """
        Return string representation of the attribute reference.

        Returns:
            (str): String representation in format "obj_name.attr"
        """
        return f"{self.obj_name}.{self.attr}"
    
    def __str__(self) -> str:
        """
        Return string representation of the attribute reference.

        Returns:
            (str): String representation (same as __repr__)
        """
        return repr(self)
    
    def __eq__(self, other: Any) -> bool:
        """
        Check equality with another attribute reference.

        Args:
            other (Any): Other object to compare with

        Returns:
            (bool): True if other is _AttributeRef with same obj_name and attr
        """
        if isinstance(other, _AttributeRef):
            return self.obj_name == other.obj_name and self.attr == other.attr
        return False
    
    def __hash__(self) -> int:
        """
        Return hash of the attribute reference.

        Returns:
            (int): Hash value based on obj_name and attr
        """
        return hash((self.obj_name, self.attr))


class _DimProxy:
    """Proxy object for capturing attribute access as dimension references."""
    
    def __getattr__(self, name: str) -> _AttributeRef:
        """
        Capture attribute access: Dim.in_channels -> _AttributeRef('self', 'in_channels').

        Args:
            name (str): Attribute name to create reference for

        Returns:
            (_AttributeRef): Attribute reference for the given name
        """
        return _AttributeRef("self", name)
    
    def __repr__(self) -> str:
        """
        Return string representation of the Dim proxy.

        Returns:
            (str): String "Dim"
        """
        return "Dim"


class _BroadcastMarker:
    """Marker for broadcast-compatible dimensions."""
    
    def __repr__(self) -> str:
        """
        Return string representation of the broadcast marker.

        Returns:
            (str): String "*"
        """
        return "*"
    
    def __eq__(self, other: Any) -> bool:
        """
        Check equality with another broadcast marker.

        Args:
            other (Any): Other object to compare with

        Returns:
            (bool): True if other is _BroadcastMarker
        """
        return isinstance(other, _BroadcastMarker)
    
    def __hash__(self) -> int:
        """
        Return hash of the broadcast marker.

        Returns:
            (int): Hash value for "*"
        """
        return hash("*")


Dim = _DimProxy()
Broadcast = _BroadcastMarker()

