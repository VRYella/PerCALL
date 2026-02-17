"""
Utils State Module - Application State Management
==================================================

This module provides state management utilities for Streamlit applications,
including session state initialization and theme management.

Key Components:
    - Session state initialization with default values
    - Theme mode management (light/dark)
    - Table density settings
    - Color theme configurations

Module Integration:
    - Used by app.py for Streamlit session state
    - Provides consistent state management patterns
    - Handles user preferences and UI settings
"""

from typing import Dict, Any

__all__ = [
    'DEFAULT_SESSION_STATE',
    'initialize_session_state',
    'get_state',
    'set_state',
]

# Default session state values
DEFAULT_SESSION_STATE = {
    'theme_mode': 'light',
    'table_density': 'relaxed',
    'color_theme': 'scientific_blue',
    'seqs': [],
    'names': [],
    'results': [],
}


def initialize_session_state(st_session_state: Dict[str, Any]) -> None:
    """
    Initialize Streamlit session state with default values.
    
    This function ensures all required state variables are present
    in the session state, initializing them with default values if missing.
    
    Args:
        st_session_state: Streamlit session_state object
    
    Example:
        >>> import streamlit as st
        >>> from utils.state import initialize_session_state
        >>> initialize_session_state(st.session_state)
    """
    for key, default_value in DEFAULT_SESSION_STATE.items():
        if key not in st_session_state:
            st_session_state[key] = default_value


def get_state(st_session_state: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get a value from session state with a default fallback.
    
    Args:
        st_session_state: Streamlit session_state object
        key: State key to retrieve
        default: Default value if key not found
    
    Returns:
        Value from session state or default
    
    Example:
        >>> theme = get_state(st.session_state, 'theme_mode', 'light')
    """
    return st_session_state.get(key, default)


def set_state(st_session_state: Dict[str, Any], key: str, value: Any) -> None:
    """
    Set a value in session state.
    
    Args:
        st_session_state: Streamlit session_state object
        key: State key to set
        value: Value to store
    
    Example:
        >>> set_state(st.session_state, 'theme_mode', 'dark')
    """
    st_session_state[key] = value
