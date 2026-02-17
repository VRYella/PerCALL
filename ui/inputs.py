"""
UI Inputs Module - Input Widgets and Forms
==========================================

This module provides input components for the Streamlit UI, including
text inputs, file uploads, and form elements.

Key Components:
    - File upload widgets
    - Text area inputs for sequence data
    - Radio buttons and select boxes
    - Number inputs for parameters

Module Integration:
    - Used by app.py for sequence input
    - Provides standardized input widgets
    - Handles file uploads and validation
"""

import streamlit as st
from typing import Optional, List, Any, Tuple

__all__ = [
    'create_file_uploader',
    'create_text_area',
    'create_radio_selector',
    'create_number_input',
    'create_select_box',
]


def create_file_uploader(
    label: str = "Choose a file",
    accepted_types: Optional[List[str]] = None,
    help_text: Optional[str] = None,
    key: Optional[str] = None
) -> Any:
    """
    Create a file uploader widget.
    
    Args:
        label: Label for the file uploader
        accepted_types: List of accepted file extensions (e.g., ['fasta', 'fa', 'txt'])
        help_text: Optional help text
        key: Optional unique key for the widget
    
    Returns:
        Uploaded file object or None
    
    Example:
        >>> file = create_file_uploader(
        ...     "Upload FASTA file",
        ...     accepted_types=['fasta', 'fa', 'txt'],
        ...     help_text="Upload a FASTA formatted sequence file"
        ... )
    """
    return st.file_uploader(
        label=label,
        type=accepted_types,
        help=help_text,
        key=key
    )


def create_text_area(
    label: str = "Input",
    value: str = "",
    height: int = 300,
    placeholder: Optional[str] = None,
    help_text: Optional[str] = None,
    key: Optional[str] = None
) -> str:
    """
    Create a text area input widget.
    
    Args:
        label: Label for the text area
        value: Default value
        height: Height in pixels
        placeholder: Placeholder text
        help_text: Optional help text
        key: Optional unique key for the widget
    
    Returns:
        Text entered by the user
    
    Example:
        >>> sequence = create_text_area(
        ...     "Paste DNA Sequence",
        ...     placeholder=">seq1\\nATGCATGC",
        ...     height=200
        ... )
    """
    return st.text_area(
        label=label,
        value=value,
        height=height,
        placeholder=placeholder,
        help=help_text,
        key=key
    )


def create_radio_selector(
    label: str,
    options: List[str],
    default_index: int = 0,
    horizontal: bool = False,
    help_text: Optional[str] = None,
    key: Optional[str] = None
) -> str:
    """
    Create a radio button selector.
    
    Args:
        label: Label for the radio buttons
        options: List of option strings
        default_index: Index of default selection
        horizontal: Display horizontally if True
        help_text: Optional help text
        key: Optional unique key for the widget
    
    Returns:
        Selected option
    
    Example:
        >>> method = create_radio_selector(
        ...     "Input Method",
        ...     ["Upload File", "Paste Sequence", "Load Example"],
        ...     horizontal=True
        ... )
    """
    return st.radio(
        label=label,
        options=options,
        index=default_index,
        horizontal=horizontal,
        help=help_text,
        key=key
    )


def create_number_input(
    label: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    value: Optional[float] = None,
    step: Optional[float] = None,
    help_text: Optional[str] = None,
    key: Optional[str] = None
) -> float:
    """
    Create a number input widget.
    
    Args:
        label: Label for the number input
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        value: Default value
        step: Step size for increment/decrement
        help_text: Optional help text
        key: Optional unique key for the widget
    
    Returns:
        Number entered by the user
    
    Example:
        >>> max_records = create_number_input(
        ...     "Max Records",
        ...     min_value=1,
        ...     max_value=100,
        ...     value=10,
        ...     step=1
        ... )
    """
    return st.number_input(
        label=label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        help=help_text,
        key=key
    )


def create_select_box(
    label: str,
    options: List[Any],
    default_index: int = 0,
    help_text: Optional[str] = None,
    key: Optional[str] = None
) -> Any:
    """
    Create a select box (dropdown) widget.
    
    Args:
        label: Label for the select box
        options: List of options
        default_index: Index of default selection
        help_text: Optional help text
        key: Optional unique key for the widget
    
    Returns:
        Selected option
    
    Example:
        >>> theme = create_select_box(
        ...     "Color Theme",
        ...     ["Light", "Dark", "Auto"],
        ...     default_index=0
        ... )
    """
    return st.selectbox(
        label=label,
        options=options,
        index=default_index,
        help=help_text,
        key=key
    )


def create_text_input(
    label: str,
    value: str = "",
    max_chars: Optional[int] = None,
    placeholder: Optional[str] = None,
    help_text: Optional[str] = None,
    key: Optional[str] = None
) -> str:
    """
    Create a single-line text input widget.
    
    Args:
        label: Label for the text input
        value: Default value
        max_chars: Maximum number of characters
        placeholder: Placeholder text
        help_text: Optional help text
        key: Optional unique key for the widget
    
    Returns:
        Text entered by the user
    
    Example:
        >>> query = create_text_input(
        ...     "Search Query",
        ...     placeholder="Enter gene name or accession",
        ...     max_chars=100
        ... )
    """
    return st.text_input(
        label=label,
        value=value,
        max_chars=max_chars,
        placeholder=placeholder,
        help=help_text,
        key=key
    )


def create_multiselect(
    label: str,
    options: List[Any],
    default: Optional[List[Any]] = None,
    help_text: Optional[str] = None,
    key: Optional[str] = None
) -> List[Any]:
    """
    Create a multiselect widget.
    
    Args:
        label: Label for the multiselect
        options: List of options
        default: List of default selections
        help_text: Optional help text
        key: Optional unique key for the widget
    
    Returns:
        List of selected options
    
    Example:
        >>> classes = create_multiselect(
        ...     "Select Motif Classes",
        ...     ["G-Quadruplex", "Z-DNA", "Cruciform"],
        ...     default=["G-Quadruplex"]
        ... )
    """
    return st.multiselect(
        label=label,
        options=options,
        default=default,
        help=help_text,
        key=key
    )


def create_button(
    label: str,
    button_type: str = "primary",
    use_container_width: bool = False,
    disabled: bool = False,
    help_text: Optional[str] = None,
    key: Optional[str] = None
) -> bool:
    """
    Create a button widget.
    
    Args:
        label: Button label
        button_type: Type of button ('primary' or 'secondary')
        use_container_width: Use full container width if True
        disabled: Disable button if True
        help_text: Optional help text
        key: Optional unique key for the widget
    
    Returns:
        True if button was clicked, False otherwise
    
    Example:
        >>> if create_button("Analyze", button_type="primary", use_container_width=True):
        ...     # Run analysis
        ...     pass
    """
    return st.button(
        label=label,
        type=button_type,
        use_container_width=use_container_width,
        disabled=disabled,
        help=help_text,
        key=key
    )
