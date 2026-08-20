class AlignmentError(RuntimeError):
    """Base error for invalid or unalignable experiment input."""


class ConfigurationError(AlignmentError):
    """Raised when project configuration is invalid."""


class MarkerParseError(AlignmentError):
    """Raised when a marker text file cannot be parsed safely."""


class StimulusMatchError(AlignmentError):
    """Raised when formal Train/Test trigger blocks are not unique."""


class SourceFormatError(AlignmentError):
    """Raised when a read-only source file has an unsupported format."""
