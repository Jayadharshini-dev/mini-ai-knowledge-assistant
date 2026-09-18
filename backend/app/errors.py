from enum import Enum

from fastapi import HTTPException, status


class ErrorCode(str, Enum):
    EMPTY_QUESTION = "EMPTY_QUESTION"
    NO_DOCUMENTS = "NO_DOCUMENTS"
    INVALID_PDF = "INVALID_PDF"
    ENCRYPTED_PDF = "ENCRYPTED_PDF"
    NO_EXTRACTABLE_TEXT = "NO_EXTRACTABLE_TEXT"
    DUPLICATE_DOCUMENT = "DUPLICATE_DOCUMENT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INDEX_MODEL_MISMATCH = "INDEX_MODEL_MISMATCH"
    INDEX_DIMENSION_MISMATCH = "INDEX_DIMENSION_MISMATCH"
    INDEX_TYPE_MISMATCH = "INDEX_TYPE_MISMATCH"
    INDEX_METADATA_MISMATCH = "INDEX_METADATA_MISMATCH"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    PROVIDER_RATE_LIMITED = "PROVIDER_RATE_LIMITED"
    PROVIDER_AUTH = "PROVIDER_AUTH"


class AppException(HTTPException):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        retryable: bool = False,
    ):
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.message = message
        self.retryable = retryable

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
        }
