import logging
from flask import jsonify

logger = logging.getLogger(__name__)

def api_success_response(data=None, message="Success", status=200):
    """
    Standard API success response.
    """
    response = {
        "success": True,
        "message": message,
        "data": data
    }
    return jsonify(response), status

def api_error_response(error=None, message="An error occurred", status=400):
    """
    Standard API error response.
    """
    logger.error(f"API error: {message} | Details: {error}")
    response = {
        "success": False,
        "message": message,
        "error": str(error) if error else None
    }
    return jsonify(response), status

def handle_api_exception(e):
    """
    Flask error handler for API exceptions.
    """
    logger.exception("Unhandled API exception")
    return api_error_response(error=e, message="Internal server error", status=500)
