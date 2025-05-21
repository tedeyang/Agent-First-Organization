from arklex.exceptions import ExceptionPrompt


class AcuityExceptionPrompt(ExceptionPrompt):
    """
    Acuity-specific exception prompts.
    """

    # get_available_dates exception prompt
    AVAILABLE_DATES_EXCEPTION_PROMPT = "Finding the available dates failed"

    # get_available_dates exception prompt
    AVAILABLE_TIMES_EXCEPTION_PROMPT = "Finding the available times failed"

    # get_session_types exception prompt
    AVAILABLE_TYPES_EXCEPTION_PROMPT = "Finding all types of info session failed"

    # book_info_session exception prompt
    BOOK_SESSION_EXCEPTION_PROMPT = "Failed to book the info session"

    # book_check exception prompt
    BOOK_CHECK_EXCEPTION_PROMPT = "No info session in the available info sessions"

    # get_apt_by_email exception first prompt
    GET_APT_BY_EMAIL_EXCEPTION_PROMPT_1 = "User didn't make any appointment"

    GET_APT_BY_EMAIL_EXCEPTION_PROMPT_2 = "Retrieving all appointments failed"

    # cancel prompt
    CANCEL_PROMPT = "Cancel the appointment failed"

    # get_type_id_by_apt_name prompt
    GET_TYPE_ID_PROMPT = "Retrieving type id failed"

    # reschedule prompt
    RESCHEDULE_PROMPT = "Rescheduling the appointment failed"
