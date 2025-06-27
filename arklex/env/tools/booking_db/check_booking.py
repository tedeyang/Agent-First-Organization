from typing import Any

import pandas as pd

from arklex.utils.logging_utils import LogContext

from ..tools import register_tool
from .utils import *

log_context = LogContext(__name__)


@register_tool(
    "Checks details of booked show(s)",
    [],
    [
        {
            "name": "query_result",
            "type": "str",
            "description": "A list of booked shows. If no booking exists, returns 'No bookings found.'",
        }
    ],
    lambda x: x and x not in (LOG_IN_FAILURE, "No bookings found."),
)
def check_booking() -> str | None:
    if not log_in():
        return LOG_IN_FAILURE

    log_context.info("Enter check booking function")
    conn: sqlite3.Connection = sqlite3.connect(booking.db_path)
    cursor: sqlite3.Cursor = conn.cursor()

    query: str = """
    SELECT * FROM
        booking b
        JOIN show s ON b.show_id = s.id
    WHERE
        b.user_id = ?
    """
    cursor.execute(query, (booking.user_id,))
    rows: list[tuple] = cursor.fetchall()
    cursor.close()
    conn.close()

    response: str = "No bookings found."
    if len(rows) == 0:
        response = NO_BOOKING_MESSAGE
    else:
        column_names: list[str] = [column[0] for column in cursor.description]
        results: list[dict[str, Any]] = [
            dict(zip(column_names, row, strict=False)) for row in rows
        ]
        results_df: pd.DataFrame = pd.DataFrame(results)
        response = "Booked shows are:\n" + results_df.to_string(index=False)
    return response
