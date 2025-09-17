from dataclasses import dataclass


@dataclass
class ClassificationEvaluationEntry:
    id: int
    llm_run_completed: bool
    predicted_class: int | None = None
    repaired_json_record: str | None = None
    details: str | None = None  # additional details about the request
    num_requests: int | None = None
    request_tokens: int | None = None  # tokens used in processing requests.
    response_tokens: int | None = None  # tokens used in generating responses.
    # total tokens used in the whole run, should generally be equal to `request_tokens + response_tokens`.
    total_tokens: int | None = None
    time: int | None = None  # time taken for the run in seconds.
