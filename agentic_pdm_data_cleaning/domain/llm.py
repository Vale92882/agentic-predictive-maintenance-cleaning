from pydantic import BaseModel, Field


class LLM_Answer(BaseModel):
    record_id: str = Field(
        description="The ID of the maintenance record being analyzed.")
    human_error: bool = Field(
        description="Indicates if the record was corrupted by a human mistake."
    )
    repaired_json_record: str | None = Field(
        default=None,
        description="If the record was corrupted, this field contains the corrected version of the record in JSON format. If no correction is needed, this field is None."
    )
    details: str | None = Field(
        default="",
        description="A detailed explanation of the reasoning behind the classification, including any corrections made to the record."
    )


class LLM_Structured_Output(BaseModel):
    predicted_class: int = Field(
        description="Classification of the record"
    )
    repaired_json_record: str | None = Field(
        default=None,
        description="If the record was corrupted, this field contains the corrected version of the record in JSON format. If no correction is needed, this field is None."
    )
    details: str | None = Field(
        default="",
        description="A detailed explanation of the reasoning behind the classification, including any corrections made to the record."
    )


class Event_Time_Validation_Output(BaseModel):
    """Output model for event time validation experiments."""
    predicted_class: int = Field(
        description="The predicted class of the maintenance record."
    )
    start_date: str = Field(
        description="The start date of the faulty period."
    )
    end_date: str = Field(
        description="The corrected end date of the faulty period."
    )
    details: str = Field(
        default="",
        description="A detailed explanation of the reasoning behind the classification, including any corrections made to the record."
    )


class LLM_Config(BaseModel):
    """Configuration for the LLM model."""
    model_name: str = Field(
        default="qwen3:8b",
        description="The name of the LLM model to use, e.g., 'gpt-3.5-turbo' or 'google-gla:gemini-1.5-flash'."
    )
    base_url: str = Field(
        default="http://pc-ubix.uni.lux:11434/v1",
        description="The base URL for the LLM API. This is used to send requests to the LLM service."
    )
    model_settings: dict = Field(
        default={"temperature": 0.0},
        description="Settings for the LLM model, such as temperature and other parameters that control the model's behavior."
    )
